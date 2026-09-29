#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import secrets
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dimerTrain import DimerBondLength, dimerVacuumSymWall
from scope import flow, source


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_free_energy(r, temperature, angular_factor=math.pi / 2.0):
    r = np.asarray(r, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    potential = 4.0 * (1.0 - (r - 3.5) ** 2) ** 2
    return potential - temperature * (2.0 * np.log(r) + math.log(angular_factor))


def angular_factor_for_prior(low, high):
    low = np.atleast_1d(np.asarray(low, dtype=np.float64))
    high = np.atleast_1d(np.asarray(high, dtype=np.float64))
    if low.size == 1:
        low = np.repeat(low, 2)
    if high.size == 1:
        high = np.repeat(high, 2)
    phi_factor = math.asin(math.sqrt(high[0])) - math.asin(math.sqrt(low[0]))
    z_factor = math.sqrt(high[1]) - math.sqrt(low[1])
    return phi_factor * z_factor


def logmeanexp(values, axis=-1):
    values = np.asarray(values, dtype=np.float64)
    vmax = np.max(values, axis=axis, keepdims=True)
    result = np.log(np.mean(np.exp(values - vmax), axis=axis))
    return result + np.squeeze(vmax, axis=axis)


class DimerEvaluator:
    def __init__(self, checkpoint_dir, device, max_total_batch=131072):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.device = device
        self.max_total_batch = int(max_total_batch)
        checkpoint = self.checkpoint_dir / "best_TrainLoss_joint.saving"
        self.prior_param, self.transformation_param_list = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        self.transformation_param_list[0]["linearBound"] = False
        if "eps" not in self.transformation_param_list[0]:
            self.transformation_param_list[0]["eps"] = 0.0
        self.prior = source.Uniform
        self.transformation_list = [flow.SplineFlow]
        self.prior_low = float(self.prior_param["low"].detach().cpu().item())
        self.prior_high = float(self.prior_param["high"].detach().cpu().item())
        self.support_angular_factor = angular_factor_for_prior(
            self.prior_low, self.prior_high
        )

    def _sample_h(self, radii, temperatures, samples):
        radii = np.asarray(radii, dtype=np.float32)
        temperatures = np.asarray(temperatures, dtype=np.float32)
        n_conditions = len(radii)
        pieces = [[] for _ in range(n_conditions)]
        remaining = int(samples)

        while remaining:
            per_condition = min(
                remaining, max(1, self.max_total_batch // n_conditions)
            )
            r = torch.as_tensor(radii, device=self.device).repeat_interleave(
                per_condition
            )[:, None]
            t = torch.as_tensor(
                temperatures, device=self.device
            ).repeat_interleave(per_condition)[:, None]
            total = n_conditions * per_condition

            with torch.inference_mode():
                z_aux = self.prior.sample(total, nvars=[2], T=t, **self.prior_param)
                log_prior = self.prior.logProbability(
                    z_aux, T=t, **self.prior_param
                )
                z = torch.cat([r, z_aux], dim=-1)
                transformed, log_det_flow = source.TransformedDistribution.forward(
                    z,
                    T=t,
                    transformationList=self.transformation_list,
                    transformationParamList=self.transformation_param_list,
                )
                physical, log_det_inverse_cv = DimerBondLength.inverse(
                    transformed, T=t
                )
                energy = dimerVacuumSymWall(physical)
                h = log_prior - log_det_flow - log_det_inverse_cv + energy / t
                h = h.reshape(n_conditions, per_condition).detach().cpu().numpy()

            for idx in range(n_conditions):
                pieces[idx].append(h[idx].astype(np.float32, copy=False))
            remaining -= per_condition

        return np.stack([np.concatenate(item) for item in pieces], axis=0)

    def evaluate(self, radii, temperatures, samples, condition_block=8, raw=False):
        radii = np.asarray(radii, dtype=np.float64)
        temperatures = np.asarray(temperatures, dtype=np.float64)
        if radii.shape != temperatures.shape:
            raise ValueError("radii and temperatures must have matching shapes")

        result = {
            "fbar": np.empty(len(radii), dtype=np.float64),
            "mc_se": np.empty(len(radii), dtype=np.float64),
        }
        raw_h = [] if raw else None

        for start in range(0, len(radii), condition_block):
            stop = min(start + condition_block, len(radii))
            h = self._sample_h(radii[start:stop], temperatures[start:stop], samples)
            mean_h = h.mean(axis=1, dtype=np.float64)
            sd_h = h.std(axis=1, ddof=1, dtype=np.float64)
            result["fbar"][start:stop] = temperatures[start:stop] * mean_h
            result["mc_se"][start:stop] = (
                temperatures[start:stop] * sd_h / math.sqrt(samples)
            )
            if raw:
                raw_h.extend(h)

        if raw:
            result["h"] = np.stack(raw_h, axis=0)
        return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def curve_metrics(estimate, exact):
    estimate = np.asarray(estimate, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    reference_idx = int(np.argmin(exact))
    estimate_relative = estimate - estimate[reference_idx]
    exact_relative = exact - exact[reference_idx]
    shape_error = estimate_relative - exact_relative
    return {
        "shape_rmse": float(np.sqrt(np.mean(shape_error**2))),
        "shape_max_abs": float(np.max(np.abs(shape_error))),
    }


def run_dense_grid(args, evaluator, output):
    radii = np.linspace(args.r_min, args.r_max, args.r_points)
    temperatures = np.linspace(args.t_min, args.t_max, args.t_points)
    r_mesh, t_mesh = np.meshgrid(radii, temperatures)
    flat_r = r_mesh.ravel()
    flat_t = t_mesh.ravel()

    seed_everything(args.eval_seed)
    values = evaluator.evaluate(
        flat_r,
        flat_t,
        samples=args.grid_samples,
        condition_block=args.condition_block,
    )
    exact = exact_free_energy(
        flat_r, flat_t, evaluator.support_angular_factor
    )
    gap = values["fbar"] - exact

    rows = []
    for idx in range(len(flat_r)):
        rows.append(
            {
                "r": float(flat_r[idx]),
                "temperature": float(flat_t[idx]),
                "samples": args.grid_samples,
                "exact_f": float(exact[idx]),
                "fbar_hat": float(values["fbar"][idx]),
                "gap_fbar_minus_exact": float(gap[idx]),
                "mc_se_fbar": float(values["mc_se"][idx]),
            }
        )
    write_csv(output / "data" / "paper_dense_grid.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    gap_mesh = axes[0].pcolormesh(r_mesh, t_mesh, gap.reshape(t_mesh.shape), shading="auto", cmap="magma")
    fig.colorbar(gap_mesh, ax=axes[0], label=r"$\bar{F}-F$")
    axes[0].set(xlabel=r"Bond length $|\mathbf{x}|$", ylabel=r"Temperature $T$", title="Pointwise difference estimate")
    se_mesh = axes[1].pcolormesh(r_mesh, t_mesh, values["mc_se"].reshape(t_mesh.shape), shading="auto", cmap="viridis")
    fig.colorbar(se_mesh, ax=axes[1], label="Monte Carlo SE")
    axes[1].set(xlabel=r"Bond length $|\mathbf{x}|$", ylabel=r"Temperature $T$", title="Monte Carlo standard error")
    fig.savefig(output / "figures" / "paper_gap_and_mcse_heatmaps.png", dpi=220)
    plt.close(fig)


def evaluate_t1_model(args, checkpoint, label, samples):
    evaluator = DimerEvaluator(checkpoint, args.device_obj, args.max_total_batch)
    radii = np.linspace(args.profile_r_min, args.profile_r_max, args.profile_r_points)
    temperatures = np.ones_like(radii)
    seed_everything(args.eval_seed)
    values = evaluator.evaluate(
        radii,
        temperatures,
        samples=samples,
        condition_block=args.condition_block,
    )
    exact = exact_free_energy(radii, temperatures)
    return {
        "label": label,
        "radii": radii,
        "exact": exact,
        "fbar": values["fbar"],
        "mc_se": values["mc_se"],
        "metrics": curve_metrics(values["fbar"], exact),
    }


def run_t1_profiles(args, output):
    checkpoints = [Path(args.paper_checkpoint)] + [
        Path(checkpoint) for checkpoint in args.seed_checkpoints
    ]
    if len(checkpoints) != 6 or len({path.resolve() for path in checkpoints}) != 6:
        raise ValueError("T=1 profile requires six distinct checkpoints, including the paper model")
    paper = evaluate_t1_model(
        args, args.paper_checkpoint, "paper_checkpoint", args.profile_samples
    )
    seed_results = []
    for checkpoint in checkpoints[1:]:
        seed_results.append(
            evaluate_t1_model(
                args, checkpoint, checkpoint.name, args.profile_samples
            )
        )

    all_results = [paper] + seed_results
    rows = []
    for result in all_results:
        for idx, radius in enumerate(result["radii"]):
            rows.append(
                {
                    "model": result["label"],
                    "r": float(radius),
                    "temperature": 1.0,
                    "samples": args.profile_samples,
                    "exact_f": float(result["exact"][idx]),
                    "fbar_hat": float(result["fbar"][idx]),
                    "gap_fbar_minus_exact": float(
                        result["fbar"][idx] - result["exact"][idx]
                    ),
                    "mc_se_fbar": float(result["mc_se"][idx]),
                }
            )
    write_csv(output / "data" / "t1_profiles.csv", rows)

    metrics = {result["label"]: result["metrics"] for result in all_results}
    metric_names = list(paper["metrics"].keys())
    model_summary = {}
    for name in metric_names:
        if not isinstance(paper["metrics"][name], (float, int)):
            continue
        vals = np.array([item["metrics"][name] for item in all_results], dtype=float)
        model_summary[name] = {
            "mean": float(vals.mean()),
            "sample_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
        }
    with (output / "summary" / "t1_metrics.json").open("w") as handle:
        json.dump(
            {
                "n_models": len(all_results),
                "per_model": metrics,
                "all_model_summary": model_summary,
            },
            handle,
            indent=2,
        )

    radii = paper["radii"]
    reference_idx = int(np.argmin(paper["exact"]))
    exact_relative = paper["exact"] - paper["exact"][reference_idx]
    paper_relative = paper["fbar"] - paper["fbar"][reference_idx]
    seed_relative = np.stack(
        [item["fbar"] - item["fbar"][reference_idx] for item in seed_results]
    )
    seed_mean = seed_relative.mean(axis=0)
    seed_sd = seed_relative.std(axis=0, ddof=1)
    seed_min = seed_relative.min(axis=0)
    seed_max = seed_relative.max(axis=0)
    ensemble_label = "Independent-seed"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
        constrained_layout=True,
    )
    ax = axes[0]
    ax.fill_between(
        radii, seed_min, seed_max, color="#4C78A8", alpha=0.16,
        label=f"{ensemble_label} min--max",
    )
    ax.fill_between(
        radii, seed_mean - seed_sd, seed_mean + seed_sd,
        color="#4C78A8", alpha=0.26, label=fr"{ensemble_label} mean $\pm$ SD",
    )
    ax.plot(radii, seed_mean, color="#4C78A8", lw=2.0, label=f"{ensemble_label} mean")
    ax.plot(radii, paper_relative, color="#7A5195", lw=2.0, label="VaFES")
    ax.plot(radii, exact_relative, color="#E45756", lw=2.0, ls="--", label="Exact")
    ax.set(ylabel=r"Free energy ($k_B T$)", title=r"Dimer profile at $T=1$")
    ax.legend(frameon=False, fontsize=9, ncol=2)

    error_ax = axes[1]
    seed_absolute_error = np.abs(seed_relative - exact_relative[None, :])
    seed_error_mean = seed_absolute_error.mean(axis=0)
    seed_error_sd = seed_absolute_error.std(axis=0, ddof=1)
    paper_absolute_error = np.abs(paper_relative - exact_relative)
    error_ax.fill_between(
        radii,
        np.maximum(0.0, seed_error_mean - seed_error_sd),
        seed_error_mean + seed_error_sd,
        color="#4C78A8",
        alpha=0.26,
        label=fr"{ensemble_label} mean $|\delta F|\,\pm$ SD",
    )
    error_ax.plot(
        radii,
        seed_error_mean,
        color="#4C78A8",
        lw=1.8,
        label=fr"{ensemble_label} mean $|\delta F|$",
    )
    error_ax.plot(
        radii,
        paper_absolute_error,
        color="#7A5195",
        lw=1.8,
        label=r"VaFES $|\delta F|$",
    )
    error_ax.axhline(0.0, color="black", lw=1.0, ls="--")
    error_ax.set(
        xlabel=r"Bond length $|\mathbf{x}|$",
        ylabel=r"Absolute $\delta$ free energy ($k_B T$)",
    )
    error_ax.set_ylim(bottom=0.0)
    error_ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output / "figures" / "t1_multiseed_profile.png", dpi=220)
    plt.close(fig)


def stable_ess(log_weights):
    '''
    This script is used to compute the effective sample size (ESS). A high ESS could not rule out model collapse.
    '''
    log_weights = np.asarray(log_weights, dtype=np.float64)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    ess = weights.sum() ** 2 / np.square(weights).sum()
    return float(ess)


def iwae_rows(log_weights, temperature, support_exact, ks):
    log_weights = np.asarray(log_weights, dtype=np.float64)
    rows = []
    for k in ks:
        groups = len(log_weights) // k
        group_log_means = logmeanexp(
            log_weights[:groups * k].reshape(groups, k), axis=1
        )
        group_free_energies = -temperature * group_log_means
        sd = group_free_energies.std(ddof=1) if groups > 1 else float("nan")
        rows.append(
            {
                "k": int(k),
                "f_k": float(group_free_energies.mean()),
                "mc_se_f_k": float(sd / math.sqrt(groups)) if groups > 1 else float("nan"),
                "support_exact_f": float(support_exact),
                "gap_f_k_minus_support_exact": float(
                    group_free_energies.mean() - support_exact
                ),
            }
        )
    return rows


def run_importance(args, evaluator, output):
    representative_r = np.array([2.5, 3.5, 4.5], dtype=float)
    representative_t = np.ones_like(representative_r)
    support_exact = exact_free_energy(
        representative_r, representative_t, evaluator.support_angular_factor
    )
    seed_everything(args.eval_seed + 1)
    values = evaluator.evaluate(
        representative_r,
        representative_t,
        samples=args.importance_samples,
        condition_block=1,
        raw=True,
    )
    log_weights = -values["h"].astype(np.float64)
    ess_rows = []
    all_iwae_rows = []
    for idx, (radius, temperature) in enumerate(
        zip(representative_r, representative_t)
    ):
        logw = log_weights[idx]
        ess = stable_ess(logw)
        ess_rows.append(
            {
                "r": float(radius),
                "temperature": float(temperature),
                "samples": int(len(logw)),
                "ess": ess,
                "relative_ess": ess / len(logw),
            }
        )
        rows = iwae_rows(
            logw, temperature, support_exact[idx], args.iwae_ks
        )
        for row in rows:
            row = {"r": float(radius), "temperature": float(temperature), **row}
            all_iwae_rows.append(row)

    write_csv(output / "data" / "paper_ess.csv", ess_rows)
    write_csv(output / "data" / "paper_iwae.csv", all_iwae_rows)

    temperature = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    radius_colors = {2.5: "#4C78A8", 3.5: "#F58518", 4.5: "#54A24B"}
    for radius in [2.5, 3.5, 4.5]:
        rows = [
            row
            for row in all_iwae_rows
            if row["temperature"] == temperature
            and row["r"] == radius
            and row["k"] <= 128
        ]
        ax.errorbar(
            [row["k"] for row in rows],
            [row["gap_f_k_minus_support_exact"] for row in rows],
            yerr=[row["mc_se_f_k"] for row in rows],
            color=radius_colors[radius],
            marker="o",
            ms=3,
            lw=1.4,
            label=fr"$|\mathbf{{x}}|={radius}$",
        )
    ax.axhline(0.0, color="black", ls="--", lw=1, label="Exact lower bound")
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel=r"IWAE sample count $K$",
        ylabel=r"$\widehat{F}_K-F_{\rm exact}$",
        title=r"Dimer IWAE hierarchy at $T=1$",
    )
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.savefig(output / "figures" / "paper_iwae_hierarchy_t1.png", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-checkpoint", required=True)
    parser.add_argument("--seed-checkpoints", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--max-total-batch", type=int, default=131072)
    parser.add_argument("--condition-block", type=int, default=8)
    parser.add_argument("--r-min", type=float, default=1.0)
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--r-points", type=int, default=101)
    parser.add_argument("--t-min", type=float, default=0.3)
    parser.add_argument("--t-max", type=float, default=1.6)
    parser.add_argument("--t-points", type=int, default=27)
    parser.add_argument("--grid-samples", type=int, default=8192)
    parser.add_argument("--profile-r-min", type=float, default=2.0)
    parser.add_argument("--profile-r-max", type=float, default=5.0)
    parser.add_argument("--profile-r-points", type=int, default=121)
    parser.add_argument("--profile-samples", type=int, default=32768)
    parser.add_argument("--importance-samples", type=int, default=1048576)
    parser.add_argument(
        "--iwae-ks",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["grid", "profiles", "importance"],
        default=["grid", "profiles", "importance"],
    )
    args = parser.parse_args()
    if args.eval_seed is None:
        args.eval_seed = secrets.randbelow(2**31 - 1) + 1
    print(f"Using seed: {args.eval_seed}", flush=True)
    args.device_obj = (
        torch.device("cpu") if args.device < 0 else torch.device(f"cuda:{args.device}")
    )
    return args


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    for subdir in ["data", "figures", "summary"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    evaluator = DimerEvaluator(
        args.paper_checkpoint, args.device_obj, args.max_total_batch
    )

    if "grid" in args.tasks:
        run_dense_grid(args, evaluator, output)
    if "profiles" in args.tasks:
        run_t1_profiles(args, output)
    if "importance" in args.tasks:
        run_importance(args, evaluator, output)


if __name__ == "__main__":
    main()
