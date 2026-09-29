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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forceUtils.fourbody import periodicProperDihedral
from forceUtils.threebody import harmonicCosine
from forceUtils.twobody import coulombPair, fourthPowerBond
from h2n2Coordinate import energyCV
from h2n2CvTrain import SigmoidCoupling
from scope import flow, source


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def energy_parameters(device, dtype):
    mass = torch.tensor(
        [[[1.0080], [14.0067], [14.0067], [1.0080]]],
        device=device,
        dtype=dtype,
    )
    charge = torch.tensor(
        [[[0.350], [-0.350], [-0.350], [0.350]]],
        device=device,
        dtype=dtype,
    )
    functs = [fourthPowerBond, coulombPair, harmonicCosine, periodicProperDihedral]
    idxs = [
        torch.tensor([[0, 1], [1, 2], [2, 3]], device=device),
        torch.tensor([[0, 3]], device=device),
        torch.tensor([[0, 1, 2], [1, 2, 3]], device=device),
        torch.tensor([[0, 1, 2, 3]], device=device),
    ]
    params = [
        torch.tensor(
            [[2.2652e7, 0.1040], [2.0480e7, 0.1250], [2.2652e7, 0.1040]],
            device=device,
            dtype=dtype,
        ),
        torch.tensor([[138.935458]], device=device, dtype=dtype),
        torch.tensor(
            [
                [503.00, torch.deg2rad(torch.tensor(106.75))],
                [503.00, torch.deg2rad(torch.tensor(106.75))],
            ],
            device=device,
            dtype=dtype,
        ),
        torch.tensor(
            [[41.80, 2, torch.deg2rad(torch.tensor(180.0))]],
            device=device,
            dtype=dtype,
        ),
    ]
    return mass, charge, functs, idxs, params


def load_paths(path_dir):
    path_dir = Path(path_dir)
    with (path_dir / "NEBpath.npy").open("rb") as handle:
        paths = np.load(handle)
        _ = np.load(handle)
        _ = np.load(handle)
    with (path_dir / "NEBpathNaive.npy").open("rb") as handle:
        inversion = np.load(handle)
        _ = np.load(handle)
        _ = np.load(handle)
    return {
        "torsion": {"selector": "fig3b", "points": paths[-1]},
        "inversion": {"selector": "fig3c", "points": inversion},
    }


def load_order65(exact_dir, selector):
    path = Path(exact_dir) / f"h2n2_exact_{selector}_order65.csv"
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows[int(row["index"])] = float(row["exact"])
    return rows


class Evaluator:
    def __init__(self, checkpoint, cv_checkpoint, device, max_batch):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.max_batch = max_batch
        self.dtype = torch.float32
        self.prior_param, self.transformation_param_list = torch.load(
            self.checkpoint / "best_TrainLoss_joint.saving",
            map_location=device,
            weights_only=False,
        )
        for params in self.transformation_param_list:
            params.setdefault("eps", 1e-7)
            params.setdefault("minLog", -50.0)
            params.setdefault("linearBound", False)
        self.cv_params = torch.load(
            Path(cv_checkpoint) / "best_TrainLoss_joint.saving",
            map_location=device,
            weights_only=False,
        )[0]
        self.mass, self.charge, self.functs, self.idxs, self.params = (
            energy_parameters(device, self.dtype)
        )

    def sample_h(self, points, samples):
        points = np.asarray(points, dtype=np.float32)
        results = []
        for point in points:
            chunks = []
            remaining = int(samples)
            while remaining:
                batch = min(remaining, self.max_batch)
                cv = torch.as_tensor(point, device=self.device).repeat(batch, 1)
                with torch.inference_mode():
                    z = source.Uniform.sample(
                        batch, nvars=[4], T=1.0, **self.prior_param
                    )
                    log_prior = source.Uniform.logProbability(
                        z, T=1.0, **self.prior_param
                    )
                    z = torch.cat([z, cv], dim=-1)
                    transformed, log_det = source.TransformedDistribution.forward(
                        z,
                        T=1.0,
                        transformationList=[flow.SplineFlow],
                        transformationParamList=self.transformation_param_list,
                    )
                    physical, inverse_log_det = (
                        source.TransformedDistribution.inverse(
                            transformed,
                            T=1.0,
                            transformationList=[SigmoidCoupling],
                            transformationParamList=self.cv_params,
                        )
                    )
                    energy = energyCV(
                        physical,
                        self.mass,
                        self.charge,
                        self.functs,
                        self.idxs,
                        self.params,
                    )
                    h = log_prior - log_det - inverse_log_det + energy
                chunks.append(h.flatten().detach().cpu().numpy().astype(np.float32))
                remaining -= batch
            results.append(np.concatenate(chunks))
        return results

    def evaluate(self, points, samples, seed):
        seed_everything(seed)
        h_arrays = self.sample_h(points, samples)
        means = np.array([h.mean(dtype=np.float64) for h in h_arrays])
        sds = np.array([h.std(ddof=1, dtype=np.float64) for h in h_arrays])
        return means, sds / math.sqrt(samples)


def metrics(estimate, exact, se):
    estimate_rel = estimate - estimate[0]
    exact_rel = exact - exact[0]
    exact_peak = int(np.argmax(exact))
    estimate_peak = int(np.argmax(estimate))
    exact_forward = exact[exact_peak] - exact[0]
    estimate_forward = estimate[estimate_peak] - estimate[0]
    exact_reverse = exact[exact_peak] - exact[-1]
    estimate_reverse = estimate[estimate_peak] - estimate[-1]
    return {
        "max_trans_referenced_abs_error": float(np.abs(estimate_rel - exact_rel).max()),
        "exact_trans_to_peak": float(exact_forward),
        "estimate_trans_to_peak": float(estimate_forward),
        "trans_barrier_abs_error": float(abs(estimate_forward - exact_forward)),
        "trans_barrier_mc_se": float(
            math.sqrt(se[estimate_peak] ** 2 + se[0] ** 2)
        ),
        "exact_cis_to_peak": float(exact_reverse),
        "estimate_cis_to_peak": float(estimate_reverse),
        "cis_barrier_abs_error": float(abs(estimate_reverse - exact_reverse)),
        "cis_barrier_mc_se": float(
            math.sqrt(se[estimate_peak] ** 2 + se[-1] ** 2)
        ),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(output, mechanisms, profiles):
    paper_color = "#7A5195"
    seed_color = "#4C78A8"
    exact_color = "#E45756"
    zero_color = "#222222"
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for mechanism, info in mechanisms.items():
        rows = profiles[mechanism]
        paper = rows["paper"]
        seeds = [value for key, value in rows.items() if key != "paper"]
        points = info["points"]
        s1 = points[:, 0]
        exact_rel = paper["exact"] - paper["exact"][0]
        paper_rel = paper["estimate"] - paper["estimate"][0]
        seed_rel = np.stack(
            [item["estimate"] - item["estimate"][0] for item in seeds]
        )
        seed_mean = seed_rel.mean(axis=0)
        seed_sd = seed_rel.std(axis=0, ddof=1)
        seed_min = seed_rel.min(axis=0)
        seed_max = seed_rel.max(axis=0)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 7.0),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
            constrained_layout=True,
        )
        axes[0].fill_between(
            s1,
            seed_min,
            seed_max,
            color=seed_color,
            alpha=0.12,
            label="Independent-seed min--max",
        )
        axes[0].fill_between(
            s1,
            seed_mean - seed_sd,
            seed_mean + seed_sd,
            color=seed_color,
            alpha=0.25,
            label=r"Independent-seed mean $\pm$ SD",
        )
        axes[0].plot(
            s1, seed_mean, color=seed_color, lw=1.8, label="Independent-seed mean"
        )
        axes[0].plot(
            s1, paper_rel, color=paper_color, lw=2, label="VaFES"
        )
        axes[0].plot(
            s1, exact_rel, "--", color=exact_color, lw=2, label="Numerical integration"
        )
        axes[0].set(
            ylabel="Free energy (kJ/mol)",
            title=f"H$_2$N$_2$ {mechanism} path",
        )
        axes[0].legend(frameon=False, fontsize=9)
        axes[1].plot(
            s1,
            np.abs(paper_rel - exact_rel),
            color=paper_color,
            lw=1.8,
            label=r"VaFES $|\delta F|$",
        )
        seed_error = np.abs(seed_rel - exact_rel[None, :])
        seed_error_mean = seed_error.mean(axis=0)
        seed_error_sd = seed_error.std(axis=0, ddof=1)
        axes[1].plot(
            s1,
            seed_error_mean,
            color=seed_color,
            lw=1.8,
            label=r"Independent-seed mean $|\delta F|$",
        )
        axes[1].fill_between(
            s1,
            np.maximum(0.0, seed_error_mean - seed_error_sd),
            seed_error_mean + seed_error_sd,
            color=seed_color,
            alpha=0.25,
            label=r"Independent-seed mean $|\delta F|\,\pm$ SD",
        )
        axes[1].axhline(0, color=zero_color, ls="--", lw=1)
        axes[1].set(
            xlabel=r"Machine-learning CV $s_1$",
            ylabel=r"Absolute $\delta$ free energy (kJ/mol)",
        )
        axes[1].set_ylim(bottom=0.0)
        axes[1].legend(frameon=False, fontsize=8)
        fig.savefig(figure_dir / f"{mechanism}_path_validation.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--cv", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--exact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--paper-samples", type=int, default=262144)
    parser.add_argument("--seed-samples", type=int, default=65536)
    parser.add_argument("--max-batch", type=int, default=65536)
    parser.add_argument("--eval-seed", type=int, default=None)
    args = parser.parse_args()
    if args.eval_seed is None:
        args.eval_seed = secrets.randbelow(2**31 - 1) + 1
    print(f"Using seed: {args.eval_seed}", flush=True)

    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    output = Path(args.output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "summary").mkdir(parents=True, exist_ok=True)
    mechanisms = load_paths(args.paths)
    models = [("paper", None, Path(args.paper), args.paper_samples)]
    for seed_path in args.seeds:
        path = Path(seed_path)
        seed = int(path.name.split("_")[-1])
        models.append((path.name, seed, path, args.seed_samples))

    config = vars(args).copy()
    config.update(
        {
            "paper": Path(args.paper).name,
            "seeds": [Path(path).name for path in args.seeds],
            "cv": Path(args.cv).name,
            "paths": Path(args.paths).name,
            "exact": Path(args.exact).name,
            "output": output.name,
        }
    )
    with (output / "run_config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    all_rows = []
    paper_metrics = {}
    profile_cache = {name: {} for name in mechanisms}
    for model_index, (label, train_seed, checkpoint, samples) in enumerate(models):
        evaluator = Evaluator(checkpoint, args.cv, device, args.max_batch)
        for mechanism_index, (mechanism, info) in enumerate(mechanisms.items()):
            points = info["points"]
            exact_map = load_order65(args.exact, info["selector"])
            exact = np.array([exact_map[idx] for idx in range(len(points))])
            means, ses = evaluator.evaluate(
                points,
                samples,
                args.eval_seed + model_index * 100 + mechanism_index,
            )
            profile_cache[mechanism][label] = {
                "estimate": means,
                "se": ses,
                "exact": exact,
            }
            if label == "paper":
                paper_metrics[mechanism] = metrics(means, exact, ses)
            for idx, point in enumerate(points):
                all_rows.append(
                    {
                        "model": label,
                        "training_seed": "" if train_seed is None else train_seed,
                        "mechanism": mechanism,
                        "selector": info["selector"],
                        "index": idx,
                        "s1": float(point[0]),
                        "z2": float(point[1]),
                        "samples": samples,
                        "exact_order65": float(exact[idx]),
                        "fbar_hat": float(means[idx]),
                        "mc_se": float(ses[idx]),
                        "exact_trans_referenced": float(exact[idx] - exact[0]),
                        "fbar_trans_referenced": float(means[idx] - means[0]),
                    }
                )

    write_csv(output / "data" / "path_profiles.csv", all_rows)
    with (output / "summary" / "metrics.json").open("w") as handle:
        json.dump(paper_metrics, handle, indent=2)
    make_figures(output, mechanisms, profile_cache)


if __name__ == "__main__":
    main()
