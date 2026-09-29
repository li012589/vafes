#!/usr/bin/env python3

import argparse
import json
import os
import random
import secrets
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scope import flow, source
from forceUtils.energy import energy
from dipeptideEnergy import charge, concise2full, functs, idxs, mass, params


def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_partial(path, axes, fbar, integrand_sd, mc_se, done, args):
    temporary = path + ".tmp.npz"
    np.savez(
        temporary,
        cv1=axes,
        cv2=axes,
        fbar=fbar,
        integrand_sd=integrand_sd,
        mc_se=mc_se,
        done=done,
        eval_batch=np.array(args.batch, dtype=np.int64),
        eval_seed=np.array(args.seed, dtype=np.int64),
        temperature=np.array(args.temperature, dtype=np.float64),
    )
    os.replace(temporary, path)


def load_or_initialize_partial(path, axes, args):
    n = args.bins * args.bins
    if os.path.exists(path):
        saved = np.load(path)
        expected = (n,)
        if (
            saved["fbar"].shape != expected
            or int(saved["eval_batch"]) != args.batch
            or int(saved["eval_seed"]) != args.seed
            or not np.allclose(saved["cv1"], axes)
            or not np.allclose(saved["cv2"], axes)
        ):
            raise RuntimeError(
                "existing partial evaluation is incompatible with this request; "
                "choose a new output name rather than mixing protocols"
            )
        return (
            saved["fbar"].copy(),
            saved["integrand_sd"].copy(),
            saved["mc_se"].copy(),
            saved["done"].astype(bool).copy(),
        )
    return (
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.zeros(n, dtype=bool),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", required=True, help="VaFES checkpoint directory")
    parser.add_argument("-device", type=int, default=0)
    parser.add_argument("-batch", type=int, default=8192,
                        help="conditional samples per native CV grid point")
    parser.add_argument("-seed", type=int, default=None,
                        help="base evaluation seed; point i uses seed+i")
    parser.add_argument("-bins", type=int, default=101)
    parser.add_argument("-lo", type=float, default=-0.13)
    parser.add_argument("-hi", type=float, default=0.13)
    parser.add_argument("-saveEvery", type=int, default=100)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31 - 1) + 1
    print(f"Using seed: {args.seed}", flush=True)

    if args.batch < 2:
        raise ValueError("batch must be at least two to estimate a standard error")
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")
    dtype = torch.float32

    with open(os.path.join(args.load, "parameter.json")) as f:
        config = json.load(f)
    args.temperature = float(config["T"])
    axes = np.linspace(args.lo, args.hi, args.bins)
    output = os.path.join(args.load, "validation_cv_fes.npz")
    partial = os.path.join(args.load, "validation_cv_fes.partial.npz")

    mass_dev = mass.to(device, dtype)
    charge_dev = charge.to(device, dtype)
    idxs_dev = [term.to(device) for term in idxs]
    params_dev = [term.to(device, dtype) for term in params]
    energy_fn = lambda q: energy(concise2full(q), mass_dev, charge_dev,
                                 functs, idxs_dev, params_dev)

    proj_v, proj_mean, _ranges = np.load(os.path.join(args.load, "projV.npz")).values()
    proj_v = torch.from_numpy(proj_v).to(device, dtype)
    proj_mean = torch.from_numpy(proj_mean).to(device, dtype)
    inv_proj_v = torch.linalg.inv(proj_v)
    temperature = torch.tensor(args.temperature, device=device, dtype=dtype)

    prior_param, transforms = torch.load(
        os.path.join(args.load, "best_TrainLoss_joint.saving"),
        map_location=device,
        weights_only=False,
    )
    for transform in transforms:
        transform.setdefault("eps", 1e-7)
        transform.setdefault("minLog", -50.0)
        transform.setdefault("linearBound", False)

    fbar, integrand_sd, mc_se, done = load_or_initialize_partial(partial, axes, args)
    cv1, cv2 = np.meshgrid(axes, axes, indexing="xy")
    grid = np.column_stack([cv1.ravel(), cv2.ravel()])
    unfinished = np.flatnonzero(~done)
    print(
        f"evaluating {args.load}: {len(unfinished)}/{len(grid)} grid points remain; "
        f"B={args.batch}, base seed={args.seed}",
        flush=True,
    )

    for count, idx in enumerate(unfinished, start=1):
        seed_torch(args.seed + int(idx))
        s = torch.tensor(grid[idx], device=device, dtype=dtype).reshape(1, 2)
        s = s.repeat(args.batch, 1)
        with torch.no_grad():
            z = source.Uniform.sample(args.batch, nvars=[31], T=temperature, **prior_param)
            zlogp = source.Uniform.logProbability(z, T=temperature, **prior_param)
            latent = torch.cat([s, z], dim=-1)
            transformed, log_det = source.TransformedDistribution.forward(
                latent,
                T=temperature,
                transformationList=[flow.SplineFlow],
                transformationParamList=transforms,
            )
            sample = transformed @ inv_proj_v + proj_mean
            h = zlogp - log_det + energy_fn(sample) / temperature
            sd = h.std(unbiased=True)
            fbar[idx] = (temperature * h.mean()).item()
            integrand_sd[idx] = (temperature * sd).item()
            mc_se[idx] = (temperature * sd / np.sqrt(args.batch)).item()
        done[idx] = True
        if count % args.saveEvery == 0 or count == len(unfinished):
            save_partial(partial, axes, fbar, integrand_sd, mc_se, done, args)
            print(f"completed {done.sum()}/{len(grid)} grid points", flush=True)

    if not done.all():
        raise RuntimeError("evaluation ended without completing all grid points")
    np.savez(
        output,
        cv1=axes,
        cv2=axes,
        fbar=fbar.reshape(args.bins, args.bins),
        integrand_sd=integrand_sd.reshape(args.bins, args.bins),
        mc_se=mc_se.reshape(args.bins, args.bins),
        eval_batch=np.array(args.batch, dtype=np.int64),
        eval_seed=np.array(args.seed, dtype=np.int64),
        temperature=np.array(args.temperature, dtype=np.float64),
    )
    os.remove(partial)


if __name__ == "__main__":
    main()
