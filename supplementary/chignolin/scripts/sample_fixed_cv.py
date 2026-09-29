#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
from pathlib import Path


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31 - 1) + 1
    print(f"Using seed: {args.seed}", flush=True)

    project_root = Path(__file__).resolve().parents[3]
    model_dir = args.model.resolve()
    args.points = args.points.resolve()
    out_dir = args.out.resolve()
    if is_within(out_dir, model_dir):
        raise RuntimeError("output directory must not be inside the source model directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    point = json.loads(args.points.read_text())["points"]["native_position"]
    target = out_dir / "samples_native_position.npz"
    meta = out_dir / "samples_native_position.json"
    if target.exists() or meta.exists():
        raise FileExistsError("refusing to overwrite native-position samples")

    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scope"))

    import numpy as np
    import torch

    from chignolinEnergy import ProteinConciseExpression, addHydrogen
    from nextForce.frontend import energy, fromOpenMM
    from scope import flow, source

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this production sampler")
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32

    config = json.loads((model_dir / "parameter.json").read_text())
    beta = torch.tensor(float(config["beta"]), device=device, dtype=dtype)
    beta_cv = bool(config.get("betaCV", False))

    helper = np.load(model_dir / "etc.npz")
    ref_heavy = torch.from_numpy(helper["refHeavy"]).to(device, dtype)
    ref_hydrogen = torch.from_numpy(helper["refHydrogen"]).to(device, dtype)
    hidx = torch.from_numpy(helper["Hidx"]).to(device)
    heavy_idx = torch.from_numpy(helper["heavyIdx"]).to(device)
    hs = torch.from_numpy(helper["Hs"]).to(device)
    idx_maj = torch.from_numpy(helper["idxMaj"]).to(device)

    checkpoint = model_dir / "best_TrainLoss_joint.saving"
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    prior_param, transformation_params = loaded[0], loaded[-1]
    transformation_list = [flow.SplineFlow]

    energy_params = fromOpenMM(
        ["amber14-all.xml", "implicit/gbn2.xml"],
        str(project_root / "etc" / "geoOpt.pdb"),
        eps=1e-5,
        device=device,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    coords_chunks = []
    energy_chunks = []
    for start in range(0, args.n_samples, args.batch):
        count = min(args.batch, args.n_samples - start)
        with torch.no_grad():
            cv = torch.tensor(
                [[float(point["x"]), float(point["y"])]],
                device=device,
                dtype=dtype,
            ).repeat(count, 1)
            beta_batch = beta.reshape(1, 1).repeat(count, 1)
            conditioning = (
                torch.cat([beta_batch, cv], dim=-1) if beta_cv else beta_batch
            )
            z_aux = source.Uniform.sample(
                count,
                nvars=[223],
                T=1,
                low=prior_param["low"],
                high=prior_param["high"],
            )
            z = torch.cat([cv, z_aux], dim=-1)
            intermediate, _ = source.TransformedDistribution.forward(
                z,
                T=conditioning,
                transformationList=transformation_list,
                transformationParamList=transformation_params,
            )
            heavy, _ = ProteinConciseExpression.inverse(
                intermediate.clone(), T=beta
            )
            full = addHydrogen(
                heavy, ref_heavy, ref_hydrogen, hidx, heavy_idx, hs, idx_maj
            )
            e = energy(full / 10, *energy_params)

        coords_chunks.append(full.detach().cpu().numpy().astype(np.float32))
        energy_chunks.append(e.detach().cpu().numpy().reshape(-1).astype(np.float32))
        print(f"{start + count}/{args.n_samples}", flush=True)

    coords = np.concatenate(coords_chunks)
    energies = np.concatenate(energy_chunks)
    cv_target = np.array([point["x"], point["y"]], dtype=np.float32)
    with target.open("xb") as handle:
        np.savez_compressed(
            handle,
            coords=coords,
            energies=energies,
            cv_target=cv_target,
        )
    meta.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_samples": args.n_samples,
                "cv_target": cv_target.tolist(),
            },
            indent=2,
        ) + "\n"
    )
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
