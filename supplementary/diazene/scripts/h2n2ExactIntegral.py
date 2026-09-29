import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2n2Coordinate import energyCV
from h2n2CvTrain import SigmoidCoupling
from scope import source

from forceUtils.twobody import fourthPowerBond, coulombPair
from forceUtils.threebody import harmonicCosine
from forceUtils.fourbody import periodicProperDihedral


def resolve_path(path, base_dir=None):
    if path is None or os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.abspath(os.path.join(base_dir, path))


def default_cv_path(load_dir):
    parameter_path = os.path.join(load_dir, "parameter.json")
    if not os.path.exists(parameter_path):
        return None
    with open(parameter_path, "r") as f:
        params = json.load(f)
    load_cv = params.get("loadCV")
    if load_cv is None:
        return None
    candidate = resolve_path(load_cv)
    if os.path.exists(candidate):
        return candidate

    return candidate


def load_path(load_dir, curve):
    if curve == "fig3b":
        path_file = os.path.join(load_dir, "NEBpath.npy")
        with open(path_file, "rb") as f:
            path_list = np.load(f)
            value_list = np.load(f)
            _ = np.load(f)
        return path_list[-1], value_list[-1], path_file
    if curve == "fig3c":
        path_file = os.path.join(load_dir, "NEBpathNaive.npy")
        with open(path_file, "rb") as f:
            path = np.load(f)
            values = np.load(f)
            _ = np.load(f)
        return path, values, path_file
    raise ValueError(f"Unknown curve: {curve}")


def h2n2_energy_parameters(device, dtype):
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
            [
                [2.2652e7, 0.1040],
                [2.0480e7, 0.1250],
                [2.2652e7, 0.1040],
            ],
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


def logaddexp_scalar(left, right):
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


class H2N2ExactIntegrator:
    def __init__(
        self,
        cv_params,
        order,
        device,
        dtype,
        outer_chunk,
        ranges=None,
    ):
        self.cv_params = cv_params
        self.order = order
        self.device = device
        self.dtype = dtype
        self.outer_chunk = outer_chunk
        self.mass, self.charge, self.functs, self.idxs, self.params = h2n2_energy_parameters(
            device, dtype
        )

        if ranges is None:
            ranges = np.array(
                [
                    [-0.15, 0.15],
                    [1e-5, 0.18],
                    [0.0, 0.20],
                    [-0.15, 0.15],
                ],
                dtype=np.float64,
            )
        self.ranges = ranges

        nodes, weights = leggauss(order)
        self.nodes = [
            (high + low) / 2.0 + (high - low) / 2.0 * nodes
            for low, high in self.ranges
        ]
        self.log_weights = [
            np.log(weights) + math.log((high - low) / 2.0)
            for low, high in self.ranges
        ]

        x2_grid, d_grid = np.meshgrid(self.nodes[3], self.nodes[2], indexing="xy")
        x2_logw, d_logw = np.meshgrid(
            self.log_weights[3],
            self.log_weights[2],
            indexing="xy",
        )
        self.inner_d = d_grid.reshape(-1).astype(np.float32)
        self.inner_x2 = x2_grid.reshape(-1).astype(np.float32)
        self.inner_logw = (d_logw.reshape(-1) + x2_logw.reshape(-1)).astype(np.float64)
        self.inner_size = self.inner_d.shape[0]

        outer_x1, outer_y1 = np.meshgrid(self.nodes[0], self.nodes[1], indexing="ij")
        outer_x1_logw, outer_y1_logw = np.meshgrid(
            self.log_weights[0],
            self.log_weights[1],
            indexing="ij",
        )
        self.outer_x1 = outer_x1.reshape(-1).astype(np.float32)
        self.outer_y1 = outer_y1.reshape(-1).astype(np.float32)
        self.outer_logw = (outer_x1_logw.reshape(-1) + outer_y1_logw.reshape(-1)).astype(
            np.float64
        )

    @property
    def evaluations_per_point(self):
        return self.order**4

    def integrate_point(self, s1, z2):
        log_z = -math.inf
        start_time = time.time()

        for start in range(0, self.outer_x1.shape[0], self.outer_chunk):
            stop = min(start + self.outer_chunk, self.outer_x1.shape[0])
            outer_count = stop - start
            row_count = outer_count * self.inner_size

            config = np.empty((row_count, 6), dtype=np.float32)
            log_weight = np.empty(row_count, dtype=np.float64)

            for local, outer_idx in enumerate(range(start, stop)):
                row_start = local * self.inner_size
                row_stop = row_start + self.inner_size
                config[row_start:row_stop, 0] = self.outer_x1[outer_idx]
                config[row_start:row_stop, 1] = self.outer_y1[outer_idx]
                config[row_start:row_stop, 2] = self.inner_d
                config[row_start:row_stop, 3] = self.inner_x2
                config[row_start:row_stop, 4] = s1
                config[row_start:row_stop, 5] = z2
                log_weight[row_start:row_stop] = self.outer_logw[outer_idx] + self.inner_logw

            config_tensor = torch.from_numpy(config).to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                config_y, inverse_log_det = source.TransformedDistribution.inverse(
                    config_tensor,
                    T=1,
                    transformationList=[SigmoidCoupling],
                    transformationParamList=self.cv_params,
                )
                energy = energyCV(
                    config_y,
                    self.mass,
                    self.charge,
                    self.functs,
                    self.idxs,
                    self.params,
                ).reshape(-1)

            term = (
                -energy.double()
                + inverse_log_det.reshape(-1).double()
                + torch.from_numpy(log_weight).to(torch.float64)
            ).cpu()
            log_z = logaddexp_scalar(log_z, torch.logsumexp(term, dim=0).item())

        exact_free_energy = -log_z
        return exact_free_energy, time.time() - start_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-load", required=True, help="Path to the H2N2 VaFES result folder.")
    parser.add_argument("-loadCV", default=None, help="Path to the trained H2N2 CV model.")
    parser.add_argument(
        "-curve",
        default="fig3b",
        choices=["fig3b", "fig3c"],
        help="Which saved path to integrate.",
    )
    parser.add_argument("-order", type=int, default=65, help="Gauss-Legendre order per dimension.")
    parser.add_argument("-outerChunk", type=int, default=8, help="Number of (x1,y1) nodes per Torch batch.")
    parser.add_argument("-out", default=None, help="Output directory. Defaults to <load>/exactIntegral.")
    parser.add_argument("-device", type=int, default=-1, help="-1 CPU, -2 MPS, or CUDA device index.")
    args = parser.parse_args()

    load_dir = resolve_path(args.load)
    cv_dir = resolve_path(args.loadCV) if args.loadCV is not None else default_cv_path(load_dir)
    if cv_dir is None or not os.path.exists(cv_dir):
        raise FileNotFoundError(
            "Could not resolve the CV model path. Pass -loadCV explicitly."
        )

    if args.device == -1:
        device = torch.device("cpu")
    elif args.device == -2:
        device = torch.device("mps")
    else:
        device = torch.device(f"cuda:{args.device}")
    dtype = torch.float32

    out_dir = resolve_path(args.out) if args.out is not None else os.path.join(load_dir, "exactIntegral")
    os.makedirs(out_dir, exist_ok=True)

    path, vafes_values, path_file = load_path(load_dir, args.curve)

    cv_params = torch.load(
        os.path.join(cv_dir, "best_TrainLoss_joint.saving"),
        map_location=device,
        weights_only=False,
    )[0]
    integrator = H2N2ExactIntegrator(
        cv_params=cv_params,
        order=args.order,
        device=device,
        dtype=dtype,
        outer_chunk=args.outerChunk,
    )

    rows = []
    print(f"load: {load_dir}")
    print(f"loadCV: {cv_dir}")
    print(f"path: {path_file}")
    print(f"curve: {args.curve}")
    print(f"order: {args.order}, evals/point: {integrator.evaluations_per_point}")

    for idx, point in enumerate(path):
        s1 = float(point[0])
        z2 = float(point[1])
        vafes = float(vafes_values[idx])
        exact, seconds = integrator.integrate_point(s1, z2)
        diff = exact - vafes
        row = {
            "index": idx,
            "s1": s1,
            "z2": z2,
            "vafes": vafes,
            "exact": exact,
            "diff": diff,
            "abs_diff": abs(diff),
            "order": args.order,
            "evals": integrator.evaluations_per_point,
            "seconds": seconds,
        }
        rows.append(row)
        print(
            f"[{idx + 1:03d}/{len(path):03d}] "
            f"idx={idx:03d} s1={s1:.6f} z2={z2:.6f} "
            f"VaFES={vafes:.6f} exact={exact:.6f} diff={diff:+.6f} "
            f"time={seconds:.2f}s",
            flush=True,
        )

        np.save(
            os.path.join(out_dir, f"h2n2_exact_{args.curve}_order{args.order}.npy"),
            np.array(rows, dtype=object),
            allow_pickle=True,
        )

    csv_path = os.path.join(out_dir, f"h2n2_exact_{args.curve}_order{args.order}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    diffs = np.array([row["diff"] for row in rows], dtype=np.float64)
    print("summary:")
    print(f"  points: {len(rows)}")
    print(f"  mean diff: {diffs.mean():+.6f}")
    print(f"  std diff: {diffs.std():.6f}")
    print(f"  mean abs diff: {np.abs(diffs).mean():.6f}")
    print(f"  max abs diff: {np.abs(diffs).max():.6f}")
    print(f"  csv: {csv_path}")


if __name__ == "__main__":
    main()
