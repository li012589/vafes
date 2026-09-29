#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


KCAL_TO_KJ = 4.184
THRESHOLD_KJ_MOL = 5.0 * KCAL_TO_KJ
TEMPERATURE_KJ_MOL = 2.49
MODEL_LABELS = ("paper", "seed_3102", "seed_3103", "seed_3104")


def shift_minimum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Surface contains no finite values")
    return values - np.min(values[finite])


def read_plumed_fes(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", usecols=(0, 1, 2))
    cv1 = np.unique(data[:, 0])
    cv2 = np.unique(data[:, 1])
    if data.shape[0] != cv1.size * cv2.size:
        raise ValueError(f"Incomplete FES grid: {path}")
    i1 = np.searchsorted(cv1, data[:, 0])
    i2 = np.searchsorted(cv2, data[:, 1])
    surface = np.full((cv2.size, cv1.size), np.nan)
    surface[i2, i1] = data[:, 2]
    return cv1, cv2, surface


def bilinear_on_grid(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
) -> np.ndarray:
    ix = np.searchsorted(x_axis, x_target, side="right") - 1
    iy = np.searchsorted(y_axis, y_target, side="right") - 1
    valid = (
        (ix >= 0)
        & (iy >= 0)
        & (ix < len(x_axis) - 1)
        & (iy < len(y_axis) - 1)
    )
    ix = np.clip(ix, 0, len(x_axis) - 2)
    iy = np.clip(iy, 0, len(y_axis) - 2)
    tx = (x_target - x_axis[ix]) / (x_axis[ix + 1] - x_axis[ix])
    ty = (y_target - y_axis[iy]) / (y_axis[iy + 1] - y_axis[iy])
    result = (
        (1 - tx) * (1 - ty) * values[iy, ix]
        + tx * (1 - ty) * values[iy, ix + 1]
        + (1 - tx) * ty * values[iy + 1, ix]
        + tx * ty * values[iy + 1, ix + 1]
    )
    result[~valid] = np.nan
    return result


def mask_metric(
    first: np.ndarray,
    second: np.ndarray,
    domain: np.ndarray,
) -> float:
    first = shift_minimum(first)
    second = shift_minimum(second)
    mask = (
        domain
        & np.isfinite(first)
        & np.isfinite(second)
        & (first <= THRESHOLD_KJ_MOL)
        & (second <= THRESHOLD_KJ_MOL)
    )
    if not mask.any():
        raise RuntimeError("Empty 5-kcal/mol region")
    residual = first[mask] - second[mask]
    residual -= np.mean(residual)
    return float(np.sqrt(np.mean(residual**2)))


def write_matrix(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_vs_column", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkers", type=Path, required=True)
    for label in MODEL_LABELS:
        parser.add_argument(f"--{label.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dipeptide_fes_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    walker_paths = sorted(
        args.walkers.glob("w*/fes.dat"),
        key=lambda path: int(path.parent.name[1:]),
    )
    if len(walker_paths) != 16:
        raise RuntimeError(f"Expected 16 walkers, found {len(walker_paths)}")
    walker_labels = [path.parent.name for path in walker_paths]
    walker_surfaces = {}
    cv1 = cv2 = None
    for label, path in zip(walker_labels, walker_paths):
        this_cv1, this_cv2, surface = read_plumed_fes(path)
        if cv1 is None:
            cv1, cv2 = this_cv1, this_cv2
        elif not (np.array_equal(cv1, this_cv1) and np.array_equal(cv2, this_cv2)):
            raise RuntimeError(f"Walker grid mismatch: {path}")
        walker_surfaces[label] = shift_minimum(surface)
    walker_stack = np.stack([walker_surfaces[label] for label in walker_labels])
    meta_surface = np.nanmean(walker_stack, axis=0)
    grid1, grid2 = np.meshgrid(cv1, cv2, indexing="xy")

    model_surfaces = {}
    for label in MODEL_LABELS:
        path = getattr(args, label)
        data = np.load(path)
        if not np.isclose(float(data["temperature"]), TEMPERATURE_KJ_MOL):
            raise RuntimeError(f"Temperature mismatch: {path}")
        model_surfaces[label] = bilinear_on_grid(
            data["cv1"].astype(float),
            data["cv2"].astype(float),
            shift_minimum(data["fbar"].astype(float)),
            grid1,
            grid2,
        )

    comparison_domain = np.logical_and.reduce([
        np.isfinite(meta_surface),
        *[np.isfinite(surface) for surface in walker_surfaces.values()],
        *[np.isfinite(surface) for surface in model_surfaces.values()],
    ])
    if not comparison_domain.any():
        raise RuntimeError("Empty common finite comparison domain")

    model_rms = np.zeros((len(MODEL_LABELS), len(MODEL_LABELS)), dtype=float)
    for i, first_label in enumerate(MODEL_LABELS):
        for j, second_label in enumerate(MODEL_LABELS):
            rms = mask_metric(
                model_surfaces[first_label],
                model_surfaces[second_label],
                comparison_domain,
            )
            model_rms[i, j] = rms
    if not np.allclose(model_rms, model_rms.T, atol=1e-12):
        raise RuntimeError("RMS matrix is not symmetric")
    write_matrix(args.output / "rms_kj_mol.csv", list(MODEL_LABELS), model_rms)

    rms = mask_metric(model_surfaces["paper"], meta_surface, comparison_domain)
    write_rows(args.output / "vafes_vs_metad.csv", [{
        "rms_kj_mol": rms,
        "rms_kcal_mol": rms / KCAL_TO_KJ,
    }])

    loo_rows = []
    for i, label in enumerate(walker_labels):
        mean_other_15 = np.mean(np.delete(walker_stack, i, axis=0), axis=0)
        rms = mask_metric(
            mean_other_15,
            walker_stack[i],
            comparison_domain,
        )
        loo_rows.append({
            "held_out_walker": label,
            "rms_kj_mol": rms,
        })
    write_rows(args.output / "walker_leave_one_out_1vs15.csv", loo_rows)


if __name__ == "__main__":
    main()
