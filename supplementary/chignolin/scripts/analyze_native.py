#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

HBOND_DA_CUTOFF_A = 3.5
HBOND_ANGLE_CUTOFF_DEG = 120.0
KEY_PAIRS = [
    ("Asp3 N-H...Thr8 O", 3, 8),
    ("Gly7 N-H...Asp3 O", 7, 3),
]


def parse_pdb(path: Path) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    models = []
    identities = []
    coords = []
    atoms = []
    saw_model = False
    with path.open() as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                saw_model = True
                coords, atoms = [], []
            elif record == "ENDMDL":
                models.append(coords)
                identities.append(atoms)
                coords, atoms = [], []
            elif record in {"ATOM", "HETATM"}:
                if line[16] not in {" ", "A"}:
                    continue
                atoms.append((line[12:16].strip(), line[17:20].strip(), int(line[22:26])))
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not saw_model:
        models, identities = [coords], [atoms]
    array = np.asarray(models, dtype=np.float64)
    if array.ndim != 3 or any(ids != identities[0] for ids in identities[1:]):
        raise RuntimeError(f"Invalid PDB models: {path}")
    return array, identities[0]


def reference_in_generated_order(
    reference: np.ndarray,
    ref_identity: list[tuple[str, str, int]],
    generated_identity: list[tuple[str, str, int]],
) -> np.ndarray:
    def atom_key(atom: tuple[str, str, int]) -> tuple[int, str, str]:
        name, resname, resid = atom
        return resid, resname, "H" if resid == 1 and name == "H1" else name

    lookup = {atom_key(atom): i for i, atom in enumerate(ref_identity)}
    return reference[:, [lookup[atom_key(atom)] for atom in generated_identity]]


def atom_index(identity: list[tuple[str, str, int]], resid: int, name: str) -> int | None:
    for i, (atom_name, _resname, atom_resid) in enumerate(identity):
        if atom_resid == resid and atom_name == name:
            return i
    return None


def angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    cosine = np.sum(first * second, axis=-1) / np.clip(denominator, 1e-12, None)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rmsd_matrix(samples: np.ndarray, references: np.ndarray, ca_indices: list[int]) -> np.ndarray:
    sample_ca = samples[:, ca_indices]
    reference_ca = references[:, ca_indices]
    columns = []
    for reference in reference_ca:
        p = sample_ca - sample_ca.mean(axis=1, keepdims=True)
        q = reference - reference.mean(axis=0, keepdims=True)
        covariance = np.einsum("nai,aj->nij", p, q)
        u, _singular, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        aligned = np.einsum("nai,nij->naj", p, rotation)
        columns.append(np.sqrt(np.mean(np.sum((aligned - q[None]) ** 2, axis=2), axis=1)))
    return np.column_stack(columns)


def metric_summary(values: np.ndarray) -> dict[str, float]:
    result = {
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }
    for cutoff in (1.0, 1.5, 2.0):
        result[f"fraction_lt_{str(cutoff).replace('.', 'p')}"] = float(np.mean(values < cutoff))
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def single_config_summary(values: np.ndarray) -> dict[str, object]:
    return {
        "rmsd_min_A": float(values.min()),
        "rmsd_max_A": float(values.max()),
        "n_nmr_models_below_1p5_A": int(np.sum(values < 1.5)),
    }


def hbond_occupancy(
    coords: np.ndarray,
    identity: list[tuple[str, str, int]],
    donor: int,
    acceptor: int,
) -> float:
    n_idx = atom_index(identity, donor, "N")
    h_idx = atom_index(identity, donor, "H")
    o_idx = atom_index(identity, acceptor, "O")
    if n_idx is None or h_idx is None or o_idx is None:
        raise KeyError((donor, acceptor, n_idx, h_idx, o_idx))
    distance = np.linalg.norm(coords[:, n_idx] - coords[:, o_idx], axis=1)
    angle = angle_deg(
        coords[:, n_idx] - coords[:, h_idx],
        coords[:, o_idx] - coords[:, h_idx],
    )
    return float(
        ((distance < HBOND_DA_CUTOFF_A) & (angle > HBOND_ANGLE_CUTOFF_DEG)).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True, type=Path)
    parser.add_argument("--metadata", nargs="+", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--geoopt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if len(args.samples) != len(args.metadata):
        raise ValueError("--samples and --metadata must have equal lengths")
    args.out.mkdir(parents=True, exist_ok=True)
    final_json = args.out / "revision_native_summary.json"
    if final_json.exists():
        raise FileExistsError(f"refusing to overwrite {final_json}")

    refs_raw, reference_identity = parse_pdb(args.reference)
    geo_raw, generated_identity = parse_pdb(args.geoopt)
    if geo_raw.shape[0] != 1:
        raise RuntimeError("geoOpt template must contain exactly one model")
    refs = reference_in_generated_order(
        refs_raw, reference_identity, generated_identity
    )
    if refs.shape != (18, 138, 3):
        raise RuntimeError(f"expected 18x138x3 references, got {refs.shape}")
    ca_indices = [
        i for i, (name, _residue, _resid) in enumerate(generated_identity)
        if name == "CA"
    ]
    if len(ca_indices) != 10:
        raise RuntimeError(f"expected 10 C-alpha atoms, found {len(ca_indices)}")
    core_ca = ca_indices[1:9]

    coords_batches: list[np.ndarray] = []
    energy_batches: list[np.ndarray] = []
    expected_cv: np.ndarray | None = None
    for sample_path, metadata_path in zip(args.samples, args.metadata):
        metadata = json.loads(metadata_path.read_text())
        with np.load(sample_path) as sample:
            coords = np.asarray(sample["coords"], dtype=np.float64)
            energies = np.asarray(sample["energies"], dtype=np.float64).reshape(-1)
            cv_target = np.asarray(sample["cv_target"], dtype=np.float64)
        if coords.shape[0] != energies.size:
            raise RuntimeError(f"coordinate/energy mismatch in {sample_path}")
        if coords.shape[1:] != (138, 3):
            raise RuntimeError(f"unexpected coordinates in {sample_path}: {coords.shape}")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(energies)):
            raise RuntimeError(f"nonfinite sample values in {sample_path}")
        if coords.shape[0] != int(metadata["n_samples"]):
            raise RuntimeError(f"sample count mismatch in {sample_path}")
        if not np.array_equal(cv_target, np.asarray(metadata["cv_target"])):
            raise RuntimeError(f"metadata CV target mismatch in {sample_path}")
        if expected_cv is None:
            expected_cv = cv_target
        elif not np.array_equal(cv_target, expected_cv):
            raise RuntimeError(f"CV target mismatch in {sample_path}")
        coords_batches.append(coords)
        energy_batches.append(energies)

    coords_all = np.concatenate(coords_batches, axis=0)
    energies_all = np.concatenate(energy_batches, axis=0)
    core_matrix = rmsd_matrix(coords_all, refs, core_ca)
    nearest_core = np.min(core_matrix, axis=1)
    representative_index = int(np.argmin(energies_all))
    representative_core = core_matrix[representative_index]

    hbond_rows = [
        {
            "label": label,
            "occupancy": hbond_occupancy(
                coords_all, generated_identity, donor, acceptor
            ),
            "pdb_model_fraction": hbond_occupancy(
                refs, generated_identity, donor, acceptor
            ),
        }
        for label, donor, acceptor in KEY_PAIRS
    ]

    selected_rows = []
    for model in range(18):
        selected_rows.append(
            {
                "nmr_model": model + 1,
                "core_residues_2_9_calpha_rmsd_A": float(representative_core[model]),
            }
        )
    write_csv(args.out / "representative_rmsd_vs_18_nmr.csv", selected_rows)
    write_csv(args.out / "key_hbond_occupancies.csv", hbond_rows)
    per_model_rows = []
    for model in range(18):
        per_model_rows.append(
            {
                "nmr_model": model + 1,
                **{
                    f"core_{key}": value
                    for key, value in metric_summary(core_matrix[:, model]).items()
                },
            }
        )
    write_csv(args.out / "ensemble_rmsd_per_nmr_model.csv", per_model_rows)

    summary = {
        "selection": {
            "cv_fixed_before_sampling_A": expected_cv.tolist(),
            "n_samples_total": int(coords_all.shape[0]),
        },
        "representative_configuration": {
            "core_residues_2_9_calpha": single_config_summary(representative_core),
        },
        "unfiltered_ensemble": {
            "nearest_nmr_fraction_lt_1p0_A": float(np.mean(nearest_core < 1.0)),
            "nearest_nmr_fraction_lt_1p5_A": float(np.mean(nearest_core < 1.5)),
        },
        "key_hbond_occupancies": hbond_rows,
    }
    final_json.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)
    fig.suptitle(
        "Native-state samples evaluated against each 1UAO model "
        f"(N={coords_all.shape[0]:,})"
    )
    models = np.arange(1, 19)
    medians = np.array([row["core_median"] for row in per_model_rows])
    lows = np.array([row["core_p05"] for row in per_model_rows])
    highs = np.array([row["core_p95"] for row in per_model_rows])
    axes[0].errorbar(
        models,
        medians,
        yerr=[medians - lows, highs - medians],
        fmt="o",
        color="#4C78A8",
        capsize=3,
        label="Ensemble median and 5-95 percentile",
    )
    axes[0].plot(
        models,
        representative_core,
        "D-",
        color="#54A24B",
        label="Minimum-energy representative configuration",
    )
    axes[0].axhline(1.5, color="#F58518", linestyle="--", label="1.5 A")
    axes[0].axhline(2.0, color="#7A5195", linestyle="--", label="2.0 A")
    axes[0].set_ylabel("Residues 2-9 C$_\\alpha$-RMSD (A)\nmedian and 5-95 percentile")
    axes[0].set_ylim(0.75, 2.05)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)

    for key, cutoff, color in (
        ("1p0", 1.0, "#E45756"),
        ("1p5", 1.5, "#F58518"),
        ("2p0", 2.0, "#7A5195"),
    ):
        axes[1].plot(
            models,
            [row[f"core_fraction_lt_{key}"] for row in per_model_rows],
            "o-",
            color=color,
            label=f"RMSD < {cutoff:.1f} A",
        )
    axes[1].set_xticks(models)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[1].set_xlabel("1UAO NMR model")
    axes[1].set_ylabel("Fraction of samples")
    axes[1].legend(frameon=False, ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out / "chignolinFoldedStructureValidation.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary["selection"], indent=2))
    print(json.dumps(summary["representative_configuration"], indent=2))
    print(json.dumps(hbond_rows, indent=2))


if __name__ == "__main__":
    main()
