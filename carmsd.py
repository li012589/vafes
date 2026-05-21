#!/usr/bin/env python3
"""
Calculate CA-RMSD (C-alpha Root Mean Square Deviation) between two PDB files.

Supports PDB files containing multiple models (e.g. NMR ensembles). All pairwise
CA-RMSD values between models from the two files are computed and displayed as a
matrix. Distance unit: Angstroms.

Usage:
    python carmsd.py -load1 structure1.pdb -load2 structure2.pdb

Example:
    python carmsd.py -load1 1uao.pdb -load2 Native.pdb
"""

import argparse
import numpy as np
import sys


def parse_ca_atoms_all_models(pdb_path):
    """
    Parse CA atom coordinates from *every* model in a PDB file.

    For PDB files without MODEL/ENDMDL records, all CA atoms are treated as
    a single model.

    Returns:
        all_coords: list of numpy arrays, each of shape (N, 3)
        residues:   list of (chain, resSeq, resName) tuples (from the first model)
        model_ids:  list of model ID integers (1-based)
    """
    models = {}
    residues_map = {}
    current_model = 0
    has_model_records = False

    with open(pdb_path, "r") as f:
        for line in f:
            record = line[:6].strip()

            if record == "MODEL":
                has_model_records = True
                current_model = int(line[10:14].strip())
                if current_model not in models:
                    models[current_model] = []
                    residues_map[current_model] = []
                continue

            if record == "ENDMDL":
                continue

            if record in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                alt_loc = line[16]
                if alt_loc not in (" ", "A", ""):
                    continue

                res_name = line[17:20].strip()
                chain_id = line[21]
                res_seq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                key = current_model if has_model_records else 1
                if key not in models:
                    models[key] = []
                    residues_map[key] = []
                models[key].append([x, y, z])
                residues_map[key].append((chain_id, res_seq, res_name))

    if not models:
        print(f"Error: No CA atoms found in {pdb_path}", file=sys.stderr)
        sys.exit(1)

    model_ids = sorted(models.keys())
    all_coords = [np.array(models[m], dtype=np.float64) for m in model_ids]
    residues = residues_map[model_ids[0]]

    return all_coords, residues, model_ids


def kabsch_rmsd(P, Q):
    """
    Compute the RMSD between two sets of points after optimal superposition
    using the Kabsch algorithm.

    Args:
        P: numpy array of shape (N, 3) - reference coordinates
        Q: numpy array of shape (N, 3) - mobile coordinates

    Returns:
        rmsd: float, the CA-RMSD in the same units as input (Angstroms)
    """
    assert P.shape == Q.shape, "Coordinate arrays must have the same shape"
    N = P.shape[0]

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    M = Q_centered.T @ P_centered
    U, S, Vt = np.linalg.svd(M)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T

    Q_rotated = (R @ Q_centered.T).T
    diff = P_centered - Q_rotated
    rmsd = np.sqrt((diff ** 2).sum() / N)

    return rmsd


def main():
    parser = argparse.ArgumentParser(
        description="Calculate CA-RMSD between two PDB structures (in Angstroms). "
                    "Supports multi-model PDB files; computes all pairwise CA-RMSD values."
    )
    parser.add_argument(
        "-load1", required=True, metavar="PDB_FILE",
        help="Path to the first PDB file"
    )
    parser.add_argument(
        "-load2", required=True, metavar="PDB_FILE",
        help="Path to the second PDB file"
    )
    parser.add_argument(
        "-core", nargs=2, type=int, metavar=("START", "END"),
        help="Only use CA atoms from residue START to END (inclusive, 1-based "
             "residue indices). E.g. -core 2 9 excludes terminal residues."
    )
    args = parser.parse_args()

    coords1, res1, ids1 = parse_ca_atoms_all_models(args.load1)
    coords2, res2, ids2 = parse_ca_atoms_all_models(args.load2)

    n_ca1 = coords1[0].shape[0]
    n_ca2 = coords2[0].shape[0]

    print(f"Structure 1: {args.load1}  ({len(ids1)} model(s), {n_ca1} CA atoms each)")
    print(f"Structure 2: {args.load2}  ({len(ids2)} model(s), {n_ca2} CA atoms each)")

    if n_ca1 != n_ca2:
        print(
            f"\nError: Number of CA atoms differs: {n_ca1} vs {n_ca2}",
            file=sys.stderr,
        )
        sys.exit(1)

    mismatch = False
    for i, (r1, r2) in enumerate(zip(res1, res2)):
        if r1[1] != r2[1] or r1[2] != r2[2]:
            if not mismatch:
                print("\nWarning: Residue mismatches detected:")
                mismatch = True
            print(f"  Position {i+1}: {r1[2]} {r1[0]}{r1[1]} vs {r2[2]} {r2[0]}{r2[1]}")

    if args.core:
        core_start, core_end = args.core
        keep = [i for i, r in enumerate(res1) if core_start <= r[1] <= core_end]
        if not keep:
            print(f"\nError: No residues found in range [{core_start}, {core_end}]",
                  file=sys.stderr)
            sys.exit(1)
        coords1 = [c[keep] for c in coords1]
        coords2 = [c[keep] for c in coords2]
        kept_res = [res1[i] for i in keep]
        n_kept = len(keep)
        res_names = ", ".join(f"{r[2]}{r[1]}" for r in kept_res)
        print(f"Core selection: residues {core_start}-{core_end} "
              f"({n_kept} CA atoms: {res_names})")

    n1 = len(ids1)
    n2 = len(ids2)
    rmsd_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            rmsd_matrix[i, j] = kabsch_rmsd(coords1[i], coords2[j])

    def print_matrix(matrix, label=None):
        if n1 == 1 and n2 == 1:
            prefix = f"  {label}: " if label else "\n"
            suffix = "" if label else "\n"
            print(f"{prefix}CA-RMSD = {matrix[0, 0]:.4f} Angstroms{suffix}")
        else:
            if label:
                print(f"\n{label}:")
            else:
                print()
            print("Pairwise CA-RMSD matrix (Angstroms):")
            print(f"  Rows: {args.load1} models {ids1}")
            print(f"  Cols: {args.load2} models {ids2}")
            print()

            col_w = 8
            header = " " * 12
            for mid in ids2:
                header += f"{'M'+str(mid):>{col_w}}"
            print(header)

            for i, mid1 in enumerate(ids1):
                row = f"  {'M'+str(mid1):<10}"
                for j in range(n2):
                    row += f"{matrix[i, j]:{col_w}.4f}"
                print(row)

            print("\nSummary:")
            print(f"  Min  CA-RMSD = {matrix.min():.4f} Angstroms")
            print(f"  Max  CA-RMSD = {matrix.max():.4f} Angstroms")
            print(f"  Mean CA-RMSD = {matrix.mean():.4f} Angstroms")

    print_matrix(rmsd_matrix)


if __name__ == "__main__":
    main()
