"""
Geometry calculations for Chignolin (10-residue mini-protein).

Provides functions to compute:
  - d(Asp3 N -- Gly7 O)
  - d(Asp3 N -- Thr8 O)
  - Gly7 phi dihedral angle
  - Gly7 psi dihedral angle

All atom indices are 0-based, matching the 138-atom Chignolin PDB
(GLY1-TYR2-ASP3-PRO4-GLU5-THR6-GLY7-THR8-TRP9-GLY10, with hydrogens).

Atom mapping (PDB 1-indexed -> 0-indexed):
  Asp3 N   :  ATOM 31  -> index 30
  Gly7 O   :  ATOM 89  -> index 88
  Thr8 O   :  ATOM 96  -> index 95
  Thr6 C   :  ATOM 74  -> index 73   (for Gly7 phi)
  Gly7 N   :  ATOM 86  -> index 85
  Gly7 CA  :  ATOM 87  -> index 86
  Gly7 C   :  ATOM 88  -> index 87
  Thr8 N   :  ATOM 93  -> index 92   (for Gly7 psi)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Atom indices (0-based) for the 138-atom Chignolin structure
# ---------------------------------------------------------------------------
IDX_ASP3_N  = 30   # Asp3 backbone N
IDX_GLY7_O  = 88   # Gly7 backbone O
IDX_THR8_O  = 95   # Thr8 backbone O

IDX_THR6_C  = 73   # Thr6 backbone C   (phi i-1)
IDX_GLY7_N  = 85   # Gly7 backbone N
IDX_GLY7_CA = 86   # Gly7 backbone CA
IDX_GLY7_C  = 87   # Gly7 backbone C
IDX_THR8_N  = 92   # Thr8 backbone N   (psi i+1)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _distance(pos, i, j):
    """Euclidean distance between atoms *i* and *j*.

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)
        Cartesian coordinates.  Leading dimensions are broadcast-friendly
        (single config, batch, etc.).
    i, j : int
        0-based atom indices.

    Returns
    -------
    ndarray, shape (...)
        Distance(s) in the same length unit as *pos*.
    """
    return np.linalg.norm(pos[..., i, :] - pos[..., j, :], axis=-1)


def _dihedral(pos, i, j, k, l):
    """Dihedral angle defined by four atom indices (i-j-k-l).

    Uses the atan2 definition so the result is in (-pi, pi].

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)
        Cartesian coordinates.
    i, j, k, l : int
        0-based atom indices.

    Returns
    -------
    ndarray, shape (...)
        Dihedral angle in **degrees**.
    """
    p0 = pos[..., i, :]
    p1 = pos[..., j, :]
    p2 = pos[..., k, :]
    p3 = pos[..., l, :]

    b0 = p1 - p0
    b1 = p1 - p2   # note: negative of (p2-p1)
    b2 = p2 - p3   # note: negative of (p3-p2)

    # normalise the central bond
    b1_norm = b1 / np.clip(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12, None)

    # project b0 and b2 onto the plane perpendicular to b1
    v = b0 - np.sum(b0 * b1_norm, axis=-1, keepdims=True) * b1_norm
    w = b2 - np.sum(b2 * b1_norm, axis=-1, keepdims=True) * b1_norm

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_norm, v) * w, axis=-1)

    return np.degrees(np.arctan2(y, x))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def distance_Asp3N_Gly7O(pos):
    """Return d(Asp3 N -- Gly7 O).

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)

    Returns
    -------
    ndarray, shape (...)
    """
    return _distance(pos, IDX_ASP3_N, IDX_GLY7_O)


def distance_Asp3N_Thr8O(pos):
    """Return d(Asp3 N -- Thr8 O).

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)

    Returns
    -------
    ndarray, shape (...)
    """
    return _distance(pos, IDX_ASP3_N, IDX_THR8_O)


def dihedral_Gly7_phi(pos):
    """Return phi dihedral of Gly7: C(Thr6)-N(Gly7)-CA(Gly7)-C(Gly7).

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)

    Returns
    -------
    ndarray, shape (...)
        Angle in degrees.
    """
    return _dihedral(pos, IDX_THR6_C, IDX_GLY7_N, IDX_GLY7_CA, IDX_GLY7_C)


def dihedral_Gly7_psi(pos):
    """Return psi dihedral of Gly7: N(Gly7)-CA(Gly7)-C(Gly7)-N(Thr8).

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)

    Returns
    -------
    ndarray, shape (...)
        Angle in degrees.
    """
    return _dihedral(pos, IDX_GLY7_N, IDX_GLY7_CA, IDX_GLY7_C, IDX_THR8_N)


def compute_chignolin_geometry(pos):
    """Compute all geometry metrics for a Chignolin configuration.

    Parameters
    ----------
    pos : ndarray, shape (..., num_atoms, 3)
        Can be a single configuration (num_atoms, 3) or a batch
        (..., num_atoms, 3).

    Returns
    -------
    dict with keys:
        'd_Asp3N_Gly7O' : ndarray  -- distance in same unit as pos
        'd_Asp3N_Thr8O' : ndarray  -- distance in same unit as pos
        'phi_Gly7'       : ndarray  -- phi angle in degrees
        'psi_Gly7'       : ndarray  -- psi angle in degrees
    """
    return {
        'd_Asp3N_Gly7O': distance_Asp3N_Gly7O(pos),
        'd_Asp3N_Thr8O': distance_Asp3N_Thr8O(pos),
        'phi_Gly7': dihedral_Gly7_phi(pos),
        'psi_Gly7': dihedral_Gly7_psi(pos),
    }
