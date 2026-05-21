import torch
import numpy as np

from forceUtils.energy import energy
from forceUtils.twobody import harmonicBond, coulombPair, ljInteraction2, _ljMeta
from forceUtils.threebody import harmonicAngle
from forceUtils.fourbody import periodicProperDihedral
from forceUtils.utils import ljType2Params

# =============================================================================
# 1. Define the new coordinates (13 atoms) using only the atoms we wish to keep.
#    Original atoms to be kept come from indices:
#      [1, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18]
# =============================================================================


mass = torch.tensor([[
    [15.0340],  # original index 1
    [12.0100],  # original index 4
    [16.0000],  # original index 5
    [14.0100],  # original index 6
    [ 1.0080],  # original index 7
    [12.0100],  # original index 8
    [ 1.0080],  # original index 9
    [15.0340],  # original index 10
    [12.0100],  # original index 14
    [16.0000],  # original index 15
    [14.0100],  # original index 16
    [ 1.0080],  # original index 17
    [15.0340]   # original index 18
]], dtype=torch.float32)

charge = torch.tensor([[
    [-0.0293],  # original index 1
    [ 0.5972],  # original index 4
    [-0.5679],  # original index 5
    [-0.4157],  # original index 6
    [ 0.2719],  # original index 7
    [ 0.0337],  # original index 8
    [ 0.0823],  # original index 9
    [-0.0016],  # original index 10
    [ 0.5973],  # original index 14
    [-0.5679],  # original index 15
    [-0.4157],  # original index 16
    [ 0.2719],  # original index 17
    [ 0.1438]   # original index 18
]], dtype=torch.float32)

ljparamEpsSig = torch.tensor([[
    [0.2837065, 0.10672614936],
    [3.39967e-01, 3.59824e-01],
    [2.95992e-01, 8.78640e-01],
    [3.25000e-01, 7.11280e-01],
    [1.06908e-01, 6.56888e-02],
    [3.39967e-01, 4.57730e-01],
    [2.47135e-01, 6.56888e-02],
    [0.2837065, 0.10672614936],
    [3.39967e-01, 3.59824e-01],
    [2.95992e-01, 8.78640e-01],
    [3.25000e-01, 7.11280e-01],
    [1.06908e-01, 6.56888e-02],
    [0.270343, 0.10672614936]
]], dtype=torch.float32)

# =============================================================================
# 2. The energy is computed as the sum of several sub-terms.
#    The list "functs" remains unchanged.
# =============================================================================

functs = [harmonicBond, coulombPair, ljInteraction2, coulombPair, ljInteraction2, harmonicAngle, periodicProperDihedral, periodicProperDihedral]

# =============================================================================
# 3. Update the idxs list.
#
#    The original idx mapping (for 22 atoms) had, for example,
#
#       idxs[0] = torch.tensor([[ 0,  1],
#                               [ 1,  2],
#                                ... ])
#
#    But after deletion, only atoms with old indices:
#         [1, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18]
#    remain and we assign them new indices in the order shown:
#
#         old -> new mapping:
#            1  ->  0
#            4  ->  1
#            5  ->  2
#            6  ->  3
#            7  ->  4
#            8  ->  5
#            9  ->  6
#           10  ->  7
#           14  ->  8
#           15  ->  9
#           16  -> 10
#           17  -> 11
#           18  -> 12
#
#    Then we filter each index array so that only interactions whose atoms
#    are all kept are present—and we remap the numbers accordingly.
# =============================================================================

idxs = [
    # Harmonic Bonds (2-body interactions)
    torch.tensor([
        [0, 1],   # from original [1, 4]
        [1, 2],   # from original [4, 5]
        [1, 3],   # from original [4, 6]
        [3, 4],   # from original [6, 7]
        [3, 5],   # from original [6, 8]
        [5, 6],   # from original [8, 9]
        [5, 7],   # from original [8,10]
        [5, 8],   # from original [8,14]
        [8, 9],   # from original [14,15]
        [8,10],   # from original [14,16]
        [10,11],  # from original [16,17]
        [10,12]   # from original [16,18]
    ], dtype=torch.int64),

    # Coulomb Pairs (2-body interactions, filtered and remapped)
    torch.tensor([
        [0, 4],   # from original [1, 7]
        [0, 5],   # from original [1, 8]
        [1, 6],   # from original [4, 9]
        [1, 7],   # from original [4,10]
        [1, 8],   # from original [4,14]
        [2, 4],   # from original [5, 7]
        [2, 5],   # from original [5, 8]
        [3, 9],   # from original [6,15]
        [3,10],   # from original [6,16]
        [4, 6],   # from original [7, 9]
        [4, 7],   # from original [7,10]
        [4, 8],   # from original [7,14]
        [5,11],   # from original [8,17]
        [5,12],   # from original [8,18]
        [6, 9],   # from original [9,15]
        [6,10],   # from original [9,16]
        [7, 9],   # from original [10,15]
        [7,10],   # from original [10,16]
        [9,11],   # from original [15,17]
        [9,12]    # from original [15,18]
    ], dtype=torch.int64),

    # LJ pairs
    torch.tensor([
        [0, 4],   # from original [1, 7]
        [0, 5],   # from original [1, 8]
        [1, 6],   # from original [4, 9]
        [1, 7],   # from original [4,10]
        [1, 8],   # from original [4,14]
        [2, 4],   # from original [5, 7]
        [2, 5],   # from original [5, 8]
        [3, 9],   # from original [6,15]
        [3,10],   # from original [6,16]
        [4, 6],   # from original [7, 9]
        [4, 7],   # from original [7,10]
        [4, 8],   # from original [7,14]
        [5,11],   # from original [8,17]
        [5,12],   # from original [8,18]
        [6, 9],   # from original [9,15]
        [6,10],   # from original [9,16]
        [7, 9],   # from original [10,15]
        [7,10],   # from original [10,16]
        [9,11],   # from original [15,17]
        [9,12]    # from original [15,18]
    ], dtype=torch.int64),

    # Coulomb Pairs Long
    torch.tensor([
        [ 0,  6],
        [ 0,  7],
        [ 0,  8],
        [ 0,  9],
        [ 0, 10],
        [ 0, 11],
        [ 0, 12],
        [ 1,  9],
        [ 1, 10],
        [ 1, 11],
        [ 1, 12],
        [ 2,  6],
        [ 2,  7],
        [ 2,  8],
        [ 2,  9],
        [ 2, 10],
        [ 2, 11],
        [ 2, 12],
        [ 3, 11],
        [ 3, 12],
        [ 4,  9],
        [ 4, 10],
        [ 4, 11],
        [ 4, 12],
        [ 6, 11],
        [ 6, 12],
        [ 7, 11],
        [ 7, 12]
    ], dtype=torch.int64),

    # LJ pairs Long
    torch.tensor([
        [ 0,  6],
        [ 0,  7],
        [ 0,  8],
        [ 0,  9],
        [ 0, 10],
        [ 0, 11],
        [ 0, 12],
        [ 1,  9],
        [ 1, 10],
        [ 1, 11],
        [ 1, 12],
        [ 2,  6],
        [ 2,  7],
        [ 2,  8],
        [ 2,  9],
        [ 2, 10],
        [ 2, 11],
        [ 2, 12],
        [ 3, 11],
        [ 3, 12],
        [ 4,  9],
        [ 4, 10],
        [ 4, 11],
        [ 4, 12],
        [ 6, 11],
        [ 6, 12],
        [ 7, 11],
        [ 7, 12]
    ], dtype=torch.int64),

    # Harmonic Angles (3-body interactions)
    torch.tensor([
        [0, 1, 2],   # from original [1, 4, 5]
        [0, 1, 3],   # from original [1, 4, 6]
        [2, 1, 3],   # from original [5, 4, 6]
        [1, 3, 4],   # from original [4, 6, 7]
        [1, 3, 5],   # from original [4, 6, 8]
        [4, 3, 5],   # from original [7, 6, 8]
        [3, 5, 6],   # from original [6, 8, 9]
        [3, 5, 7],   # from original [6, 8,10]
        [3, 5, 8],   # from original [6, 8,14]
        [6, 5, 7],   # from original [9, 8,10]
        [6, 5, 8],   # from original [9, 8,14]
        [7, 5, 8],   # from original [10,8,14]
        [5, 8, 9],   # from original [8,14,15]
        [5, 8,10],   # from original [8,14,16]
        [9, 8,10],   # from original [15,14,16]
        [8,10,11],   # from original [14,16,17]
        [8,10,12],   # from original [14,16,18]
        [11,10,12]   # from original [17,16,18]
    ], dtype=torch.int64),

    # Periodic Proper Dihedrals (4-body interactions)
    torch.tensor([
        [0, 1, 3, 4],
        [0, 1, 3, 5],
        [2, 1, 3, 4],
        [2, 1, 3, 4],
        [2, 1, 3, 5],
        [1, 3, 5, 6],
        [1, 3, 5, 7],
        [1, 3, 5, 7],
        [1, 3, 5, 7],
        [1, 3, 5, 8],
        [1, 3, 5, 8],
        [4, 3, 5, 6],
        [4, 3, 5, 7],
        [4, 3, 5, 8],
        [3, 5, 8, 9],
        [3, 5, 8, 10],
        [3, 5, 8, 10],
        [3, 5, 8, 10],
        [6, 5, 8, 9],
        [6, 5, 8, 9],
        [6, 5, 8, 10],
        [7, 5, 8, 9],
        [7, 5, 8, 10],
        [7, 5, 8, 10],
        [7, 5, 8, 10],
        [5, 8, 10, 11],
        [5, 8, 10, 12],
        [9, 8, 10, 11],
        [9, 8, 10, 11],
        [9, 8, 10, 12]
    ], dtype=torch.int64),

    # Additional dihedral set: no atoms were removed from these interactions
    torch.tensor([
        [0, 3, 1, 2],   # from original [1,6,4,5]
        [1, 5, 3, 4],   # from original [4,8,6,7]
        [5,10, 8, 9],   # from original [8,16,14,15]
        [8,12,10,11]    # from original [14,18,16,17]
    ], dtype=torch.int64)
]

ljcal = torch.zeros(20, 2, dtype=torch.float32)
ljcal[:, 0], ljcal[:, 1] = ljType2Params(ljparamEpsSig[0, :, 0], ljparamEpsSig[0, :, 1], idxs[2])
ljcal[:, 1] = ljcal[:, 1] * 0.5

ljlong = torch.zeros(28, 2, dtype=torch.float32)
ljlong[:, 0], ljlong[:, 1] = ljType2Params(ljparamEpsSig[0, :, 0], ljparamEpsSig[0, :, 1], idxs[4])
# =============================================================================
# 4. Update the parameters (params). Each element of params corresponds row‐by‐row
#    to the respective idx array (so filter out the rows that belonged to removed 
#    interactions).
#
#    For example, for idxs[0], originally there were 21 bonds; here only the rows
#    corresponding to interactions with original indices [3,4,5,6,7,8,9,10,14,15,16,17]
#    (i.e. 12 bonds) are kept.
# =============================================================================

params = [
    # For harmonicBond:
    torch.tensor([
        [265265.600000, 0.152200],  # row 3 from original params[0]
        [476976.000000, 0.122900],  # row 4
        [410032.000000, 0.133500],  # row 5
        [363171.200000, 0.101000],  # row 6
        [282001.600000, 0.144900],  # row 7
        [284512.000000, 0.109000],  # row 8
        [259408.000000, 0.152600],  # row 9
        [265265.600000, 0.152200],  # row 10
        [476976.000000, 0.122900],  # row 14
        [410032.000000, 0.133500],  # row 15
        [363171.200000, 0.101000],  # row 16
        [282001.600000, 0.144900],  # row 17
    ], dtype=torch.float32),

    # For coulombPair (all values are identical in this example)
    torch.tensor([
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
    ], dtype=torch.float32) * 0.8333,

    ljcal,

    torch.tensor([
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458],
        [138.935458]
    ], dtype=torch.float32),

    ljlong,

    # For harmonicAngle:
    torch.tensor([
        [669.440000, 2.101376],  # row 6
        [585.760000, 2.035054],  # row 7
        [669.440000, 2.145010],  # row 8
        [418.400000, 2.094395],  # row 9
        [418.400000, 2.127556],  # row 10
        [418.400000, 2.060187],  # row 11
        [418.400000, 1.911136],  # row 12
        [669.440000, 1.914626],  # row 13
        [527.184000, 1.921608],  # row 14
        [418.400000, 1.911136],  # row 15
        [418.400000, 1.911136],  # row 16
        [527.184000, 1.939061],  # row 17
        [669.440000, 2.101376],  # row 24
        [585.760000, 2.035054],  # row 25
        [669.440000, 2.145010],  # row 26
        [418.400000, 2.094395],  # row 27
        [418.400000, 2.127556],  # row 28
        [418.400000, 2.060187],  # row 29
    ], dtype=torch.float32),

    # For periodicProperDihedral:
    torch.tensor([
        [10.460000, 2.000000, 3.141593],
        [10.460000, 2.000000, 3.141593],
        [10.460000, 2.000000, 3.141593],
        [8.3680000, 1.000000, 0.000000],
        [10.460000, 2.000000, 3.141593],
        [0.000000, 0.000000, 0.000000],
        [8.3680000, 1.000000, 0.000000],
        [8.3680000, 2.000000, 0.000000],
        [1.6736000, 3.000000, 0.000000],
        [1.129680, 2.000000, 0.000000],
        [1.7572800, 3.000000, 0.000000],
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000],
        [1.882800, 1.000000, 3.141593],
        [6.6107200, 2.0000, 3.141593],
        [2.3012000, 3.000000, 3.141593],
        [3.347200, 1.000000, 0.000000],
        [0.3347200, 3.000000, 3.141593],
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 0.000000],
        [0.836800, 1.000000, 0.000000],
        [0.8368000, 2.000000, 0.000000],
        [1.6736000, 3.000000, 0.000000], 
        [10.460000, 2.000000, 3.141593],
        [10.460000, 2.000000, 3.141593],
        [10.460000, 2.000000, 3.141593],
        [8.3680000, 1.000000, 0.000000],
        [10.460000, 2.000000, 3.141593]
    ], dtype=torch.float32),
    # For the additional dihedral the parameters remain unchanged:
    torch.tensor([
        [43.93200,  2.000000,  3.141593],
        [4.602400,  2.000000,  3.141593],
        [43.93200,  2.000000,  3.141593],
        [4.602400,  2.000000,  3.141593]
    ], dtype=torch.float32)
]

# =============================================================================
# 5. Calculate and print the energy.
# =============================================================================

energyFull = lambda config: energy(config, mass, charge, functs, idxs, params)

def full2concise(config):
    r'''
    Convert from a full coordinate system to the concise one.
    Order:CH3C(=O)N(H)C(H)(CH3)C(=O)N(H)CH3
    Args:
        config (ndarray, [batch, 3, 13]): dipeptide in full coordinate
    '''
    # Translate all points so the fifth point (index 4) is the origin.
    origin = config[:, 5:6, :]  # use the fifth point as the origin
    points_translated = config - origin  # translated points

    # Get the translated eighth point (index 7) and third point (index 2).
    a = points_translated[:, 8, :]  # new y-axis vector
    v = points_translated[:, 3, :]  # third point for the xy-plane

    # Compute the normalized y-axis.
    u = a / torch.norm(a, dim=1, keepdim=True)

    # Compute the projection of v onto the y-axis and its perpendicular part.
    v_dot_u = torch.sum(v * u, dim=1, keepdim=True)
    v_parallel = v_dot_u * u
    v_perpendicular = v - v_parallel

    # Compute the normalized x-axis.
    x_dir = v_perpendicular / torch.norm(v_perpendicular, dim=1, keepdim=True)

    # Compute the z-axis from y cross x.
    z_dir = torch.cross(u, x_dir, dim=1)

    # Build the rotation matrix with x, y, z as columns.
    rotation_matrix = torch.stack([x_dir, u, z_dir], dim=-1)

    # Apply the rotation matrix to the translated points.
    _config = torch.bmm(points_translated, rotation_matrix)

    flipMask = 2 * (_config[:, 7:8, -1] > 0).int() - 1
    _config[:, :, -1] = flipMask * _config[:, :, -1]

    concise = torch.zeros(config.shape[0], 33).to(config)
    concise[:, :9] = _config[:, :3].reshape(-1, 9)
    concise[:, 9:11] = _config[:, 3, :-1]
    concise[:, 11:14] = _config[:, 4]
    concise[:, 14:20] = _config[:, 6:8].reshape(-1, 6)
    concise[:, 20] = _config[:, 8, 1]
    concise[:, 21:] = _config[:, 9:].reshape(-1, 12)

    return concise


def concise2full(config):
    r'''
    Convert from a concise coordinate to full
    concise coord:
    smile: CH3C(=O)N(-----H)C(--------H)(CH3)---C(======O)N(H)CH3
    dof:   3  3  3 2(x>0) 3 0(origin) 3  3(z>0) 1(on y) 3 3 3 3   Total: 33
    Args:
        configs (ndarray, [batch, 34]): dipeptide in concise coordinate.
    '''
    full = torch.zeros(config.shape[0], 13, 3).to(config)
    # CH3C(=o)
    full[:, :3] = config[:, :9].reshape(-1, 3, 3)
    #N-
    full[:, 3, :-1] = config[:, 9:11]
    #H-
    full[:, 4] = config[:, 11:14]
    # (H)(CH3)
    full[:, 6:8] = config[:, 14:20].reshape(-1, 2, 3)
    # -C-
    full[:, 8, 1] = config[:, 20]
    # (=o)C(H)CH3
    full[:, 9:] = config[:, 21:].reshape(-1, 4, 3)
    return full

energyConcise = lambda config: energyFull(concise2full(config))
