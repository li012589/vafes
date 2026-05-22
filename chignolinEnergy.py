import torch
import math
import numpy as np

from localCoordinate import forward, inverse
from localCartesianCoordinate import forwardCartesian, inverseCartesian
from localTorsionCoordinate import forwardTorsion, inverseTorsion

from scope import flow


_Natom=77

_idxO=52
_Oamino = 7
_idxY=17
_Yamino = 2
_idxX=1
_Xamino = 0
_idxZ=73
_Zamino = 9


def _preprocess(config, Natom=_Natom, idxO=_idxO, idxY=_idxY, idxX=_idxX, idxZ=_idxZ):
    r'''
    Convert from a arbitrary coordinate system to the convention.
    Use CA at index = idxO as the origin (0,0,0), CA at index = idxY on the y axis (0, y>0, 0), CA at index = idxX on the xoy plane (x>0, y, 0), CA at index = idxZ on the xoz plane (x, y, z>0).
    Args:
        config (ndarray, [batch, 3, 77]): full coordinate configurations;
        idxO (int): the index of atom used as origin;
        idxY (int): the index of atom on the positive y axis;
        idxX (int): the index of atom on the xoy plane;
        idxZ (int): the index of atom on the xoz plane.
    '''
    # Translate all points so idxO is the origin.
    origin = config[:, idxO:idxO+1, :]  # use idxO as the origin
    points_translated = config - origin  # translated points

    # Get translated idxY and idxX points.
    a = points_translated[:, idxY, :]  # new y-axis vector
    v = points_translated[:, idxX, :]  # third point for the xy-plane

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

    return _config

basisVectorIdx = torch.tensor([[52, 1], [1, 73]])
basisRotation = torch.tensor([5])
basisRotationCentering = torch.tensor([2])
_allDisIdx = basisVectorIdx

residueIdx = [
    torch.tensor([[52, 51, 53], [1, 0, 2], [17, 16, 18], [73, 74, 72]]),
    torch.tensor([[58, 59, 60], [49, 48, 47], [6, 5, 4], [24, 25, 26]]),
    torch.tensor([[42, 41, 40], [31, 32, 33]])
]
residueRotation = [0, 2, 8, 14, 18, 20, 30, 32, 42, 44, 48, 50, 54]
residueRotatioCentering = torch.tensor([math.pi+1, math.pi, -2, math.pi, 1, -2, -2, -2, 2, -1.5, -2, 1, math.pi-1])
_allForIdx = torch.tensor([item.tolist() for sublist in residueIdx for item in sublist])

peptideBondIdx = [
    torch.tensor([[51, 52, 53, 58], [53, 52, 51, 49], [18, 17, 16, 6], [16, 17, 18, 24]]),
    torch.tensor([[49, 48, 47, 42], [24, 25, 26, 31]])
]

sideChainFrameIdx = [
    torch.cat([
        torch.tensor([[1, 4, 2, 3], [5, 16, 6, 7], [17, 24, 18, 19], [25, 31, 26, 27], [32, 40, 33, 34], [41, 47, 42, 43], [48, 51, 49, 50], [52, 58, 53, 54], [59, 72, 60, 61]]), # O defined by Ca_i-C_i-N_{i+1}
        torch.tensor([[4, 6, 5, 8], [16, 18, 17, 20], [24, 26, 25, 28], [31, 33, 32, 35], [40, 42, 41, 44], [51, 53, 52, 55], [58, 60, 59, 62],
                      [72, 73, 74, 75]]), # 22
    ], dim=0),
    torch.tensor([[4, 5, 8, 9], [16, 17, 20, 21], [25, 28, 24, 30], [31, 32, 35, 36], [40, 41, 44, 46], [51, 52, 55, 57], [58, 59, 62, 63]]),
    torch.tensor([[5, 8, 9, 10], [17, 20, 21, 22], [25, 28, 30, 29], [32, 35, 36, 37], # 33
                  [46, 41, 44, 45], [57, 52, 55, 56],
                  [59, 62, 63, 65]]), # 36
    torch.tensor([[8, 9, 10, 12], [35, 36, 37, 38], # 38
                  [62, 63, 65, 67]]),
    torch.tensor([[63, 67, 65, 68]]),
]

torsionRotatePi = [
    6, 7, 8, 9, 10, 11, 12, 13, 14, # residue Os
    25, # side chain PRO #4
    32, # side chain PRO #4
    36, # C6+C5, flip
    37, # C6
    39, # C5
    40,  # C6
    23, 24, 26, 27, 28, 29, 33 #  tri peaks from tetrahedron of C
]
peptideTorsionCentering = torch.tensor([-0.7, 1.3, 2.0, 3, math.pi, -0.7])

_allTorsionIdx = torch.tensor([item.tolist() for sublist in (peptideBondIdx + sideChainFrameIdx) for item in sublist])

_specialTorsion = torch.tensor([0, 1, 2, 3, 4, 5, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 36, 38, # total: 19
                                6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 34, 35, 37, 39, 40])
_specialTorsionBack = torch.sort(_specialTorsion)[1]

planeAtomIdx = [
    [9, 10, 12, 11, 13, 14, 15], # C6 ring
    [63, 65, 67, 66, 64], [67, 65, 68, 69, 70, 71], # C6 + C5 ring
    [73, 74, 75, 76], [20, 21, 22, 23], [36, 37, 38, 39] # O - C - O
]

def _forwardDis(cartesian_coords: torch.Tensor):
    """
    Forward transform: Cartesian (x, y, z) -> (r, a, b)

    Input:
        cartesian_coords: Tensor of shape [batch, 3], representing (x, y, z)

    Output:
        transformed_coords: Tensor of shape [batch, 3], representing (r, a, b)
        log_abs_det_jacobian: Tensor of shape [batch, 1], representing log|det(J)|

    Coordinates:
        r = sqrt(x^2 + y^2 + z^2)  (radius)
        a = z / r                  (normalized z, cos(theta))
        b = atan2(y, x)            (azimuth)

    Jacobian:
        log|det(J_forward)| = -2 * log(r)
        This depends only on r, not a/b, which improves stability near the z-axis.
    """
    x, y, z = cartesian_coords.unbind(dim=1)  # split into [batch] tensors

    # Compute r = sqrt(x^2 + y^2 + z^2).
    r = torch.sqrt(x**2 + y**2 + z**2)

    # Compute a = z / r with a safety clamp.
    a = z / torch.clamp(r, min=1e-5)

    # Compute b = atan2(y, x).
    b = torch.atan2(y, x)

    # Pack output coordinates as [r, a, b].
    transformed_coords = torch.stack([r, a, b], dim=1)

    # log|det(J)| = -2 * log(r), depending only on r.
    log_abs_det = -2 * torch.log(torch.clamp(r, min=1e-5))
    log_abs_det = log_abs_det.unsqueeze(1)  # [batch] -> [batch, 1]

    return transformed_coords, log_abs_det


def _inverseDis(spherical_coords: torch.Tensor):
    """
    Inverse transform: (r, a, b) -> Cartesian (x, y, z)

    Input:
        spherical_coords: Tensor of shape [batch, 3], representing (r, a, b)

    Output:
        cartesian_coords: Tensor of shape [batch, 3], representing (x, y, z)
        log_abs_det_jacobian: Tensor of shape [batch, 1], representing log|det(J)|

    Coordinates:
        x = r * sqrt(1 - a²) * cos(b)
        y = r * sqrt(1 - a²) * sin(b)
        z = r * a

    Jacobian:
        log|det(J_inverse)| = 2 * log(r)
        This depends only on r, not a/b, which improves stability near the z-axis.
    """
    r, a, b = spherical_coords.unbind(dim=1)  # split into [batch] tensors

    # Safely compute sqrt(1 - a^2) near a = +/-1.
    # Clamp to avoid negative values from floating-point error.
    sqrt_1_minus_a2 = torch.sqrt(1.0 - a**2 + 1e-6)

    # Compute Cartesian coordinates.
    x = r * sqrt_1_minus_a2 * torch.cos(b)
    y = r * sqrt_1_minus_a2 * torch.sin(b)
    z = r * a

    # Pack output coordinates as [x, y, z].
    cartesian_coords = torch.stack([x, y, z], dim=1)

    # log|det(J)| = 2 * log(r), depending only on r.
    log_abs_det = 2 * torch.log(r + 1e-5)
    log_abs_det = log_abs_det.unsqueeze(1)  # [batch] -> [batch, 1]

    return cartesian_coords, log_abs_det


def getRotationMatrix(fixed, mobile, reflection=True):
    # Calculate cross-covariance matrices
    cov = (fixed.unsqueeze(-1) * mobile.unsqueeze(-2)).sum(2)
    # for gradient stable, avoid 0 or small singluar values from collinear points.
    cov = cov + 1e-3 * torch.eye(cov.shape[-1]).to(cov)
    v, _, w = torch.linalg.svd(cov)
    if not reflection:
        # Remove possibility of reflected atom coordinates
        detSign = torch.sign(torch.linalg.det(v) * torch.linalg.det(w))
        # Out-of-place: multiply last column of v by detSign (avoid inplace on SVD output)
        sign_matrix = torch.ones(v.shape[-1], device=v.device, dtype=v.dtype)
        sign_matrix = sign_matrix.expand(detSign.shape + (v.shape[-1],)).clone()
        sign_matrix[..., -1] = detSign
        v = v * sign_matrix.unsqueeze(-2)
    matrices = torch.matmul(v, w)
    return matrices


def addHydrogen(configs, refHeavy, refHydrogen, Hidx, heavyIdx, maskH, maskNCO):
    heavy = configs[:, heavyIdx] - configs[:, Hidx].unsqueeze(-2)
    rotMatrix = getRotationMatrix(heavy, refHeavy.unsqueeze(0), reflection=False)

    mask = ~(refHydrogen == 0)
    hydrogen = torch.matmul(rotMatrix, refHydrogen.permute(0, 2, 1).unsqueeze(0)).permute(0, 1, 3, 2)
    hydrogen = hydrogen + configs[:, Hidx].unsqueeze(-2)
    hydrogen = torch.masked_select(hydrogen, mask.unsqueeze(0)).reshape(configs.shape[0], -1, 3)

    newConfig = torch.zeros(configs.shape[0], 138, 3).to(configs)
    newConfig[:, maskNCO] = configs
    newConfig[:, maskH] = hydrogen
    return newConfig


class ProteinConciseExpression(flow.Bijector):
    r'''
    expression protein atom coordinates in a concise way.
    '''
    @staticmethod
    def bijection(inverse, x, T, *args, **kwargs):
        if not inverse:
            return _full2concise(x)
        else:
            return _concise2full(x)

def _full2concise(configs):
    '''
    import matplotlib.pyplot as plt
    '''

    batch = configs.shape[0]
    logDet = 0

    # handle all distances
    disVec = configs[:, _allDisIdx[:, 1]] - configs[:, _allDisIdx[:, 0]]
    localDisVec, disLogDet = _forwardDis(disVec.reshape(-1, 3))
    localDisVec = localDisVec.reshape(batch, _allDisIdx.shape[0], 3)
    disLogDet = disLogDet.reshape(batch, _allDisIdx.shape[0])
    disLogDet[:, 0] = disLogDet[:, 0] + torch.log(torch.clamp(localDisVec[:, 0, 0], min=1e-5))
    disLogDet = disLogDet.sum(-1, keepdim=True)
    logDet = logDet + disLogDet

    # handle all residues
    atoms = configs[:, _allForIdx]
    localResidue, residueLogDet = forward(atoms.reshape(-1, 3, 3))
    localResidue, residueLogDet = localResidue.reshape(batch, len(_allForIdx), 3, 3)[:, :, 1:], residueLogDet.reshape(batch, len(_allForIdx)).sum(-1, keepdim=True)
    localResidue = localResidue.reshape(batch, -1)
    logDet = logDet + residueLogDet

    # residue rotations
    localResidue[:, residueRotation] = (localResidue[:, residueRotation] + residueRotatioCentering + math.pi) % (2 * math.pi) - math.pi

    # handle all torsion
    torsionAtom = configs[:, _allTorsionIdx]
    localTorsionAtom, torsionAtomLogDet = forwardTorsion(torsionAtom.reshape(-1, 4, 3))
    localTorsionAtom, torsionAtomLogDet = localTorsionAtom[:, -1].reshape(batch, len(_allTorsionIdx), 3), torsionAtomLogDet.reshape(batch, len(_allTorsionIdx)).sum(-1, keepdim=True)
    logDet = logDet + torsionAtomLogDet

    # torsion rotations
    localTorsionAtom[:, torsionRotatePi, -1] = (localTorsionAtom[:, torsionRotatePi, -1] + (math.pi) + math.pi) % (2 * math.pi) - math.pi
    localTorsionAtom[:, :6, -1] = (localTorsionAtom[:, :6, -1] + peptideTorsionCentering + math.pi) % (2 * math.pi) - math.pi

    # handle plane atoms
    localPlaneAtom = []
    for idxL, idxLst in enumerate(planeAtomIdx):
        planeAtom = configs[:, idxLst]
        _localPlaneAtom, _ = forwardCartesian(planeAtom)
        _localPlaneAtom = _localPlaneAtom[:, 3:]
        localPlaneAtom.append(_localPlaneAtom)
    localPlaneAtom = torch.cat(localPlaneAtom, dim=1)

    # handle basis
    localBasis = torch.cat([localDisVec[:, -1, 0:1], configs[:, 17, 1:2], localDisVec[:, 0, [0, -1]].reshape(batch, 2), localDisVec[:, -1, 1:]], dim=-1)

    # basis rotation
    localBasis[:, basisRotation] = (localBasis[:, basisRotation] + basisRotationCentering + math.pi) % (2 * math.pi) - math.pi

    localResults = torch.cat([
        localBasis,
        localResidue.view(batch, len(_allForIdx), 6)[:, :, :3].reshape(batch, -1), # angles for the frames
        localTorsionAtom[:, :, -1][:, _specialTorsion],# torsion anlges
        localTorsionAtom[:, :, :-1].reshape(batch, -1),
        localResidue.view(batch, len(_allForIdx), 6)[:, :, 3:].reshape(batch, -1),
        localPlaneAtom.reshape(batch, -1)
    ], dim=-1)

    return localResults, logDet


def _concise2full(conciseConfig):
    batch = conciseConfig.shape[0]
    configs = torch.zeros(batch, _Natom, 3).to(conciseConfig)
    logDet = 0

    localBasis = conciseConfig[:, :6]
    localResidueAngle = conciseConfig[:, 6:36].reshape(batch, 10, 3)
    localTorsionAngle = conciseConfig[:, 36:77][:, _specialTorsionBack].unsqueeze(-1)
    localTorsionPos = conciseConfig[:, 77:159].reshape(batch, 41, 2)
    localResiduePos = conciseConfig[:, 159:189].reshape(batch, 10, 3)
    localPlaneAtom = conciseConfig[:, 189:].reshape(batch, -1, 3)

    localTorsionAtom = torch.cat([localTorsionPos, localTorsionAngle], dim=-1)
    localResidue = torch.cat([localResidueAngle, localResiduePos], dim=-1).reshape(batch, -1)

    # rotations
    localBasis[:, basisRotation] = localBasis[:, basisRotation] - basisRotationCentering.to(configs)
    localTorsionAtom[:, torsionRotatePi, -1] = localTorsionAtom[:, torsionRotatePi, -1] - (math.pi)
    localTorsionAtom[:, :6, -1] = localTorsionAtom[:, :6, -1] - peptideTorsionCentering.to(configs)
    localResidue[:, residueRotation] = localResidue[:, residueRotation] - residueRotatioCentering.to(configs)

    # put togther localDisVec
    localDisVec = torch.zeros(batch, 2, 3).to(configs)
    localDisVec[:, 0, [0, -1]] = localBasis[:, [2, 3]]
    localDisVec[:, 1, 0] = localBasis[:, 0]
    localDisVec[:, 1, [1, 2]] = localBasis[:, 4:]


    # inverse basis
    configs[:, 17, 1] = localBasis[:, 1]
    disVec, disLogDet = _inverseDis(localDisVec.reshape(-1, 3))
    disVec = disVec.reshape(batch, 2, 3)
    disLogDet = disLogDet.reshape(batch, 2)
    disLogDet[:, 0] = disLogDet[:, 0] - torch.log(localDisVec[:, 0, 0] + 1e-5)
    disLogDet = disLogDet.sum(-1, keepdim=True)
    logDet = logDet + disLogDet
    for idx in range(basisVectorIdx.shape[0]):
        configs[:, basisVectorIdx[idx, 1]] = disVec[:, idx] + configs[:, basisVectorIdx[idx, 0]]

    # inverse frames
    localResidue = localResidue.reshape(batch, 10, 2, 3)
    counter = 0
    bondCounter = 0
    for idxL, _frameIdx in enumerate(residueIdx):
        if idxL >= 1:
            # solve peptide bonds
            _bondIdx = peptideBondIdx[idxL-1]
            _peptide, peptideLogDet = inverseTorsion(torch.cat([configs[:, _bondIdx[:, :-1]], localTorsionAtom[:, bondCounter:bondCounter+_bondIdx.shape[0]].unsqueeze(-2)], dim=-2).reshape(-1, 4, 3))
            _peptide, peptideLogDet = _peptide.reshape(batch, -1, 4, 3), peptideLogDet.reshape(batch, -1).sum(-1, keepdim=True)
            logDet = logDet + peptideLogDet
            bondCounter += _bondIdx.shape[0]
            configs[:, _bondIdx[:, -1]] = _peptide[:, :, -1]

        # solve the frame
        _residue, frameLogDet = inverse(torch.cat([configs[:, _frameIdx[:, 0]].unsqueeze(-2), localResidue[:, counter:counter+_frameIdx.shape[0]]], dim=-2).reshape(-1, 9))
        _residue, frameLogDet = _residue.reshape(batch, -1, 3, 3), frameLogDet.reshape(batch, -1).sum(-1, keepdim=True)
        logDet = logDet + frameLogDet
        counter += _frameIdx.shape[0]
        configs[:, _frameIdx[:, 1:]] = _residue[:, :, 1:]

    # inverse frame for side chain
    for idxL, _scfIdx in enumerate(sideChainFrameIdx):
        _bondAtom, scfLogDet = inverseTorsion(torch.cat([configs[:, _scfIdx[:, :-1]], localTorsionAtom[:, bondCounter:bondCounter+_scfIdx.shape[0]].unsqueeze(-2)], dim=-2).reshape(-1, 4, 3))
        _bondAtom, scfLogDet = _bondAtom.reshape(batch, -1, 4, 3), scfLogDet.reshape(batch, -1).sum(-1, keepdim=True)
        logDet = logDet + scfLogDet
        bondCounter += _scfIdx.shape[0]
        configs[:, _scfIdx[:, -1]] = _bondAtom[:, :, -1]

    # handle plane atoms
    counter = 0
    for idxL, _planeIdx in enumerate(planeAtomIdx):
        _plane, _ = inverseCartesian(torch.cat([configs[:, _planeIdx[:3]], localPlaneAtom[:, counter:counter+len(_planeIdx)-3]], dim=-2))
        counter += len(_planeIdx) - 3
        configs[:, _planeIdx[3:]] = _plane[:, 3:]

    return configs, logDet
