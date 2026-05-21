import torch


def rMatrix(pos):
    r'''
    compute the vec matrix and the distance matrix from position of all atoms.
    Args:
        pos (ndarray, [batch, N, 3]): positions of N atoms.
    '''
    pos = pos.unsqueeze(-2)
    vecMatrix = pos - pos.transpose(-2, -3)
    distanceMatrix = torch.norm(vecMatrix, dim=-1)
    return vecMatrix, distanceMatrix


def disFromPos(pos):
    '''
    compute the distance from position of 2 atoms.
    Args:
        pos (ndarray, [batch, bonds, 2, 3]): position of 2 atoms.
    '''
    return torch.norm(pos[:, :, 0] - pos[:, :, 1], dim=-1)


def cosFromPos(pos, eps=1e-7):
    '''
    compute the angle from position of 3 atoms.
    Args:
        pos (ndarray, [batch, angles, 3, 3]): position of 3 atoms;
        eps (float, default 1e-7): the eps to avoid zero norm.
    '''
    vecIJ = pos[:, :, 0] - pos[:, :, 1]
    vecKJ = pos[:, :, -1] - pos[:, :, 1]
    cos = (vecIJ * vecKJ).sum(-1) / torch.clamp(torch.norm(vecIJ, dim=-1) * torch.norm(vecKJ, dim=-1), min=eps)
    return cos


def angleFromPos(pos, eps=1e-7):
    '''
    compute the angle from position of 3 atoms.
    Args:
        pos (ndarray, [batch, angles, 3, 3]): position of 3 atoms;
        eps (float, default 1e-7): the clip eps to avoid numerical erro from acos.
    '''
    cos = torch.clamp(cosFromPos(pos, eps), max=1-eps, min=-1+eps)
    return torch.acos(cos)


def dihedralFromPos(pos, eps=1e-7):
    '''
    compute the dihedral from position of 4 atoms.
    Args:
        pos (ndarray, [batch, dihedrals, 4, 3]): position of 4 atoms;
        eps (float, default 1e-7): the eps to avoid zero norm.
    '''
    b0 = pos[:, :, 1] - pos[:, :, 0]
    b1 = - pos[:, :, 2] + pos[:, :, 1]
    b2 = - pos[:, :, 3] + pos[:, :, 2]

    b1Norm = b1 / torch.clamp(torch.norm(b1, dim=-1, keepdim=True), min=eps)

    v = b0 - (b0 * b1Norm).sum(-1, keepdim=True) * b1Norm
    w = b2 - (b2 * b1Norm).sum(-1, keepdim=True) * b1Norm
    #v = -1.0 * v

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.linalg.cross(b1Norm, v) * w, dim=-1)

    return torch.atan2(y, x)


def ljCparamsFromEpsSig(sigma, epsilon):
    r'''
    c12_{ij} = 4 \epsilon_{ij} \sigma_{ij}^12
    c6_{ij} = 4 \epsilon_{ij} \sigma_{ij}^6
    Args:
        sigma (ndarray, [atomNum]): \sigma_{ii};
        epsilon (ndarray, [atomNum]): \epsilon_{ii};
    Ret:
        c12 (ndarray, [atomNum]): C^{(12)}_{ii};
        c6 (ndarray, [atomNum]): C^{(6)}_{ii}.
    '''
    c12 = 4 * epsilon * sigma**12
    c6 = 4 * epsilon * sigma**6
    return c12, c6


def ljType1Params(c12, c6, idx):
    r'''
    Args:
        c12 (ndarray, [atomNum]): C^{(12)}_{ii};
        c6 (ndarray, [atomNum]): C^{(6)}_{ii};
        idx (ndarray, [bonds, 2] or None): ij indices of each bond, none for using all parameters;
    Ret:
        c12 (ndarray, [bonds]): C^{(12)}_{ij};
        c6 (ndarray, [bonds]): C^{(6)}_{ij}.
    '''
    if idx is None:
        c12IJ = c12.unsqueeze(-1)
        c12IJ = (c12IJ * c12IJ.transpose(-1, -2))**0.5
        c6IJ = c6.unsqueeze(-1)
        c6IJ = (c6IJ * c6IJ.transpose(-1, -2))**0.5
    else:
        c12IJ = (c12[idx[:, 0]] * c12[idx[:, 1]])**0.5
        c6IJ = (c6[idx[:, 0]] * c6[idx[:, 1]])**0.5
    return c12IJ, c6IJ


def ljType2Params(sigma, epsilon, idx=None):
    r'''
    Args:
        sigma (ndarray, [atomNum]): \sigma{ii};
        epsilon (ndarray, [atomNum]): \epsilon_{ii};
        idx (ndarray, [bonds, 2] or None): ij indices of each bond, none for using all parameters;
    Ret:
        sigmaIJ (ndarray, [bonds]): \sigma{ij};
        epsilonIJ (ndarray, [bonds]): \epsilon_{ij}.
    '''
    if idx is None:
        sigmaIJ = sigma.unsqueeze(-1)
        sigmaIJ = (sigmaIJ + sigmaIJ.transpose(-1, -2)) / 2
        epsilonIJ = epsilon.unsqueeze(-1)
        epsilonIJ = (epsilonIJ * epsilonIJ.transpose(-1, -2))**0.5
    else:
        sigmaIJ = (sigma[idx[:, 0]] + sigma[idx[:, 1]]) / 2
        epsilonIJ = (epsilon[idx[:, 0]] * epsilon[idx[:, 1]])**0.5
    return sigmaIJ, epsilonIJ
