import torch

import openmm.unit as u
from math import pi


GBN2_NECK_CUT = 0.68
GBN2_NECK_SCALE = 0.826836
GBN2_OFFSET = 0.0195141

E_CHARGE = 1.602176634e-19 * u.coulomb
EPSILON0 = 1e-6*8.8541878128e-12/(u.AVOGADRO_CONSTANT_NA*E_CHARGE**2) * u.farad/u.meter
ONE_4PI_EPS0 = 1/(4*pi*EPSILON0) * EPSILON0.unit


def _customGBPotential(distance, charge, radius, B, solventDielectric=78.5, soluteDielectric=1, SA='ACE', cutoff=None, kappa=0, offset=0):
    # translated from openmm, at: https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/internal/customgbforces.py#L354

    if kappa > 0:
        E1 = -0.5 * ONE_4PI_EPS0 * (1 / soluteDielectric - torch.exp(-kappa * B) / solventDielectric) * charge.square() / B
    elif kappa < 0:
        raise ValueError('kappa/ionic strength must be >= 0')
    else:
        E1 = -0.5 * ONE_4PI_EPS0 * (1 / soluteDielectric - 1 / solventDielectric) * charge.square() / B

    if SA == 'ACE':
        E2 = 28.3919551 * (radius + 0.14).square() * (radius / B)**6
    else:
        raise ValueError('Unknown surface area method: '+SA)

    charge1 = charge.unsqueeze(-1)
    charge2 = charge.unsqueeze(-2)
    B1 = B.unsqueeze(-1)
    B2 = B.unsqueeze(-2)
    f = (distance.square() + B1 * B2 * (-distance.square() / (4 * B1 * B2)).exp()).sqrt()
    if cutoff is None:
        if kappa >0:
            E3 = -ONE_4PI_EPS0 * (1 / soluteDielectric - torch.exp(-kappa * B) / solventDielectric) * charge1 * charge2 / f
        else:
            E3 = -ONE_4PI_EPS0 * (1 / soluteDielectric - 1 / solventDielectric) * charge1 * charge2 / f
    else:
        if kappa > 0:
            E3 = -ONE_4PI_EPS0 * (1 / soluteDielectric - torch.exp(-kappa * B) / solventDielectric) * charge1 * charge2 * (1/f - 1/cutoff)
        else:
            E3 = -ONE_4PI_EPS0 * (1 / soluteDielectric - 1 / solventDielectric) * charge1 * charge2 * (1/f - 1/cutoff)
    return E1.sum(-1, keepdim=True) + E2.sum(-1, keepdim=True) + E3.triu(diagonal=1).sum(dim=[-1, -2]).unsqueeze(-1)


def gbn2Potential(distance, charge, Or, Sr, alpha, beta, gamma, d0, m0, maxCutoff=1e9, solventDielectric=78.5, soluteDielectric=1, SA='ACE', cutoff=None, kappa=0, eps=1e-10):
    r'''
    GBN2 implicit solvent potential.
    Args:
        distance (ndarray, [batch, N, N]): the distance matirx
    '''
    distance = distance.masked_fill(distance > maxCutoff, 0)

    radius = Or + GBN2_OFFSET
    radius1 = radius.unsqueeze(-1)
    radius2 = radius.unsqueeze(-2)
    Or1 = Or.unsqueeze(-1)
    Sr2 = Sr.unsqueeze(-2)

    D = (distance - Sr2).abs()
    L = torch.maximum(Or1, D)
    L2 = torch.clip(L**2, min=eps)
    L = torch.clip(L, min=eps)
    U = distance + Sr2
    U.diagonal(dim1=-2, dim2=-1).zero_()
    U2 = torch.clip(U**2, min=eps)
    U = torch.clip(U, min=eps)
    distanceEps = torch.clip(distance, min=eps)

    Ivdw = (distance > (Sr2 - Or1)).float() * 0.5 * (1/L - 1/U + 0.25 * (distance - Sr2.square() / distanceEps) * (1/U2 - 1 / L2) + 0.5 * (L.log() - U.log()) / distanceEps)
    Ineck = ((radius1 + radius2 + GBN2_NECK_CUT) > distance).float() * m0 / (1 + 100 * (distance-d0).square() + 0.3*1000000*(distance - d0).pow(6))
    I = Ivdw + GBN2_NECK_SCALE * Ineck
    I.diagonal(dim1=-2, dim2=-1).zero_()
    I = I.sum(dim=-1)

    psi = I * Or
    B = 1 / (1 / torch.clip(Or, min=eps) - (alpha * psi - beta * psi**2 + gamma * psi**3).tanh() / torch.clip(radius, min=eps))

    return _customGBPotential(distance, charge, radius, B, solventDielectric, soluteDielectric, SA, cutoff, kappa, GBN2_OFFSET)