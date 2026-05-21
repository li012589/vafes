import torch

import openmm.unit as u
from math import pi

E_CHARGE = 1.602176634e-19 * u.coulomb
EPSILON0 = 1e-6*8.8541878128e-12/(u.AVOGADRO_CONSTANT_NA*E_CHARGE**2) * u.farad/u.meter
ONE_4PI_EPS0 = 1/(4*pi*EPSILON0) * EPSILON0.unit


def coulombPair(rIJ, kqIqJ):
    r'''
    Compute the potential of coulomb pair using the Eq:
    $ V_c(r_{ij}) = k * \frac{q_i q_j}{r_{ij}} $
    Args:
        rIJ (ndarray, [batch, Npair]): the relative distance of i-th atom to j-th one;
        kqIqJ (ndarray, [batch, Npair]): the charge contributions and the coulomb constant.
    Return:
        ndarray, [batch, 1]: the summation of all coulomb pairs.
    '''
    return (kqIqJ / rIJ) * ONE_4PI_EPS0
