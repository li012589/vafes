import torch
from .utils import disFromPos
# we implement potential energy with 2 atoms per term, see https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html for reference.


def harmonicPair(rIJ, kIJ, bIJ):
    r'''
    compute the potential of bond using the Eq:
    $ V_b(r_{ij}) = \frac{1}{2} k^b_{ij} (r_{ij} - b_{ij})^2 $
    Args:
        rIJ (ndarray, [batch, Npair]): the relative distance of i-th atom to j-th one;
        kIJ (ndarray, [batch, Npair]): the $\k^n_{ij}$;
        bIJ (ndarray, [batch, Npair]): the $\b_{ij}$.
    '''
    return 1/2 * (kIJ * (rIJ - bIJ)**2)


def fourthPowerPair(rIJ, kIJ, bIJ):
    r'''
    compute the potential of bond using the Eq:
    $ V_b(r_{ij}) = \frac{1}{4} k^b_{ij} (r_{ij}^2 - b_{ij}^2)^2 $
    Args:
        rIJ (ndarray, [batch, Npair]): the relative distance of i-th atom to j-th one;
        kIJ (ndarray, [batch, Npair]): the $\k^n_{ij}$;
        bIJ (ndarray, [batch, Npair]): the $\b_{ij}$.
    '''
    return 1/4 * (kIJ * (rIJ**2 - bIJ**2)**2)
