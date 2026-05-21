import torch
from .utils import disFromPos
# we implement potential energy with 2 atoms per term, see https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html for reference.


def harmonicBond(pos, mass, charge, idx, param):
    r'''
    compute the potential of bond using the Eq:
    $ V_b(r_{ij}) = \frac{1}{2} k^b_{ij} (r_{ij} - b_{ij})^2 $
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [bonds, 2]): idex of the 2 atoms in each bond;
        param (ndarray, [bonds, 2]): parameter list for each bond, with the first one being $k^b_{ij}$, and the second being $b_{ij}$.
    '''
    rIJ = disFromPos(pos[:, idx])
    return 1/2 * (param[:, 0] * (rIJ - param[:, 1])**2).sum(-1, keepdim=True)


def fourthPowerBond(pos, mass, charge, idx, param):
    r'''
    compute the potential of bond using the Eq:
    $ V_b(r_{ij}) = \frac{1}{4} k^b_{ij} (r_{ij}^2 - b_{ij}^2)^2 $
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [bonds, 2]): idex of the 2 atoms in each bond;
        param (ndarray, [bonds, 2]): parameter list for each bond, with the first one being $k^b_{ij}$, and the second being $b_{ij}$.
    '''
    rIJ = disFromPos(pos[:, idx])
    return 1/4 * (param[:, 0] * (rIJ**2 - param[:, 1]**2)**2).sum(-1, keepdim=True)


def coulombPair(pos, mass, charge, idx, param, eps=1e-5):
    r'''
    compute the potential of coulomb pair using the Eq:
    $ V_c(r_{ij}) = f * \frac{q_i q_j}{\epsilion_r r_{ij}} $
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [bonds, 2]): idex of the 2 atoms in each pair;
        param (ndarray, [bonds, 1]): parameter for the coulomb pair, i.e., $f/\epsilion_r$.
    '''
    rIJ = torch.clip(disFromPos(pos[:, idx]), min=eps)
    qIqJ = (charge[:, idx[:, 0]] * charge[:, idx[:, 1]]).squeeze(-1)
    return (param[:, 0] * qIqJ / rIJ).sum(-1, keepdim=True)


def _ljMeta(term12, term6, coeff12, coeff6, coeff=1):
    r'''
    meta function used to form type1/2 lj interaction
    '''
    return (coeff12 * term12 - coeff6 * term6) * coeff


def ljInteraction(pos, mass, charge, idx, param, max=1e7):
    r'''
    compute the potential of LJ potential pair using the Eq:
    $ V_{LJ}(r_{ij}) = C^{(12)}_{ij} / r^12_{ij} - C^{(6)}_{ij} / r^6_{ij}$
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [bonds, 2]): idex of the 2 atoms in each pair;
        param (ndarray, [bonds, 2]): parameter for the LJ potential pair, with the first one being $C^{(12)}_{ij}$, and the second being $C^{(6)}_{ij}$.
    '''
    rIJ = disFromPos(pos[:, idx])
    term6 = torch.clip(1 / rIJ**6, max=max)
    term12 = term6**2
    return _ljMeta(term12, term6, param[:, 0], param[:, 1]).sum(-1, keepdim=True)


def ljInteraction2(pos, mass, charge, idx, param, max=1e7):
    r'''
    compute the potential of LJ potential pair using the Eq:
    $ V_{LJ}(r_{ij}) = 4 \epsilon_{ij} (\sigma_{ij} / r)^12_{ij} - (\sigma_{ij} / r)^6_{ij}$
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [bonds, 2]): idex of the 2 atoms in each pair;
        param (ndarray, [bonds, 3]): parameter for the LJ potential pair, with the first one being $\sigma_{ij}$, and the second being $\epsilon_{ij}$.
    '''
    rIJ = disFromPos(pos[:, idx])
    term6 = torch.clip((param[:, 0] / rIJ)**6, max=max)
    term12 = term6**2
    return _ljMeta(term12, term6, 1, 1, 4 * param[:, 1]).sum(-1, keepdim=True)