import torch
from .utils import dihedralFromPos
# we implement potential energy with 4 atoms per term, see https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html for reference.


def periodicProperDihedral(pos, mass, charge, idx, param):
    '''
    compute the potential of dihedrals using the Eq:
    $ V_d(\phi_{ijkl}) = k^{\phi}_{ijkl} (1 + \cos(n_{ijkl} \phi - \phi^s_{ijkl})) $
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [dihedrals, 4]): idex of the 4 atoms in each dihedral;
        param (ndarray, [angles, 3]): parameter list for each dihedral, with the first one being $k^{\phi}_{ijkl}$, the second being $n_{ijkl}$, and the third being $\phi^s_{ijkl}$.
    '''
    phiIJKL = dihedralFromPos(pos[:, idx])
    return (param[:, 0] * (1 + torch.cos(param[:, 1] * phiIJKL - param[:, 2]))).sum(-1, keepdim=True)
