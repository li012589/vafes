import torch
# we implement potential energy with 4 atoms per term, see https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html for reference.


def periodicProperDihedral(phiIJKL, kIJKL, nIJKL, phi0IJKL):
    r'''
    compute the potential of dihedrals using the Eq:
    $ V_d(\phi_{ijkl}) = k^{\phi}_{ijkl} (1 + \cos(n_{ijkl} \phi_{ijkl} - \phi^s_{ijkl})) $
    Args:
        phiIJKL (ndarray, [batch, Ndihedral]): the dihedral angle bewteen i-th atom, j-th atom, k-th atom and l-th atom;
        kIJKL (ndarray, [batch, Ndihedral]): the $k^{\phi}_{ijkl}$;
        nIJKL (ndarray, [batch, Ndihedral]): the $n_{ijkl}$;
        phi0IJKL(ndarray, [batch, Ndihedral]): the $\phi^s_{ijkl}$.
    '''
    return (kIJKL * (1 + torch.cos(nIJKL * phiIJKL - phi0IJKL)))
