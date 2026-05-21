import torch
from .utils import angleFromPos, cosFromPos
# we implement potential energy with 3 atoms per term, see https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html for reference.


def harmonicAngle(pos, mass, charge, idx, param):
    '''
    compute the potential of angles using the Eq:
    $ V_a(\theta_{ijk}) = \frac{1}{2} k^{\theta}_{ijk} (\theta_{ijk} - \theta^0_{ijk})^2 $
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [angles, 3]): idex of the 3 atoms in each angle;
        param (ndarray, [angles, 2]): parameter list for each angle, with the first one being $k^{\theta}_{ijk}$, and the second being $\theta^0_{ijk}$.
    '''
    thetaIJK = angleFromPos(pos[:, idx])

    return 1/2 * (param[:, 0] * (thetaIJK - param[:, 1])**2).sum(-1, keepdim=True)


def harmonicCosine(pos, mass, charge, idx, param):
    '''
    compute the potential of cosine using the Eq:
    $ V_a(\theta_{ijk}) = \frac{1}{2} k^{\theta}_{ijk} (\cos \theta_{ijk} - \cos \theta^0_{ijk})^2 $
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        idx (ndarray, [angles, 3]): idex of the 3 atoms in each angle;
        param (ndarray, [angles, 2]): parameter list for each angle, with the first one being $k^{\theta}_{ijk}$, and the second being $\theta^0_{ijk}$.
    '''
    cosIJK = cosFromPos(pos[:, idx])

    return 1/2 * (param[:, 0] * (cosIJK - torch.cos(param[:, 1]))**2).sum(-1, keepdim=True)