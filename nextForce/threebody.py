import torch
# we implement potential energy with 3 atoms per term, see https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html for reference.


def harmonicAngle(thetaIJK, kIJK, theta0IJK):
    r'''
    compute the potential of angles using the Eq:
    $ V_a(\theta_{ijk}) = \frac{1}{2} k^{\theta}_{ijk} (\theta_{ijk} - \theta^0_{ijk})^2 $
    Args:
        thetaIJK (ndarray, [batch, Nangles]): the angle between i-th atom, j-th atom, and k-th atom;
        kIJK (ndarray, [batch, Nangles]): the $k^{\theta}_{ijk}$;
        theta0IJK (ndarray, [batch, Nangles]): the $\theta^0_{ijk}$.
    '''
    return 1/2 * (kIJK * (thetaIJK - theta0IJK)**2)


def harmonicCosine(cosIJK, kIJK, theta0IJK):
    r'''
    compute the potential of cosine using the Eq:
    $ V_a(\theta_{ijk}) = \frac{1}{2} k^{\theta}_{ijk} (\cos \theta_{ijk} - \cos \theta^0_{ijk})^2 $
    Args:
        cosIJK (ndarray, [batch, Nangles]): the angle cosine between i-th atom, j-th atom, and k-th atom;
        vecs (ndarray, [batch, Nangles, 2, 3]): the two vectors define the angle;
        kIJK (ndarray, [batch, Nangles]): the $k^{\theta}_{ijk}$;
        theta0IJK (ndarray, [batch, Nangles]): the $\cos \theta^0_{ijk}$.
    '''
    return 1/2 * (kIJK * (cosIJK - torch.cos(theta0IJK))**2)