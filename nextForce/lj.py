import torch

def ljPairType1(rIJ, c12IJ, c6IJ):
    r'''
    compute the potential of LJ potential pair using the Eq:
    $ V_{LJ}(r_{ij}) = C^{(12)}_{ij} / r^12_{ij} - C^{(6)}_{ij} / r^6_{ij}$
    Args:
        rIJ (ndarray, [batch, Npair]): the relative distance of i-th atom to j-th one;
        c12IJ (ndarray, [batch, Npair]): the $\C^{12}_{ij}$;
        c6IJ (ndarray, [batch, Npair]): the $\C^{6}_{ij}$;
    '''
    r6IJ = rIJ**6
    return (c12IJ / r6IJ**2 - c6IJ / r6IJ)


def ljPairType2(rIJ, sigmaIJ, epsilonIJ):
    r'''
    compute the potential of LJ potential pair using the Eq:
    $ V_{LJ}(r_{ij}) = 4 \epsilon_{ij} [(\sigma_{ij} / r_{ij})^12 - (\sigma_{ij} / r_{ij})^6]$
    Args:
        rIJ (ndarray, [batch, Npair]): the relative distance of i-th atom to j-th one;
        sigmaIJ (ndarray, [batch, Npair]): the $\sigma_{ij}$;
        epsilonIJ (ndarray, [batch, Npair]): the $\epsilon{ij}$.
    '''
    sigmaIJr6IJ = (sigmaIJ / rIJ)**6
    return (4 * epsilonIJ * sigmaIJr6IJ * (sigmaIJr6IJ - 1))

