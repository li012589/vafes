import torch


def linearProximity(rIJ, gradIJ, maxPotentialIJ, bIJ=0):
    r'''
    Args:
        rIJ (ndarray, [batch, Npair]): the relative distance of i-th atom to j-th one;
        gradIJ (int or ndarray, [batch, Npair]): the linear slope for each pair;
        maxPotentialIJ (int or ndarray, [batch, Npair]): the maximum potential allowed (r=0);
        bIJ (float or ndarray, [batch, Npair]): the base distance;
    '''
    return -gradIJ * (rIJ - bIJ) + maxPotentialIJ