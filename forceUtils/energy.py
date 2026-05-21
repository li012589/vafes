import torch

def energy(pos, mass, charge, functs=[], idxs=[], params=[]):
    '''
    compute the potential energy.
    Args:
        pos (ndarray, [batch, N, 3]): position of N atoms;
        mass (ndarray, [batch, N, 1]): the mass of N atoms;
        charge (ndarray, [batch, N, 1]): the charge of N atoms;
        functs (list of functions): functions of sub terms for energy;
        idxs (list of ndarry): the idex of the atoms, the length should be the same as functs;
        params (list of ndarray): the parameters used in sub functions, the length should be the same as functs.
    '''
    energy = 0
    for fun, idx, param in zip(functs, idxs, params):
        energy += fun(pos, mass, charge, idx, param)

    return energy

def force(pos, mass, charge, functs=[], idxs=[], params=[]):
    """Calculate force as the negative gradient of energy w.r.t. positions"""
    pos.requires_grad_(True)
    energy = 0
    for fun, idx, param in zip(functs, idxs, params):
        energy += fun(pos, mass, charge, idx, param)
    force = -torch.autograd.grad(energy, pos, create_graph=False)[0]
    pos.requires_grad_(False)
    return force