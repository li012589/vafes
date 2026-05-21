import openmm as mm
from openmm import app
from pdbfixer import PDBFixer

from openmm.openmm import HarmonicAngleForce, PeriodicTorsionForce, NonbondedForce, CustomGBForce, HarmonicBondForce
from openmm.app.forcefield import ForceField, NoCutoff

import torch
import numpy as np

from .utils import ljType2Params, rMatrix, dihedralFromPos, angleFromPos

from .coulomb import coulombPair
from .lj import ljPairType2
from .functional import makePairPotential, makeVecBasedPotential
from .proximityBehavior import linearProximity
from .GBN2 import gbn2Potential
from .twobody import harmonicPair
from .threebody import harmonicAngle
from .fourbody import periodicProperDihedral


def _bisectionSearch(f, x, top, target, maxIter=100):
    r'''
    Use bisection search to search for f(x*) = target in range [0, top].
    '''
    bot = torch.zeros_like(x)
    for i in range(maxIter):
        pos = f(x) > target
        top = torch.where(pos, x, top)
        bot = torch.where(pos, bot, x)
        x = torch.where(pos, (x + bot) / 2, (x + top) / 2)
    return x


def _potentials(config, funcs, params):
    r'''
    Compute the potential values.
    Args:
        config (ndarray, [batch, N, 3] or [batch, Nterm]): the 3-dimensional coordinate of the atoms ([batch, N, 3]),or the distance matrix between all atoms ([batch, Nterm]);
        funcs (tuple of functions, Tuple(Func)): the functions used to compute each energy contribution;
    '''
    E = 0
    for f, p in zip(funcs, params):
        E += f(config, **p)
    return E


def _contributions(config, funcs, params):
    r'''
    Compute the potential contributions.
    Args:
        config (ndarray, [batch, N, 3] or [batch, Nterm]): the 3-dimensional coordinate of the atoms ([batch, N, 3]),or the distance matrix between all atoms ([batch, Nterm]);
        funcs (tuple of functions, Tuple(Func)): the functions used to compute each energy contribution;
    '''
    Elst = []
    for f, p in zip(funcs, params):
        Elst.append(f(config, **p))
    return torch.cat(Elst, dim=-1)


def energy(pos, pairPotentialFuncs, termPotentialFuncs, pairPotentialParams, termPotentialParams):
    r'''
    Compute the total energy of the molecular.
    Args:
        pos (ndarray, [batch, N, 3]): the three-dimensional coordinate of each atom in the molecular;
        pairPotentialFuncs (tuple of functions, Tuple(Func)): the functions used to compute each energy contribution from pair interactions;
        termPotentialFuncs (tuple of functions, Tuple(Func)): the functions used to compute each energy contribution from term interactions like angles and dihedrals;
        pairPotentialParams (tuple of Dicts of params, Tuple(Dict)): the parameters for pairPotentialFuncs;
        termPotentialParams (tuple of Dicts of params, Tuple(Dict)): the parameters for termPotentialFuncs
    '''
    _, distance = rMatrix(pos)

    pairPotValue = _potentials(distance, pairPotentialFuncs, pairPotentialParams)
    termPotValue = _potentials(pos, termPotentialFuncs, termPotentialParams)
    return pairPotValue + termPotValue


def energyContributions(pos, pairPotentialFuncs, termPotentialFuncs, pairPotentialParams, termPotentialParams):
    r'''
    Compute the energy contributions of the molecular.
    Args:
        pos (ndarray, [batch, N, 3]): the three-dimensional coordinate of each atom in the molecular;
        pairPotentialFuncs (tuple of functions, Tuple(Func)): the functions used to compute each energy contribution from pair interactions;
        termPotentialFuncs (tuple of functions, Tuple(Func)): the functions used to compute each energy contribution from term interactions like angles and dihedrals;
        pairPotentialParams (tuple of Dicts of params, Tuple(Dict)): the parameters for pairPotentialFuncs;
        termPotentialParams (tuple of Dicts of params, Tuple(Dict)): the parameters for termPotentialFuncs
    '''
    _, distance = rMatrix(pos)

    pairElst = _contributions(distance, pairPotentialFuncs, pairPotentialParams)
    termElst = _contributions(pos, termPotentialFuncs, termPotentialParams)
    return torch.cat([pairElst, termElst], dim=-1)


def fromOpenMM(forceFiles, pdbFile, nonbondedMethod=NoCutoff, regulation=True, regGrad=1e6, regPot=1e6, maxCutoff=1e9, solventDielectric=78.5, soluteDielectric=1, SA='ACE', kappa=0, eps=1e-10, maxPotential=torch.inf, dtype=torch.float32, device=torch.device('cpu')):
    r'''
    Import energy function from openMM
    Args:
        forceFiles (str): force files used;
        pdbFile (str): pdb file used;
    '''

    pdbContext = PDBFixer(pdbFile)
    pdbContext.findNonstandardResidues()
    pdbContext.replaceNonstandardResidues()
    pdbContext.findMissingResidues()
    pdbContext.findMissingAtoms()
    pdbContext.addMissingAtoms()
    pdbContext.addMissingHydrogens(7.0)

    forcefield = ForceField(*forceFiles)
    system = forcefield.createSystem(pdbContext.topology, nonbondedMethod=nonbondedMethod)

    forces = system.getForces()

    pairPotentialFuncs = []
    pairPotentialParams = []

    termPotentialFuncs = []
    termPotentialParams = []

    for force in forces:
        if isinstance(force, HarmonicAngleForce):
            _paramDict = _readHarmonicAngle(force, dtype=dtype, device=device)
            termPotentialFuncs.append(lambda pos, idx, k, theta0: makeVecBasedPotential(idx, angleFromPos, eps=eps, max=maxPotential)(harmonicAngle)(pos, k, theta0))
            termPotentialParams.append(_paramDict)

        elif isinstance(force, PeriodicTorsionForce):
            _paramDict = _readPeriodicTorsionForce(force, dtype=dtype, device=device)
            termPotentialFuncs.append(lambda pos, idx, k, n, phi0: makeVecBasedPotential(idx, dihedralFromPos, eps=eps, max=maxPotential)(periodicProperDihedral)(pos, k, n, phi0))
            termPotentialParams.append(_paramDict)

        elif isinstance(force, NonbondedForce):
            _coulomDict, _ljDict  = _readNonbondedForce(force, regulation, regGrad, dtype=dtype, device=device)
            if regulation:
                pairPotentialFuncs.append(lambda distance, charge, maxRange: makePairPotential(None, maxRange, eps=eps, max=maxPotential)(coulombPair)(distance, charge))
                pairPotentialParams.append(_coulomDict)

                ljPairWithLinearProximity = lambda rIJ, sigmaIJ, epsilonIJ, maxPotentialIJ, distanceIJ: torch.where(rIJ < distanceIJ, linearProximity(rIJ, regGrad, maxPotentialIJ), ljPairType2(rIJ, sigmaIJ, epsilonIJ))
                pairPotentialFuncs.append(lambda distance, sigma, epsilon, ljMaxGrad, maxRange: makePairPotential(None, cutoffRange=0, eps=eps, max=maxPotential)(ljPairWithLinearProximity)(distance, sigma, epsilon, ljMaxGrad, maxRange))
                pairPotentialParams.append(_ljDict)
            else:
                pairPotentialFuncs.append(lambda distance, charge: makePairPotential(None, cutoffRange=0, eps=eps, max=maxPotential)(coulombPair)(distance, charge))
                pairPotentialParams.append(_coulomDict)

                pairPotentialFuncs.append(lambda distance, sigma, epsilon: makePairPotential(None, cutoffRange=0, eps=eps, max=maxPotential)(ljPairType2)(distance, sigma, epsilon))
                pairPotentialParams.append(_ljDict)

        elif isinstance(force, CustomGBForce):
            if nonbondedMethod is NoCutoff:
                cutoff=None
            else:
                raise ValueError('Cuoff method not implemented!')
            _paramDict = _readCustomGBForce(force, dtype=dtype, device=device)
            pairPotentialFuncs.append(lambda distance, charge, Or, Sr, alpha, beta, gamma, d0, m0: gbn2Potential(distance, charge, Or, Sr, alpha, beta, gamma, d0, m0, maxCutoff, solventDielectric, soluteDielectric, SA, cutoff, kappa, eps))
            pairPotentialParams.append(_paramDict)

        elif isinstance(force, HarmonicBondForce):
            _paramDict = _readHarmonicBondForce(force, regulation=regulation, regPot=regPot, dtype=dtype, device=device)
            if regulation:
                harmonicBondWithLinearProximity = lambda rIJ, kIJ, bIJ, r1IJ, r2IJ, grad1IJ, grad2IJ: torch.where(rIJ > r1IJ, linearProximity(rIJ, -grad1IJ, regPot, r1IJ), torch.where(rIJ < r2IJ, linearProximity(rIJ, -grad2IJ, regPot, r2IJ), harmonicPair(rIJ, kIJ, bIJ)))
                pairPotentialFuncs.append(lambda distance, idx, k, bondLength, r1, r2, grad1, grad2: makePairPotential(idx, 0, eps=0, max=maxPotential)(harmonicBondWithLinearProximity)(distance, k, bondLength, r1, r2, grad1, grad2))
            else:
                pairPotentialFuncs.append(lambda distance, idx, k, bondLength: makePairPotential(idx, 0, eps=0, max=maxPotential)(harmonicPair)(distance, k, bondLength))
            pairPotentialParams.append(_paramDict)

        else:
            print('[WARNNING] Force type, '+ str(type(force)) + ', is not implemented yet. It is thus ignored!')

    return pairPotentialFuncs, termPotentialFuncs, pairPotentialParams, termPotentialParams


def _readHarmonicAngle(force, dtype=torch.float64, device=torch.device('cpu')):
    r'''
    Get parameters for harmonic angle interactions.
    '''
    forceParam = [force.getAngleParameters(i) for i in range(force.getNumAngles())]
    forceParam = (torch.stack([torch.tensor(f[:3], dtype=dtype, device=device) for f in forceParam]),
                  torch.stack([torch.tensor((f[3]._value, f[4]._value), dtype=dtype, device=device) for f in forceParam]))
    idx = forceParam[0].int()
    paramDict = {'idx': idx, 'k': forceParam[1][:, 1].unsqueeze(0), 'theta0': forceParam[1][:, 0].unsqueeze(0)}
    return paramDict


def _readPeriodicTorsionForce(force, dtype=torch.float64, device=torch.device('cpu')):
    r'''
    Get parameters for periodic torision interactions.
    '''
    forceParam = [force.getTorsionParameters(i) for i in range(force.getNumTorsions())]
    forceParam = (torch.stack([torch.tensor(f[:4], dtype=dtype, device=device) for f in forceParam]),
                  torch.stack([torch.tensor((f[4], f[5]._value, f[6]._value), dtype=dtype, device=device) for f in forceParam]).unsqueeze(0))
    idx = forceParam[0].int()
    paramDict = {'idx': idx, 'k': forceParam[1][0, :, 2].unsqueeze(0), 'n': forceParam[1][0, :, 0].unsqueeze(0), 'phi0': forceParam[1][0, :, 1].unsqueeze(0)}
    return paramDict


def _readNonbondedForce(force, regulation=True, regGrad=1e6, dtype=torch.float64, device=torch.device('cpu')):
    r'''
    Get parameters for non-bonded interactions.
    '''
    forceParam = [[j._value for j in force.getParticleParameters(i)] for i in range(force.getNumParticles())]

    charge, sigma, k = [term.squeeze().unsqueeze(0) for term in torch.split(torch.as_tensor(forceParam, dtype=dtype, device=device), 1, dim=-1)]

    charge = charge.unsqueeze(-1)
    charge = charge * charge.transpose(-1, -2)

    sigma, k = ljType2Params(sigma, k)

    exceptionParams = [force.getExceptionParameters(i) for i in range(force.getNumExceptions())]
    pairs = [(*sorted([i[0], i[1]]),) for i in exceptionParams]
    exceptions = (*zip(*[[i._value for i in j[2:]] for j in exceptionParams]),)

    for (i, j), cE, sE, kE in zip(pairs, *exceptions):
        charge[0, i, j] = charge[0, j, i] = cE
        sigma[0, i, j] = sigma[0, j, i] = sE
        k[0, i, j] = k[0, j, i] = kE

    if regulation:
        kMask = k == 0

        # minimum of Lennard-Jones potential
        rmin = 2**(1/6) * sigma
        x0 = rmin / 2

        f = lambda x: torch.where(kMask, -regGrad, 4 * k * (6*(sigma/x)**6-12*(sigma/x)**12) / x)
        rMaxGrad = _bisectionSearch(f, x0, rmin, target=-regGrad)
        rMaxGrad.masked_fill_(kMask, 0)
        lj = lambda x: torch.where(kMask, 0, k*((sigma/x)**12-(sigma/x)**6))
        LJMaxGrad = lj(rMaxGrad) + rMaxGrad * regGrad

        coulombDict = {'charge': charge, 'maxRange': rMaxGrad}
        ljDict = {'sigma': sigma, 'epsilon': k, 'ljMaxGrad': LJMaxGrad, 'maxRange': rMaxGrad}
    else:
        coulombDict = {'charge': charge}
        ljDict = {'sigma': sigma, 'epsilon': k}

    coulombDict = {key: torch.triu(value, diagonal=1).reshape(value.shape[0], -1) for key, value in coulombDict.items()}
    ljDict = {key: torch.triu(value, diagonal=1).reshape(value.shape[0], -1) for key, value in ljDict.items()}
    return coulombDict, ljDict


def _readCustomGBForce(force, dtype=torch.float64, device=torch.device('cpu')):
    r'''
    Get parameters for custom GB interactions of implicit water.
    '''
    forceParam = [force.getParticleParameters(i) for i in range(force.getNumParticles())]

    charge, Or, Sr, alpha, beta, gamma, radindex = torch.split(torch.as_tensor(forceParam, dtype=dtype, device=device), 1, dim=-1)

    # Tabulated functions
    d0 = force.getTabulatedFunction(0).getFunctionParameters()
    d0 = torch.as_tensor(d0[-1], dtype=dtype, device=device).reshape((d0[0], d0[1]))

    m0 = force.getTabulatedFunction(1).getFunctionParameters()
    m0 = torch.as_tensor(m0[-1], dtype=dtype, device=device).reshape((m0[0], m0[1]))

    radindex = radindex.long()
    d0 = d0[radindex].squeeze().transpose(0, 1)[radindex].squeeze().unsqueeze(0)
    m0 = m0[radindex].squeeze().transpose(0, 1)[radindex].squeeze().unsqueeze(0)

    paramDict = {'charge': charge.squeeze().unsqueeze(0), 'Or': Or.squeeze().unsqueeze(0), 'Sr': Sr.squeeze().unsqueeze(0), 'alpha': alpha.squeeze().unsqueeze(0), 'beta': beta.squeeze().unsqueeze(0), 'gamma': gamma.squeeze().unsqueeze(0), 'd0': d0, 'm0': m0}
    return paramDict


def _readHarmonicBondForce(force, regulation=True, regPot=1e6, dtype=torch.float64, device=torch.device('cpu')):
    r'''
    Get parameters for harmonic bound interactions.
    '''
    forceParam = [[j._value if type(j) is not int else j for j in force.getBondParameters(i)] for i in range(force.getNumBonds())]
    forceParam = torch.as_tensor(forceParam, dtype=dtype, device=device)
    idx, bondLength, k = torch.split(forceParam, [2, 1, 1], dim=-1)
    paramDict = {'idx': idx.int(), 'k': k.squeeze().unsqueeze(0), 'bondLength': bondLength.squeeze().unsqueeze(0)}
    if regulation:
        _r = torch.sqrt(regPot * 2 / k)
        _r1 = bondLength + _r
        _r2 = bondLength - _r
        _grad1 = k * _r
        _grad2 = -_grad1

        paramDict['r1'] = _r1.squeeze().unsqueeze(0)
        paramDict['r2'] = _r2.squeeze().unsqueeze(0)
        paramDict['grad1'] = _grad1.squeeze().unsqueeze(0)
        paramDict['grad2'] = _grad2.squeeze().unsqueeze(0)
    return paramDict