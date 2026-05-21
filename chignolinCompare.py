import os

from scope import source, flow, utils

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import json

from nextForce.frontend import fromOpenMM, energy
from chignolinEnergy import ProteinConciseExpression, addHydrogen


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    dtype = torch.float32
    batch = 128
    bins = 50
    cmap = 'viridis'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-load", default=None, help="path to load free energy model")
    parser.add_argument("-device", type=int, default=-1, help="device, -1 for cpu, 0-N for i-th GPU, -2 for mps")
    parser.add_argument("-beta", type=float, default=-1, help="temperature")
    args = parser.parse_args()

    if args.device == -1:
        device = "cpu"
    elif args.device == -2:
        device = "mps"
    else:
        device = "cuda:"+str(args.device)
    device = torch.device(device)

    with open(os.path.join(args.load, "parameter.json"), "r") as f:
        config = json.load(f)

    betaCV = config.get('betaCV', False)

    proteinEnergyParams = fromOpenMM(['amber14-all.xml', 'implicit/gbn2.xml'], os.path.join(SCRIPT_DIR, 'etc', 'geoOpt.pdb'), eps=1e-5, device=device)

    helperFile = np.load(os.path.join(args.load, 'etc.npz'))

    refHeavy = helperFile['refHeavy']
    refHydrogen = helperFile['refHydrogen']
    Hidx = helperFile['Hidx']
    heavyIdx = helperFile['heavyIdx']
    Hs = helperFile['Hs']
    idxMaj = helperFile['idxMaj']
    ranges = helperFile['ranges']

    refHeavy = torch.from_numpy(refHeavy).to(device, dtype)
    refHydrogen = torch.from_numpy(refHydrogen).to(device, dtype)
    Hidx = torch.from_numpy(Hidx).to(device)
    heavyIdx = torch.from_numpy(heavyIdx).to(device)
    Hs = torch.from_numpy(Hs).to(device)
    idxMaj = torch.from_numpy(idxMaj).to(device)
    ranges = torch.from_numpy(ranges).to(device, dtype)

    if args.beta == -1:
        beta = torch.tensor(config['beta']).to(device, dtype)
    else:
        beta = torch.tensor(args.beta).to(device, dtype)

    max1 = torch.tensor(17).to(device, dtype)
    min1 = torch.tensor(3.4).to(device, dtype)
    max2 = torch.tensor(10).to(device, dtype)
    min2 = torch.tensor(4.5).to(device, dtype)

    test1 = torch.linspace(min1, max1, 7)
    test2 = torch.linspace(min2, max2, 7)
    test1, test2 = torch.meshgrid(test1, test2)
    test1 = test1.reshape(-1)
    test2 = test2.reshape(-1)

    energyFn = lambda config: energy(addHydrogen(config,
                                                 refHeavy, refHydrogen, Hidx, heavyIdx, Hs, idxMaj) / 10,
                                     *proteinEnergyParams)

    nvars = [223]
    prior = source.TruncatedGaussian
    transformationList = [flow.SplineFlow]
    priorParam, muNet, sigmaNet, transformationParamList = torch.load(os.path.join(args.load, "best_TrainLoss_joint.saving"), map_location=device, weights_only=False)

    lossLst = []
    energyLst = []
    for idx in range(len(test1)):
        _cv1 = test1[idx].reshape(1, 1).to(device)
        _cv2 = test2[idx].reshape(1, 1).to(device)
        with torch.no_grad():
            _cv12 = torch.cat([_cv1, _cv2], dim=-1)
            _cv12 = _cv12.repeat(batch, 1)
            beta_ = torch.Tensor([[beta]]).repeat(batch, 1).to(_cv12)

            if betaCV:
                beta_input = torch.cat([beta_, _cv12], dim=-1)
            else:
                beta_input = beta_

            z = source.Uniform.sample(batch, nvars=nvars, T=1, low=priorParam['low'], high=priorParam['high'])
            zlogProb = source.Uniform.logProbability(z, T=1, low=priorParam['low']-1e-5, high=priorParam['high']+1e-5)


            z = torch.cat([_cv12, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=beta_input, transformationList=transformationList, transformationParamList=transformationParamList)

            sample, ilogDet = ProteinConciseExpression.inverse(_sample, T=beta)

            es = energyFn(sample)
            loss = zlogProb - logDet - ilogDet + es * beta

        lossLst.append(loss.mean().detach().item())
        energyLst.append(es.min().item())

    lossLst = np.array(lossLst)
    lossSum = lossLst.mean()
    energyLst = np.array(energyLst)

    printString = "epoch: {:d}, L: {:.5f}, "
    printString += "time: {:.2f}, best: {:.5f}"
    resultLst = [-1, lossSum]
    resultLst += [-1, -1]
    print(printString.format(*resultLst))

    for idx in range(len(test1)):
        printString = 'Protein >> @Dist_{:.1f}'.format(test1[idx].item())
        resultLst = []
        printString += "@D2_{:.1f}".format(test2[idx].item()) + ":{:.2f}, e: {:.2f}"
        resultLst += [lossLst[idx].item(), energyLst[idx].item()]
        print(printString.format(*resultLst))

    cv1Range = np.linspace(min1.item(), max1.item(), bins)
    cv2Range = np.linspace(min2.item(), max2.item(), bins)
    cv1, cv2 = np.meshgrid(cv1Range, cv2Range)
    cv1, cv2 = torch.from_numpy(cv1).to(device, dtype).reshape(-1, 1), torch.from_numpy(cv2).to(device, dtype).reshape(-1, 1)
    cv12 = torch.cat([cv1, cv2], dim=-1)

    lnZLst = []
    errsLst = []
    logDetLst = []
    energyLst = []
    sampleLst = []
    sampleEnergyLst = []
    for idx in range(bins**2):
        _cv12 = cv12[idx].unsqueeze(0)
        with torch.no_grad():
            _cv12 = _cv12.repeat(batch, 1)
            beta_ = torch.Tensor([[beta]]).repeat(batch, 1).to(_cv12)

            if betaCV:
                beta_input = torch.cat([beta_, _cv12], dim=-1)
            else:
                beta_input = beta_

            z = source.Uniform.sample(batch, nvars=nvars, T=1, low=priorParam['low'], high=priorParam['high'])
            zlogProb = source.Uniform.logProbability(z, T=1, low=priorParam['low']-1e-5, high=priorParam['high']+1e-5)


            z = torch.cat([_cv12, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=beta_input, transformationList=transformationList, transformationParamList=transformationParamList)

            sample, ilogDet = ProteinConciseExpression.inverse(_sample, T=beta)

            es = energyFn(sample)
            sampleLst.append(addHydrogen(sample, refHeavy, refHydrogen, Hidx, heavyIdx, Hs, idxMaj).unsqueeze(0))
            sampleEnergyLst.append(es.detach().cpu().reshape(-1))

            loss = zlogProb - logDet - ilogDet + es * beta

        energyLst.append(es.mean().detach().item())
        logDetLst.append(logDet.mean().detach().item())
        lnZLst.append(loss.mean().detach().item())
        errsLst.append(loss.std().detach().item())
    lnZLst = np.array(lnZLst).reshape(bins, bins)
    errsLst = np.array(errsLst).reshape(bins, bins)
    logDetLst = np.array(logDetLst).reshape(bins, bins)
    energyLst = np.array(energyLst).reshape(bins, bins)
    sampleLst = torch.cat(sampleLst).detach().cpu().numpy()
    sampleEnergyLst = torch.cat(sampleEnergyLst).numpy()

    cv12 = cv12.detach().cpu().numpy()
    with open(os.path.join(args.load, "samplesCV.npy"), "wb") as f:
        np.save(f, cv12)
        np.save(f, sampleLst)

    with open(os.path.join(args.load, "sampleEnergies.npy"), "wb") as f:
        np.save(f, sampleEnergyLst)

    with open(os.path.join(args.load, "estCVfreeE.npy"), "wb") as f:
        np.save(f, cv12)
        np.save(f, lnZLst)
        np.save(f, errsLst)

    plt.figure(figsize=(12, 9))
    plt.title('estimated CV Free energy of Chignolin folding')
    plt.contourf(cv12[:, 0].reshape(lnZLst.shape), cv12[:, 1].reshape(lnZLst.shape), np.clip(lnZLst, max=1260), levels=40, cmap=cmap)
    plt.colorbar()
    plt.xlabel('CV1')
    plt.ylabel('CV2')
    plt.savefig(os.path.join(args.load, 'cvFreeEest.pdf'))

    plt.close("all")
