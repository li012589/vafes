import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

from h2n2Coordinate import energyCV, _coord2Cv
from h2n2CvTrain import SigmoidCoupling

from forceUtils.energy import energy
from forceUtils.twobody import fourthPowerBond, coulombPair
from forceUtils.threebody import harmonicAngle, harmonicCosine
from forceUtils.fourbody import periodicProperDihedral


if __name__ == '__main__':
    device = torch.device('cpu')
    dtype = torch.float32
    batch = 512
    bins = 101
    cmap = 'viridis'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-load", default=None, help="path to load free energy model")
    parser.add_argument("-loadCV", default=None, help="path to load CV model")
    args = parser.parse_args()

    h2n2mass = torch.tensor([[[1.0080], [14.0067], [14.0067], [1.0080]]]).to(device, dtype)
    h2n2charge = torch.tensor([[[0.350], [-0.350], [-0.350], [0.350]]]).to(device, dtype)
    h2n2functs = [fourthPowerBond, coulombPair, harmonicCosine, periodicProperDihedral]
    h2n2idxs = [torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.tensor([[0, 3]]), torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[0, 1, 2, 3]])]
    h2n2params= [torch.tensor([[2.2652e7, 0.1040], [2.0480e7, 0.1250], [2.2652e7, 0.1040]]), torch.tensor([[138.935458]]), torch.tensor([[503.00, torch.deg2rad(torch.tensor(106.75))], [503.00, torch.deg2rad(torch.tensor(106.75))]]), torch.tensor([[41.80, 2, torch.deg2rad(torch.tensor(180.0))]])]
    h2n2params = [term.to(device, dtype) for term in h2n2params]
    pos = torch.tensor([[[-.1121, 0.0763, 0.000], [-0.0607, -0.0140, -0.000], [0.0607, 0.0140, -0.000], [.1121, -0.0763, 0.000]]]).to(device, dtype)
    cv = _coord2Cv(pos)

    energyFn = lambda config: energyCV(config, h2n2mass, h2n2charge, h2n2functs, h2n2idxs, h2n2params)

    ranges = torch.tensor([[-0.15, 0.15], [1e-5, 0.18], [0, 0.20], [-0.15, 0.15], [0, 1], [0, 0.14]])

    T = torch.tensor(1.0)
    testY = torch.tensor([0.0995, 0, -0.0995])
    testZ = torch.tensor([0.0, 0.11])
    maxY = ranges[-2, 1]
    minY = ranges[-2, 0]
    maxZ = ranges[-1, 1]
    minZ = ranges[-1, 0]

    nvars = [4]
    prior = source.Uniform
    transformationList = [flow.SplineFlow]
    priorParam, transformationParamList = torch.load(
        os.path.join(args.load, "best_TrainLoss_joint.saving"),
        map_location=device,
        weights_only=False,
    )
    transformationParamList[0]['linearBound'] = False

    cvTransformationParamList = torch.load(
        os.path.join(args.loadCV, "best_TrainLoss_joint.saving"),
        map_location=device,
        weights_only=False,
    )[0]

    with torch.no_grad():
        cv = cv.repeat(3, 1)
        cv[:, -2] = testY
        cvf, _ = source.TransformedDistribution.forward(cv, T=1, transformationList=[SigmoidCoupling], transformationParamList=cvTransformationParamList)
        testY = cvf[:, -2].detach()

    lossLst = []
    for _Y in testY:
        _Y = _Y.repeat(batch).unsqueeze(-1).to(device)
        for _Z in testZ:
            _Z = _Z.repeat(batch).unsqueeze(-1).to(device)
            with torch.no_grad():
                z = prior.sample(batch, nvars=nvars, T=1, **priorParam)
                zlogProb = prior.logProbability(z, T=1, **priorParam)

                z = torch.cat([z, _Y, _Z], dim=-1)

                _sample, logDet = source.TransformedDistribution.forward(z, T=1, transformationList=transformationList, transformationParamList=transformationParamList)

                sample, ilogDet = source.TransformedDistribution.inverse(_sample, T=1, transformationList=[SigmoidCoupling], transformationParamList=cvTransformationParamList)

                loss = zlogProb - logDet - ilogDet + energyFn(sample) / 1
            lossLst.append(loss.mean().detach().item())

    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()

    printString = "epoch: {:d}, L: {:.5f}, "
    printString += "time: {:.2f}, best: {:.5f}"
    resultLst = [-1, lossSum]
    resultLst += [-1, -1]
    print(printString.format(*resultLst))
    for idxT, _Y in enumerate(testY):
        printString = ' >> Y_{:.1f}'.format(_Y.item()) + "_h2n2"
        resultLst = []
        for idx, _Z in enumerate(testZ):
            printString += "@B" + str(_Z.item()) + ":{:.2f}  "
            resultLst += [lossLst[idx + len(testZ) * idxT].item()]
        print(printString.format(*resultLst))

    Yrange = np.linspace(ranges[-2][0], ranges[-2][1], bins)
    Zrange = np.linspace(ranges[-1][0], ranges[-1][1], bins)
    Y, Z = np.meshgrid(Yrange, Zrange)
    Y, Z = torch.from_numpy(Y).to(device, dtype).reshape(-1, 1), torch.from_numpy(Z).to(device, dtype).reshape(-1, 1)
    YZ = torch.cat([Y, Z], dim=-1)

    lnZLst = []
    errsLst = []
    ilogDetLst = []
    logDetLst = []
    energyLst = []
    for idx in range(bins**2):
        _YZ = YZ[idx].unsqueeze(0).repeat(batch, 1)
        with torch.no_grad():
            z = prior.sample(batch, nvars=nvars, T=1, **priorParam)
            zlogProb = prior.logProbability(z, T=1, **priorParam)

            z = torch.cat([z, _YZ], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=1, transformationList=transformationList, transformationParamList=transformationParamList)
            sample, ilogDet = source.TransformedDistribution.inverse(_sample, T=1, transformationList=[SigmoidCoupling], transformationParamList=cvTransformationParamList)
            energy = energyFn(sample)

            loss = zlogProb - logDet - ilogDet + energy / 1

        energyLst.append(energy.mean().detach().item())
        logDetLst.append(logDet.mean().detach().item())
        ilogDetLst.append(ilogDet.mean().detach().item())
        lnZLst.append(loss.mean().detach().item())
        errsLst.append(loss.std().detach().item())
    lnZLst = np.array(lnZLst).reshape(bins, bins)
    errsLst = np.array(errsLst).reshape(bins, bins)
    logDetLst = np.array(logDetLst).reshape(bins, bins)
    ilogDetLst = np.array(ilogDetLst).reshape(bins, bins)
    energyLst = np.array(energyLst).reshape(bins, bins)

    with open(os.path.join(args.load, "estCVfreeE.npy"), "wb") as f:
        np.save(f, YZ)
        np.save(f, lnZLst)
        np.save(f, errsLst)

    lnZLst = np.flip(lnZLst, 0)
    Zrange = np.flip(Zrange.numpy(), 0)
    tickIdx = [0, bins//2, bins-1]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.tick_params(labelsize=22)
    plt.title('Free energy surface of diazene', fontsize=27)
    plt.contourf(YZ[:, 0].reshape(lnZLst.shape), YZ[:, 1].reshape(lnZLst.shape), np.clip(lnZLst, max=700), levels=50, cmap=cmap)
    plt.colorbar()
    plt.xlabel('Machine-learning CV', fontsize=27)
    plt.xticks(
        ticks=tickIdx,
        labels=[f"{val:.2f}" for val in Yrange[tickIdx]]
    )
    plt.ylabel('Z coordinate', fontsize=27)
    plt.yticks(
        ticks=tickIdx,
        labels=[f"{val:.2f}" for val in Zrange[tickIdx]]
    )
    plt.savefig(os.path.join(args.load, 'cvFreeEest.pdf'))
    plt.close("all")
