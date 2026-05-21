import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse
import json

from h2n2Coordinate import energyCV, _coord2Cv, _cv2Coord
from h2n2CvTrain import SigmoidCoupling

from forceUtils.energy import energy
from forceUtils.twobody import fourthPowerBond, coulombPair
from forceUtils.threebody import harmonicAngle, harmonicCosine
from forceUtils.fourbody import periodicProperDihedral


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-load", default=None, help="path to load free energy model")
    parser.add_argument("-loadCV", default=None, help="path to load CV model")
    parser.add_argument("-device", type=int, default=-1, help="device, -1 for cpu, 0-N for i-th GPU, -2 for mps")
    parser.add_argument("-maxSteps", type=int, default=7000, help="maximum optimization steps for NEB")
    args = parser.parse_args()

    if args.device == -1:
        device = torch.device('cpu')
    elif args.device == -2:
        device = torch.device('mps')
    else:
        device = torch.device(f'cuda:{args.device}')

    dtype = torch.float32
    batch = 256
    bins = 60
    nebK = 10000
    cmap = 'viridis'
    maxSteps = args.maxSteps
    miniBatch = 64

    pic_dir = os.path.join(args.load, 'pic')
    os.makedirs(pic_dir, exist_ok=True)

    with open(os.path.join(args.load, "parameter.json"), "r") as f:
        config = json.load(f)

    h2n2mass = torch.tensor([[[1.0080], [14.0067], [14.0067], [1.0080]]]).to(device, dtype)
    h2n2charge = torch.tensor([[[0.350], [-0.350], [-0.350], [0.350]]]).to(device, dtype)
    h2n2functs = [fourthPowerBond, coulombPair, harmonicCosine, periodicProperDihedral]
    h2n2idxs = [torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.tensor([[0, 3]]), torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[0, 1, 2, 3]])]
    h2n2params= [torch.tensor([[2.2652e7, 0.1040], [2.0480e7, 0.1250], [2.2652e7, 0.1040]]), torch.tensor([[138.935458]]), torch.tensor([[503.00, torch.deg2rad(torch.tensor(106.75))], [503.00, torch.deg2rad(torch.tensor(106.75))]]), torch.tensor([[41.80, 2, torch.deg2rad(torch.tensor(180.0))]])]
    h2n2params = [term.to(device, dtype) for term in h2n2params]
    pos = torch.tensor([[[-.1121, 0.0763, 0.000], [-0.0607, -0.0140, -0.000], [0.0607, 0.0140, -0.000], [.1121, -0.0763, 0.000]]]).to(device, dtype)
    cv = _coord2Cv(pos)

    energyFn = lambda config: energyCV(config, h2n2mass, h2n2charge, h2n2functs, h2n2idxs, h2n2params)

    ranges = torch.tensor([[-0.15, 0.15], [1e-5, 0.18], [0, 0.20], [-0.15, 0.15], [0, 1], [0, 0.14]]).to(device, dtype)

    T = torch.tensor(1.0).to(device, dtype)
    testY = torch.tensor([0.0995, 0, -0.0995]).to(device, dtype)
    testZ = torch.tensor([0.0, 0.11]).to(device, dtype)

    p1 = torch.tensor([0.045, 0]).to(device, dtype)
    p2 = torch.tensor([0.956, 0]).to(device, dtype)

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

    def V(configs):
        N = configs.shape[0]
        _configs = configs.reshape(N, 1, 2).repeat(1, batch, 1).reshape(-1, 2)

        z = prior.sample(batch * N, nvars=nvars, T=T, **priorParam)
        zlogProb = prior.logProbability(z, T=T, **priorParam)

        z = torch.cat([z, _configs], dim=-1)

        _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

        sample, ilogDet = source.TransformedDistribution.inverse(_sample, T=1, transformationList=[SigmoidCoupling], transformationParamList=cvTransformationParamList)

        E = energyFn(sample)
        loss = zlogProb - logDet - ilogDet +  E / T

        minIdx = torch.argmin(E.reshape(N, batch), dim=-1)
        sample = sample.reshape(N, batch, -1)[torch.arange(N), minIdx, :]
        loss = loss.reshape(N, batch).mean(-1)
        return loss, sample

    def Vgrad(configs):
        _B = int(np.ceil(configs.shape[0] / miniBatch))
        Vvalue = []
        dVdx = []
        sample = []
        for i in range(_B):
            _configs = configs[miniBatch * i : miniBatch * (i + 1)]
            _Vvalue, _sample = V(_configs)
            _dVdx = torch.autograd.grad(_Vvalue, _configs, grad_outputs=torch.ones(_configs.shape[0]).to(device), create_graph=False, retain_graph=False)[0]
            Vvalue.append(_Vvalue)
            dVdx.append(_dVdx)
            sample.append(_sample)
        Vvalue = torch.cat(Vvalue, dim=0)
        dVdx = torch.cat(dVdx, dim=0)
        sample = torch.cat(sample, dim=0)
        return Vvalue, dVdx, sample

    def neb(initial, final, V, Vgrad, numImages=25, k=10000000, lr=7e-5, maxSteps=1000, tol=1e-4, eps=1e-6, guess=None, optimizer=None):
        torch.set_grad_enabled(False)
        N = initial.shape[-1]
        if guess is None:
            path = torch.zeros(numImages, N).to(initial)

            path = initial + torch.arange(1, numImages+1).to(initial).unsqueeze(-1)/(numImages + 1) * (final - initial)
        else:
            path = guess

        prevPath = path.clone().detach()
        Vinitial, configInitial = V(initial.unsqueeze(0))
        Vfinal, configFinal = V(final.unsqueeze(0))

        pathFull = torch.cat([p1.unsqueeze(0), path, p2.unsqueeze(0)], dim=0)
        pathList = [pathFull.detach().cpu().numpy()]
        pathConfig = []
        VList = []

        path = torch.nn.Parameter(path.requires_grad_(True))
        if optimizer is None:
            optimizer = torch.optim.Adam([path], lr=lr)

        for step in range(maxSteps):
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            V, dVdx, configX = Vgrad(path)
            torch.set_grad_enabled(False)
            VList.append(torch.cat([Vinitial, V, Vfinal]).detach().cpu().numpy())
            _pathConfig = _cv2Coord(torch.cat([configInitial, configX, configFinal]))
            pathConfig.append(_pathConfig.detach().cpu().numpy())
            F = -dVdx

            xPrev = torch.roll(path, 1, 0)
            xPrev[0] = initial
            xNext = torch.roll(path, -1, 0)
            xNext[-1] = final

            Vprev = torch.roll(V, 1, 0)
            Vprev[0] = Vinitial.squeeze()
            Vnext = torch.roll(V, -1, 0)
            Vnext[-1] = Vfinal.squeeze()

            signMask = Vnext >= Vprev
            dVmax = torch.max(torch.abs(Vnext - V), torch.abs(Vprev - V))
            dVmin = torch.min(torch.abs(Vnext - V), torch.abs(Vprev - V))

            tauPlus = xNext - path
            tauPlus = tauPlus / torch.norm(tauPlus, dim=-1, keepdim=True)
            tauMinus = path - xPrev
            tauMinus = tauMinus / torch.norm(tauMinus, dim=-1, keepdim=True)

            tau = torch.zeros_like(tauPlus)
            tau[signMask] = tauPlus[signMask] * dVmax[signMask].unsqueeze(-1) + tauMinus[signMask] * dVmin[signMask].unsqueeze(-1)
            tau[~signMask] = tauPlus[~signMask] * dVmin[~signMask].unsqueeze(-1) + tauMinus[~signMask] * dVmax[~signMask].unsqueeze(-1)

            tauNorm = torch.norm(tau, dim=-1, keepdim=True)
            tau[(tauNorm < eps).squeeze()] = 0
            tau = tau / tauNorm

            Fperp = F - (F * tau).sum(-1, keepdim=True) * tau

            dxPrev = xPrev - path
            dxNext = xNext - path
            Fspring = k * (dxNext + dxPrev)

            Ftang = (Fspring * tau).sum(-1, keepdim=True) * tau

            Ftotal = Fperp + Ftang

            path.grad = -Ftotal

            optimizer.step()

            delta = path - prevPath

            prevPath = path.clone().detach()

            print(step, delta.norm().item(), path.max().item(), path.min().item())
            pathFull = torch.cat([p1.unsqueeze(0), path, p2.unsqueeze(0)], dim=0)
            pathList.append(pathFull.detach().cpu().numpy())
            if step % 20 == 0:
                plt.plot(pathFull[:, 0].detach().cpu().numpy(), pathFull[:, 1].detach().cpu().numpy())
                plt.savefig(os.path.join(pic_dir, 'cvPath'+str(step)+'.pdf'))
            if torch.abs(delta.norm().sum()) <= tol:
                break

        return np.array(pathList), np.array(VList), np.array(pathConfig)


    naivePath = torch.zeros(bins, nvars[0]).to(p1)
    naivePath = p1 + torch.arange(1, bins+1).to(p1).unsqueeze(-1)/(bins + 1) * (p2 - p1)

    Vinitial, configInitial = V(p1.unsqueeze(0))
    Vfinal, configFinal = V(p2.unsqueeze(0))

    Vvalue, configs = V(naivePath)

    naivePath = torch.cat([p1.unsqueeze(0), naivePath, p2.unsqueeze(0)], dim=0).detach().cpu().numpy()
    VList = torch.cat([Vinitial, Vvalue, Vfinal]).detach().cpu().numpy()
    pathConfig = _cv2Coord(torch.cat([configInitial, configs, configFinal])).detach().cpu().numpy()

    with open(os.path.join(args.load, 'NEBpathNaive.npy'), 'wb') as f:
        np.save(f, naivePath)
        np.save(f, VList)
        np.save(f, pathConfig)

    with open(os.path.join(args.load, 'estCVfreeE.npy'), 'rb') as f:
        cv = np.load(f)
        lnZLst = np.load(f)

    with torch.no_grad():
        highLine = 0.09
        upDown = bins//5
        guess = torch.zeros(bins, p1.shape[-1]).to(p1)
        guess[:, 0] = (p1 + torch.arange(1, bins+1).to(p1).unsqueeze(-1)/(bins + 1) * (p2 - p1))[:, 0]
        _tmp = torch.arange(1, upDown+1).to(p1).unsqueeze(-1)/(upDown + 1) * highLine
        guess[:, 1] = highLine
        guess[:upDown, 1] = _tmp.squeeze()
        guess[-upDown:, 1] = torch.flip(_tmp, dims=(0,)).squeeze()

    pathList, VList, pathConfig = neb(p1, p2, V, Vgrad, bins, k=nebK, maxSteps=maxSteps, guess=guess)
    with open(os.path.join(args.load, 'NEBpath.npy'), 'wb') as f:
        np.save(f, pathList)
        np.save(f, VList)
        np.save(f, pathConfig)
    path = pathList[-1]

    plt.figure()
    plt.contourf(cv[:, 0].reshape(lnZLst.shape), cv[:, 1].reshape(lnZLst.shape), lnZLst, levels=45, cmap=cmap)
    plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2, zorder=2)
    plt.savefig(os.path.join(pic_dir, 'cvPath.pdf'))
    plt.close()
