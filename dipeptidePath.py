import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import argparse
import json

from forceUtils.energy import energy
from dipeptideEnergy import mass, charge, functs, idxs, params, concise2full


if __name__ == '__main__':
    device = torch.device('cpu')
    dtype = torch.float32
    batch = 256
    bins = 100
    cmap = 'viridis'
    maxSteps = 10000
    miniBatch = 64

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-load", default=None, help="path to load free energy model")
    parser.add_argument("-loadV", default=None, help="path to load V matrix from TICA")
    args = parser.parse_args()

    pic_dir = os.path.join(args.load, 'pic')
    os.makedirs(pic_dir, exist_ok=True)

    with open(os.path.join(args.load, "parameter.json"), "r") as f:
        config = json.load(f)

    mass = mass.to(device, dtype)
    charge = charge.to(device, dtype)
    idxs = [term.to(device) for term in idxs]
    params = [term.to(device, dtype) for term in params]
    energyFn = lambda config: energy(concise2full(config), mass, charge, functs, idxs, params)

    if args.loadV is None:
        projV, projMean, ranges = np.load(os.path.join(args.load, 'projV.npz')).values()
    else:
        projV, projMean, ranges = np.load(args.loadV).values()
    projV, projMean, ranges = torch.from_numpy(projV).to(device, dtype), torch.from_numpy(projMean).to(device, dtype), torch.from_numpy(ranges).to(device, dtype)

    invProjV = torch.linalg.inv(projV)

    T = torch.tensor(config['T']).to(device, dtype)

    p1 = torch.tensor([-0.112, -0.115]).to(device)
    p2 = torch.tensor([0.1, 0.11]).to(device)
    max1 = ranges[0, 1]
    min1 = ranges[0, 0]
    max2 = ranges[1, 1]
    min2 = ranges[1, 0]

    nvars = [31]
    prior = source.Uniform
    transformationList = [flow.SplineFlow]
    priorParam, transformationParamList = torch.load(os.path.join(args.load, "best_TrainLoss_joint.saving"), map_location=device, weights_only=False)

    # Add missing parameters for backward compatibility with old saved models
    # Newer versions of SplineFlow require eps, minLog, and linearBound
    for tfm_params in transformationParamList:
        if 'eps' not in tfm_params:
            tfm_params['eps'] = 1e-7
        if 'minLog' not in tfm_params:
            tfm_params['minLog'] = -50.0
        if 'linearBound' not in tfm_params:
            tfm_params['linearBound'] = False

    def V(configs):
        N = configs.shape[0]
        _configs = configs.reshape(N, 1, 2).repeat(1, batch, 1).reshape(-1, 2)

        z = prior.sample(batch * N, nvars=nvars, T=T, **priorParam)
        zlogProb = prior.logProbability(z, T=T, **priorParam)

        z = torch.cat([_configs, z], dim=-1)

        _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

        sample = _sample @ invProjV + projMean

        E = energyFn(sample)
        loss = (zlogProb - logDet) + E / T

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
            if _B > 1 and (i + 1) % max(1, _B // 5) == 0:
                print(f"  Vgrad batch {i + 1}/{_B} ({(i + 1) / _B * 100:.1f}%)")
        Vvalue = torch.cat(Vvalue, dim=0)
        dVdx = torch.cat(dVdx, dim=0)
        sample = torch.cat(sample, dim=0)
        return Vvalue, dVdx, sample

    def neb(initial, final, V, Vgrad, numImages=25, k=10000000, lr=7e-5, maxSteps=1000, tol=1e-4, eps=1e-6, optimizer=None):
        torch.set_grad_enabled(False)
        N = initial.shape[-1]
        path = torch.zeros(numImages, N).to(initial)

        path = initial + torch.arange(1, numImages+1).to(initial).unsqueeze(-1)/(numImages + 1) * (final - initial)

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

        print(f"Starting NEB optimization with maxSteps={maxSteps}, numImages={numImages}")
        for step in range(maxSteps):
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            V, dVdx, configX = Vgrad(path)
            torch.set_grad_enabled(False)
            VList.append(torch.cat([Vinitial, V, Vfinal]).detach().cpu().numpy())
            _pathConfig = concise2full(torch.cat([configInitial, configX, configFinal]))
            pathConfig.append(_pathConfig.detach().cpu().numpy())
            F = -dVdx

            if (step + 1) % max(1, maxSteps // 10) == 0 or step == maxSteps - 1:
                path_norm = path.norm().item()
                delta_norm = (path - prevPath).norm().item()
                print(f"NEB Step {step + 1}/{maxSteps} ({(step + 1) / maxSteps * 100:.1f}%) - Path norm: {path_norm:.6f}, Delta: {delta_norm:.6e}")

            xPrev = torch.roll(path, 1, 0)
            xPrev[0] = initial
            xNext = torch.roll(path, -1, 0)
            xNext[-1] = final

            Vprev = torch.roll(V, 1, 0)
            Vprev[0] = Vinitial
            Vnext = torch.roll(V, -1, 0)
            Vnext[-1] = Vfinal

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


    with open(os.path.join(args.load, 'estCVfreeE.npy'), 'rb') as f:
        cv = np.load(f)
        lnZLst = np.load(f)

    print(f"\nStarting NEB calculation...")
    print(f"Start point: p1 = [{p1[0].item():.4f}, {p1[1].item():.4f}]")
    print(f"End point: p2 = [{p2[0].item():.4f}, {p2[1].item():.4f}]")
    print(f"Number of images: {bins}")
    print(f"Max steps: {maxSteps}")
    print(f"Device: {device}\n")

    pathList, VList, pathConfig = neb(p1, p2, V, Vgrad, bins, maxSteps=maxSteps)
    
    print(f"\nNEB calculation completed!")
    print(f"Total steps taken: {len(pathList)}")
    print(f"Saving results...")
    
    with open(os.path.join(args.load, 'NEBpath.npy'), 'wb') as f:
        np.save(f, pathList)
    
    print(f"Results saved to: {os.path.join(args.load, 'NEBpath.npy')}")
    path = pathList[-1]

    plt.figure()
    plt.contourf(cv[:, 0].reshape(lnZLst.shape), cv[:, 1].reshape(lnZLst.shape), lnZLst, levels=45, cmap=cmap)
    plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2, zorder=2)
    plt.savefig(os.path.join(pic_dir, 'cvPath.pdf'))
    plt.close()
