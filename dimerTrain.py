import os

from scope import source, flow, utils
from dimerExact import F as dimerF

import time
import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


def dimerVacuumSymWall(x, h=4, r0=2.5, s=1, base=0):
    r'''
    Two particles interact via a double-wall potential
    We manually fix one in (0, 0, 0)

    U = h * [1 - (r - r0 - s)**2 / s**2]**2
    where h = 4
          r0 = 2.5
          s = 1
    where r is the bond length.
    The energy unit is kBT

    x of the dimension [b, no, xyz]
    '''
    bondLength = torch.sum(x**2, dim=-1, keepdim=True)**0.5
    U = h * (1.0 - (bondLength - r0 - s)**2 / s**2)**2 - base
    return U


class DimerBondLength(flow.Bijector):
    r'''
    Compute the bond length of two particles reversiblly as a CV.
    '''
    @staticmethod
    def bijection(inverse, x, T, *args, **kwargs):
        r'''
        the inverse output x should be of [batch, 3], the x direction coord is also positive;
        the forward output z should be of [batch, 3], the first element is the bond length. The rest two are ratio of y^2 or z^2 w.r.t bond length, with in range of [-1, 1]. The negative sign means the y or z are of negative values.
        In the forward mod, the transformation is as follows:
        [x, y, z] --> [\sqrt(x^2 +y^2 +z^2), sign(y) y^2 / (x^2 + y^2), sign(z) z^2 / (x^2 + y^2 + z^2)]
        '''
        if not inverse:
            bondLength = torch.sum(x**2, dim=-1, keepdim=True)
            ratioZ = (x[:, -1:]**2 / bondLength) * torch.sign(x[:, -1:])
            _xy2 = torch.sum(x[:, :-1]**2, dim=-1, keepdim=True)
            ratioY = (x[:, 1:2]**2 / _xy2) * torch.sign(x[:, 1:2])
            jac2 = -2 * x / _xy2**2 * (x[:, 1:2]**2)
            jac2[:, 1] += 2 * x[:, 1] / _xy2.squeeze()
            jac2[:, -1] = 0
            jac2 *= torch.sign(x[:, 1:2])
            jac3 = -2 * x / bondLength**2 * (x[:, -1:]**2)
            jac3[:, -1] += 2 * x[:, -1] / bondLength.squeeze()
            jac3 *= torch.sign(x[:, -1:])
            bondLength = bondLength**0.5
            jac1 = x / bondLength
            jac = torch.cat([jac1.unsqueeze(1), jac2.unsqueeze(1), jac3.unsqueeze(1)], dim=1)
            logDet = torch.log(torch.det(jac)).unsqueeze(-1)
            return torch.cat([bondLength, ratioY, ratioZ], dim=-1), logDet
        else:
            bondLength = x[:, :1]**2
            zs = torch.abs(x[:, -1:] * bondLength)
            xys = bondLength - zs
            zs = zs**0.5 * torch.sign(x[:, -1:])
            ys = torch.abs(x[:, 1:2] * xys)
            xs = (xys - ys)**0.5
            ys = ys**0.5 * torch.sign(x[:, 1:2])
            orig = torch.cat([xs, ys, zs], dim=-1)

            jac1 = orig / x[:, :1]
            jac2 = -2 * orig / xys**2 * (orig[:, 1:2]**2)
            jac2[:, 1] += 2 * orig[:, 1] / xys.squeeze()
            jac2[:, -1] = 0
            jac2 *= torch.sign(orig[:, 1:2])
            jac3 = -2 * orig / bondLength**2 * (orig[:, -1:]**2)
            jac3[:, -1] += 2 * orig[:, -1] / bondLength.squeeze()
            jac3 *= torch.sign(orig[:, -1:])
            jac = torch.cat([jac1.unsqueeze(1), jac2.unsqueeze(1), jac3.unsqueeze(1)], dim=1)
            logDet = -torch.log(torch.det(jac)).unsqueeze(-1)
            return orig, logDet


if __name__ == "__main__":
    print(torch.seed())

    parser = argparse.ArgumentParser(description="Train the dimer free-energy model")
    parser.add_argument("-epoch", type=int, default=10000, help="number of training epochs")
    parser.add_argument("-epochSteps", type=int, default=35, help="optimization steps per epoch")
    parser.add_argument("-folder", default=None, help="path to save outputs")
    parser.add_argument("-device", type=int, default=-1, help="device, -1 for cpu, 0-N for i-th GPU, -2 for mps")
    args = parser.parse_args()

    T = torch.tensor(1.0)
    maxBond = 6
    minBond = 1
    testBondLen = torch.tensor([2., 2.5, 3.5, 4.5, 5])
    testT = torch.tensor([0.5, 0.7, 1.0])
    maxZ = 1.0 - 1e-7
    minZ = 0 + 1e-7
    maxT = 1.6
    minT = .3
    base = 0

    energyFn = lambda sample: dimerVacuumSymWall(sample, base=base)

    groundTruth = []
    for _T in testT:
        _T = _T.item()
        for _bond in testBondLen:
            _bond = _bond.item()
            groundTruth.append(-np.log(dimerF(_bond, _T, base=base)[0]))

    lr = 7.e-4
    eps = 1.e-8
    lrdecay = 0.997
    warmup = 500
    batchSize = 512
    evalBatchSize = 512
    maxIter = args.epoch
    stepNum = args.epochSteps
    saveStep = 50
    clipGrad = 0.0
    lamd1 = 5.0

    name = "dimer_fixed_b" + str(base)+ "_T" +str(minT) + "_" + str(maxT) +"_B" +str(minBond) + "_" + str(maxBond)
    rootFolder = args.folder if args.folder is not None else os.path.join("opt", name)
    utils.createWorkSpace(rootFolder)

    K = 50
    pwLinearNnet = 16
    pwLinearNetVector = [20, 50, 100, 150]

    if args.device == -1:
        device = torch.device("cpu")
    elif args.device == -2:
        device = torch.device("mps")
    else:
        device = torch.device(f"cuda:{args.device}")

    nvars = [2]
    prior = source.Uniform
    priorParam = {'low': minZ, 'high': maxZ}
    priorParam = source.Uniform.initalize(priorParam)

    maskList = []
    maskConpList = []
    netList = []
    for n in range(pwLinearNnet):
        b = torch.zeros(1, 3)
        bp = torch.ones(1, 3)
        b[:, n%2 + 1] = 1
        bp[:, n%2 + 1] = 0
        netList.append([utils.layer.SimpleMLPreshape([3, *pwLinearNetVector, 2 * K + 2], (len(pwLinearNetVector)) * [nn.ELU()] + [None], reshapeBack=True, shape=[-1, 2 * K + 2, 1])])
        maskList.append(b)
        maskConpList.append(bp)
    maskList = torch.cat(maskList, 0).to(torch.uint8)
    maskConpList = torch.cat(maskConpList, 0).to(torch.uint8)

    boundary = (torch.tensor(minZ), torch.tensor(maxZ), torch.tensor(minZ), torch.tensor(maxZ))
    sections = (K, K, 1, 1)
    spline = utils.spline.SteffenSplineFn
    splineAllParams = utils.spline.SteffenSplineFn.initalize()

    transformation = flow.SplineFlow
    transformationParams = flow.SplineFlow.initalize(maskList=maskList, maskConpList=maskConpList, networkList=netList, sections=sections, boundary=boundary, spline=spline, splineAllParams=splineAllParams)

    transformationList = [transformation]
    transformationParamList = [transformationParams]

    transformationParamList = utils.put(transformationParamList, device)
    priorParam = utils.put(priorParam, device)

    bestTrainLoss = 99999999

    params = list(utils.yieldTrainable([priorParam, transformationParamList]))
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print('total nubmer of trainable parameters:', nparams)

    optimizer = torch.optim.Adamax(params, lr=lr)

    def lr_lambda(epoch):
        return min(1.,1.) * np.power(lrdecay, epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.8)

    LOSS = []

    for e in range(maxIter):
        tstart = time.time()
        for s in range(stepNum):
            optimizer.zero_grad()
            bondLen = torch.rand(batchSize, 1) * (maxBond - minBond) + minBond
            bondLen = bondLen.to(device)
            Tsample = torch.rand(batchSize, 1) * (maxT - minT) + minT
            Tsample = Tsample.to(device)

            z = prior.sample(batchSize, nvars=nvars, T=Tsample, **priorParam)
            zlogProb = prior.logProbability(z, T=Tsample, **priorParam)

            z = torch.cat([bondLen, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=Tsample, transformationList=transformationList, transformationParamList=transformationParamList)

            sample, ilogDet = DimerBondLength.inverse(_sample, T=Tsample)

            if torch.any(torch.isnan(ilogDet)) or torch.any(torch.isinf(ilogDet)):
                raise RuntimeError("Dimer inverse Jacobian produced NaN/Inf values")

            loss = zlogProb - logDet - ilogDet + energyFn(sample) / Tsample
            loss = loss.mean()

            loss.backward()
            optimizer.step()

        trainTime = time.time() - tstart

        lossLst = []
        for _T in testT:
            _T = _T.repeat(batchSize).unsqueeze(-1).to(device)
            for _bondLen in testBondLen:
                _bondLen = _bondLen.repeat(batchSize).unsqueeze(-1).to(device)
                with torch.no_grad():
                    z = prior.sample(batchSize, nvars=nvars, T=_T, **priorParam)
                    zlogProb = prior.logProbability(z, T=_T, **priorParam)

                    z = torch.cat([_bondLen, z], dim=-1)

                    _sample, logDet = source.TransformedDistribution.forward(z, T=_T, transformationList=transformationList, transformationParamList=transformationParamList)

                    sample, ilogDet = DimerBondLength.inverse(_sample, T=_T)

                    loss = zlogProb - logDet - ilogDet + energyFn(sample) / _T
                lossLst.append(loss.mean().detach().item())

        lossLst = np.array(lossLst)
        lossSum = lossLst.sum()

        LOSS.append(lossSum)
        if lossSum < bestTrainLoss:
            bestTrainLoss = lossSum.item()
            torch.save([priorParam, transformationParamList], os.path.join(rootFolder, 'best_TrainLoss_joint.saving'))
            print("--> Updated best model: ", bestTrainLoss)

        printString = "epoch: {:d}, L: {:.5f}, "
        printString += "time: {:.2f}, best: {:.5f}"
        resultLst = [e, lossSum]
        resultLst += [trainTime, bestTrainLoss]
        print(printString.format(*resultLst))
        for idxT, _T in enumerate(testT):
            printString = ' >> T_{:.1f}'.format(_T.item()) + "_dimer"
            resultLst = []
            for idx, _bondLen in enumerate(testBondLen):
                printString += "@B" + str(_bondLen.item()) + ":{:.2f}" + "/{:.2f}"
                resultLst += [lossLst[idx + len(testBondLen) * idxT].item()]
                resultLst += [groundTruth[idx + len(testBondLen) * idxT]]
            print(printString.format(*resultLst))

        scheduler.step()

        if e % saveStep == 0 or e == 0:
            torch.save([priorParam, transformationParamList], os.path.join(rootFolder, 'savings', name + "_epoch_" + str(e) + ".saving"))
            with h5py.File(os.path.join(rootFolder, "records", "LOSS"+'.hdf5'), 'w') as f:
                f.create_dataset("LOSS", data=np.array(LOSS))
            lossfig = plt.figure(figsize=(8, 5))
            lossax = lossfig.add_subplot(111)
            epoch = len(LOSS)
            lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss", markersize=2.5)
            lossax.set_xlim(0, epoch)
            lossax.legend()
            lossax.set_title("Loss Curve")
            plt.savefig(os.path.join(rootFolder, 'pic', 'lossCurve.png'), bbox_inches="tight", pad_inches=0)
            plt.close()
            utils.cleanSaving(rootFolder, e, 6 * saveStep, name)
