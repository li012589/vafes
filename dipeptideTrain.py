import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import argparse, json, h5py, glob

from forceUtils.energy import energy
from dipeptideEnergy import mass, charge, functs, idxs, params, concise2full


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOADV = os.path.join(SCRIPT_DIR, 'etc', 'dipeptideMeta.npz')


def resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path.removeprefix('./'))


if __name__ == "__main__":
    rngseed = torch.seed()
    torch.manual_seed(rngseed)
    print("Using torch seed:", rngseed)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-folder", default=None, help="path to save and load folder")
    parser.add_argument("-device", type=int, default=-1, help="device, -1 for cpu, 0-N for i-th GPU, -2 for mps")
    parser.add_argument("-load", action='store_true', help="if load or not")
    parser.add_argument("-double", action='store_true', help="float64 or float32")

    group = parser.add_argument_group("learning parameters")
    group.add_argument("-lr", type=float, default=7e-4, help="learning rate")
    group.add_argument("-eps", type=float, default=1e-8, help="eps for adam")
    group.add_argument("-lrdecay", type=float, default=0.997, help="decay learning rate")
    group.add_argument("-warmup", type=int, default=500, help="epoch before learning rate changing")
    group.add_argument("-epoch", type=int, default=10000, help="epoch/train steps")
    group.add_argument("-epochSteps", type=int, default=35, help="opt steps in one epoch")
    group.add_argument("-batch", type=int, default=512, help="batch size")
    group.add_argument("-evalBatch", type=int, default=512, help="batch size for evaluation")
    group.add_argument("-saveStep", type=int, default=50, help="save model per steps")
    group.add_argument("-clipGrad", type=float, default=-1, help="maximum scale of gradient, negative for not activated")

    group = parser.add_argument_group("model parameters")
    group.add_argument("-K", type=int, default=50, help="bin number of nsf flow")
    group.add_argument("-layer", type=int, default=16, help="num of transformation layers")
    group.add_argument("-mlpVector", default=[128, 256, 512, 1024], type=int, nargs="+", help="hidden dim of MLP used in spline flow")

    group = parser.add_argument_group("target parameters")
    parser.add_argument("-loadV", default=DEFAULT_LOADV, help="path to load V matrix from TICA")
    group.add_argument("-T", type=float, default=1, help="temperature")

    args = parser.parse_args()

    if args.folder is None:
        rootFolder = "./opt/dipeptide_T" + str(args.T) + "_b" + str(args.K) + "_n" + str(args.layer) + "_" + str(args.mlpVector)
        print("No specified saving path, using", rootFolder)
    else:
        rootFolder = args.folder
        print("Using", rootFolder)
    utils.createWorkSpace(rootFolder)

    open(os.path.join(rootFolder, "output.log"), "w").close()

    def print_to_both(*args, **kwargs):
        with open(os.path.join(rootFolder, 'output.log'), 'a') as file:
            original_print(*args, file=file)
        original_print(*args)

    original_print = __builtins__.print
    __builtins__.print = print_to_both

    for f in glob.glob("*.seed", root_dir=rootFolder):
        print("delete old seed:", f)
        os.remove(os.path.join(rootFolder, f))
    try:
        open(os.path.join(rootFolder, str(rngseed) + ".seed"), "w").close()
    except:
        print("failed to save seed")

    if not args.load:
        with open(os.path.join(rootFolder, "parameter.json"), "w") as f:
            config = vars(args)
            json.dump(config, f)
    else:
        with open(os.path.join(rootFolder, "parameter.json"), "r") as f:
            config = json.load(f)

    locals().update(config)
    loadV = resolve_repo_path(loadV)

    if args.device == -1:
        device = "cpu"
    elif args.device == -2:
        device = "mps"
    else:
        device = "cuda:"+str(args.device)
    device = torch.device(device)

    if args.double:
        dtype = torch.float64
    else:
        dtype = torch.float32

    mass = mass.to(device, dtype)
    charge = charge.to(device, dtype)
    idxs = [term.to(device) for term in idxs]
    params = [term.to(device, dtype) for term in params]
    energyFn = lambda config: energy(concise2full(config), mass, charge, functs, idxs, params)

    projV, projMean, ranges = np.load(loadV).values()
    np.savez(os.path.join(rootFolder, 'projV'), projV, projMean, ranges)
    projV, projMean, ranges = torch.from_numpy(projV).to(device, dtype), torch.from_numpy(projMean).to(device, dtype), torch.from_numpy(ranges).to(device, dtype)

    invProjV = torch.linalg.inv(projV)

    T = torch.tensor(T).to(device, dtype)
    test1 = torch.tensor([-0.112, -0.03, 0.1, -0.06, -0.05, 0.012, 0, 0])
    test2 = torch.tensor([-0.112, -0.05, 0.1, -0.05, -0.085, -0.05, 0.1, 0.05])
    max1 = ranges[0, 1]
    min1 = ranges[0, 0]
    max2 = ranges[1, 1]
    min2 = ranges[1, 0]

    nvars = [31]
    prior = source.Uniform
    priorParam = {'low': ranges[2:, 0], 'high': ranges[2:, 1], 'outBoundE': 0}
    priorParam = source.Uniform.initalize(priorParam)

    maskList = []
    maskConpList = []
    netList = []
    boundaryList = []
    for n in range(layer):
        if n % 2 == 0:
            b = torch.zeros(1, 33).bool().to(device)
            bp = torch.ones(1, 33).bool().to(device)
            b[:, [2+n*2 for n in range(16)]] = 1
            bp[:, [2+n*2 for n in range(16)]] = 0
            boundaryList.append((torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b), torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b)))
            netList.append([utils.layer.SimpleMLPreshape([18, *mlpVector, (2 * K + 2) * 16], (len(mlpVector)) * [nn.ELU()] + [None], reshapeBack=True, shape=[-1, 2 * K + 2, 2])])
        else:
            b = torch.zeros(1, 33).bool().to(device)
            bp = torch.ones(1, 33).bool().to(device)
            b[:, [3+n*2 for n in range(15)]] = 1
            bp[:, [3+n*2 for n in range(15)]] = 0
            boundaryList.append((torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b), torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b)))
            netList.append([utils.layer.SimpleMLPreshape([19, *mlpVector, (2 * K + 2) * 15], (len(mlpVector)) * [nn.ELU()] + [None], reshapeBack=True, shape=[-1, 2 * K + 2, 2])])
        maskList.append(b)
        maskConpList.append(bp)
    maskList = torch.cat(maskList, 0).to(torch.uint8)
    maskConpList = torch.cat(maskConpList, 0).to(torch.uint8)

    sections = (K, K, 1, 1)
    spline = utils.spline.SteffenSplineFn
    splineAllParams = utils.spline.SteffenSplineFn.initalize()

    transformation = flow.SplineFlow
    transformationParams = flow.SplineFlow.initalize(maskList=maskList, maskConpList=maskConpList, networkList=netList, sections=sections, boundaryList=boundaryList, spline=spline, splineAllParams=splineAllParams)

    transformationList = [transformation]
    transformationParamList = [transformationParams]

    transformationParamList = utils.put(transformationParamList, device)
    priorParam = utils.put(priorParam, device)

    bestTrainLoss = 99999999

    paramsList = list(utils.yieldTrainable([transformationParamList]))
    paramsList = list(filter(lambda p: p.requires_grad, paramsList))
    nparams = sum([np.prod(p.size()) for p in paramsList])
    print('total nubmer of trainable parameters:', nparams)

    optimizer = torch.optim.Adamax(paramsList, lr=lr)

    def lr_lambda(epoch):
        return min(1.,1.) * np.power(lrdecay, epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.8)

    LOSS = []
    for e in range(epoch):
        tstart = time.time()
        for s in range(epochSteps):
            optimizer.zero_grad()
            cv1 = torch.rand(batch, 1).to(device, dtype) * (max1 - min1) + min1
            cv2 = torch.rand(batch, 1).to(device, dtype) * (max2 - min2) + min2

            z = prior.sample(batch, nvars=nvars, T=T, **priorParam)
            zlogProb = prior.logProbability(z, T=T, **priorParam)

            z = torch.cat([cv1, cv2, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

            sample = _sample @ invProjV + projMean

            if (
                torch.any(torch.isnan(sample))
                or torch.any(torch.isnan(logDet))
                or torch.any(torch.isinf(sample))
                or torch.any(torch.isinf(logDet))
            ):
                raise RuntimeError("Dipeptide training produced invalid sample/Jacobian values")

            loss = zlogProb - logDet + energyFn(sample) / T
            loss = loss.mean()

            loss.backward()
            optimizer.step()

        trainTime = time.time() - tstart

        lossLst = []
        for idx in range(len(test1)):
            _cv1 = test1[idx].repeat(evalBatch).unsqueeze(-1).to(device)
            _cv2 = test2[idx].repeat(evalBatch).unsqueeze(-1).to(device)
            with torch.no_grad():
                z = prior.sample(evalBatch, nvars=nvars, T=T, **priorParam)
                zlogProb = prior.logProbability(z, T=T, **priorParam)

                z = torch.cat([_cv1, _cv2, z], dim=-1)

                _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

                sample = _sample @ invProjV + projMean
                loss = zlogProb - logDet + energyFn(sample) / T

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
        for idx in range(len(test1)):
            printString = 'ALDP >> @A_{:.1f}'.format(test1[idx].item())
            resultLst = []
            printString += "@B_{:.1f}".format(test2[idx].item()) + ":{:.2f}  "
            resultLst += [lossLst[idx].item()]
            print(printString.format(*resultLst))

        scheduler.step()

        if e % saveStep == 0 or e == 0:
            torch.save([priorParam, transformationParamList], os.path.join(rootFolder, 'savings', 'ALDPNF' + '_epoch_' + str(e) + ".saving"))
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
            utils.cleanSaving(rootFolder, e, 6 * saveStep, 'ALDPNF')
