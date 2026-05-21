import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import argparse, json, h5py, glob

from nextForce.frontend import fromOpenMM, energy, energyContributions
from chignolinEnergy import ProteinConciseExpression, addHydrogen
from network import ResNet1d


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HELPER_FILE = os.path.join(SCRIPT_DIR, 'etc', 'chignolinMeta.npz')


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
    parser.add_argument("-retrain", default=None, help="path to save and load folder")
    parser.add_argument("-loadOpt", action='store_true', help="load optimizer state when retrain")
    parser.add_argument("-double", action='store_true', help="float64 or float32")

    group = parser.add_argument_group("learning parameters")
    group.add_argument("-lr", type=float, default=7e-4, help="learning rate")
    group.add_argument("-eps", type=float, default=1e-8, help="eps for adam")
    group.add_argument("-lrdecay", type=float, default=0.997, help="decay learning rate")
    group.add_argument("-warmup", type=int, default=500, help="epoch before learning rate changing")
    group.add_argument("-epoch", type=int, default=10000, help="epoch/train steps")
    group.add_argument("-epochSteps", type=int, default=35, help="opt steps in one epoch")
    group.add_argument("-batch", type=int, default=64, help="batch size")
    group.add_argument("-evalBatch", type=int, default=256, help="batch size for evaluation")
    group.add_argument("-saveStep", type=int, default=50, help="save model per steps")
    group.add_argument("-clipGrad", type=float, default=-1, help="maximum scale of gradient, negative for not activated")
    group.add_argument("-noReg", type=float, default=None, help="if set, rollback when loss > bestTrainLoss * (1 - noReg)")

    group = parser.add_argument_group("model parameters")
    group.add_argument("-K", type=int, default=128, help="bin number of nsf flow")
    group.add_argument("-mlpVector", default=[512, 512], type=int, nargs="+", help="hidden dim of MLP used in spline flow")
    group.add_argument("-couplingLayer1", type=int, default=12, help="how many coupling layers")
    group.add_argument("-couplingLayer2", type=int, default=10, help="how many coupling layers")
    group.add_argument("-kernelSize", type=int, default=47, help="size of the cnn kernels")
    group.add_argument("-channels", type=int, default=16, help="hidden channel dimension")
    group.add_argument("-hiddenChannels", type=int, default=64, help="hidden channel dimension")
    group.add_argument("-hiddenConvLayers", type=int, default=4, help="num of hidden cnn layers")
    group.add_argument("-hiddenWidth", type=int, default=64, help="dimension of the hidden mlp")
    group.add_argument("-hiddenFcLayers", type=int, default=2, help="num of hidden Fc layers")

    group = parser.add_argument_group("target parameters")
    parser.add_argument("-loadetc", default=DEFAULT_HELPER_FILE, help="path to load etc info")
    group.add_argument("-beta", type=float, default=0.4, help="temperature")
    group.add_argument("-betalow", type=float, default=5, help="low temperature start point in training")
    group.add_argument("-betahigh", type=float, default=0.05, help="high temperature end point in training")
    group.add_argument("-betaPTlow", type=float, default=0.7, help="low temperature start point in training")
    group.add_argument("-betaPThigh", type=float, default=0.1, help="high temperature end point in training")
    group.add_argument("-addMesh", type=int, default=None, help="create n-by-n mesh grid in cv12 space and combine with random samples")
    group.add_argument("-betaCV", action='store_true', help="use betaCV (beta + cv12) as conditioning, adds 2 dims to network input")

    args = parser.parse_args()

    if args.folder is None:
        rootFolder = "chignolinTrain_beta" + str(args.beta) + "_b" + str(args.K) + "_cp" + str(args.couplingLayer1) + '_' + str(args.couplingLayer2) + "_" + str(args.mlpVector) + "_ks" + str(args.kernelSize) + "_c" + str(args.channels) + '_' + str(args.hiddenChannels) + "_" + str(args.hiddenWidth) + '_n' + str(args.hiddenConvLayers) + '_' +str(args.hiddenFcLayers) + '_Tr' + str(args.betahigh) + '_' + str(args.betalow) + '_' + str(args.betaPThigh) + '_' + str(args.betaPTlow)
        if args.retrain:
            rootFolder = "retrain_" + rootFolder
        rootFolder = os.path.join('opt', rootFolder)
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
    loadetc = resolve_repo_path(loadetc)

    extraInputDims = 2 if betaCV else 0

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

    proteinEnergyParams = fromOpenMM(['amber14-all.xml', 'implicit/gbn2.xml'], os.path.join(SCRIPT_DIR, 'etc', 'geoOpt.pdb'), eps=1e-7, device=device)

    helperFile = np.load(loadetc)

    refHeavy = helperFile['refHeavy']
    refHydrogen = helperFile['refHydrogen']
    Hidx = helperFile['Hidx']
    heavyIdx = helperFile['heavyIdx']
    Hs = helperFile['Hs']
    idxMaj = helperFile['idxMaj']
    ranges = helperFile['ranges']

    np.savez(os.path.join(rootFolder, 'etc'),
             refHeavy=refHeavy, refHydrogen=refHydrogen,
             Hidx=Hidx, heavyIdx=heavyIdx,
             Hs=Hs, idxMaj=idxMaj,
             ranges=ranges)
    refHeavy = torch.from_numpy(refHeavy).to(device, dtype)
    refHydrogen = torch.from_numpy(refHydrogen).to(device, dtype)
    Hidx = torch.from_numpy(Hidx).to(device)
    heavyIdx = torch.from_numpy(heavyIdx).to(device)
    Hs = torch.from_numpy(Hs).to(device)
    idxMaj = torch.from_numpy(idxMaj).to(device)
    ranges = torch.from_numpy(ranges).to(device, dtype)

    energyContrFn = lambda config: energyContributions(addHydrogen(config,
                                                                   refHeavy, refHydrogen, Hidx, heavyIdx, Hs, idxMaj) / 10,
                                                       *proteinEnergyParams)
    energyFn = lambda config: energy(addHydrogen(config,
                                                 refHeavy, refHydrogen, Hidx, heavyIdx, Hs, idxMaj) / 10,
                                     *proteinEnergyParams)

    beta = torch.tensor(beta).to(device, dtype) # base temperature

    max1 = 17
    min1 = 3.4
    max2 = 10
    min2 = 4.5

    test1 = torch.linspace(min1, max1, 7)
    test2 = torch.linspace(min2, max2, 7)
    test1, test2 = torch.meshgrid(test1, test2)
    test1 = test1.reshape(-1)
    test2 = test2.reshape(-1)

    test1 = torch.tensor([5.4, 12.5, 13, 5.5, 7.2, 8, 12.5]).to(device)
    test2 = torch.tensor([5.5, 5.6, 9, 7.2, 7.3, 5.5, 7]).to(device)

    bestTrainLoss = 99999999
    if args.retrain is None:
        N = 223
        nvars = [N]
        prior = source.TruncatedGaussian
        _mu = (ranges[2:, 0] + ranges[2:, 1]) / 2
        _logsigma = torch.randn(N) / np.sqrt(N) + 1
        priorParam = prior.initalize({'low': ranges[2:, 0], 'high': ranges[2:, 1], 'mu': _mu, 'logsigma': _logsigma})
        muNet = utils.layer.SimpleMLP([3 + extraInputDims, *mlpVector, N], (len(mlpVector)) * [nn.ELU()] + [nn.Sigmoid()])
        sigmaNet = utils.layer.SimpleMLP([3 + extraInputDims, *mlpVector, N], (len(mlpVector)) * [nn.ELU()] + [nn.ELU()])

        maskList = []
        maskConpList = []
        netList = []
        boundaryList = []

        b = torch.zeros(1, 225).bool().to(device)
        bp = torch.ones(1, 225).bool().to(device)

        b[:, 2:6] = 1
        bp[:, 2:] = 0
        netList.append([ResNet1d([3 + extraInputDims] + mlpVector + [channels * 4], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
        boundaryList.append((torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1)), torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1))))
        maskList.append(b)
        maskConpList.append(bp)

        b = torch.zeros(1, 225).bool().to(device)
        bp = torch.ones(1, 225).bool().to(device)

        b[:, 6:42] = 1
        bp[:, 6:] = 0
        netList.append([ResNet1d([7 + extraInputDims] + mlpVector + [channels * 36], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
        boundaryList.append((torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1)), torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1))))
        maskList.append(b)
        maskConpList.append(bp)

        b = torch.zeros(1, 225).bool().to(device)
        bp = torch.ones(1, 225).bool().to(device)

        b[:, 42:55] = 1
        bp[:, 42:] = 0
        netList.append([ResNet1d([43 + extraInputDims] + mlpVector + [channels * 13], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
        boundaryList.append((torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1)), torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1))))
        maskList.append(b)
        maskConpList.append(bp)

        for n in range(couplingLayer1):
            b = torch.zeros(1, 225).bool().to(device)
            bp = torch.ones(1, 225).bool().to(device)
            if n % 2 == 0:
                b[:, 42:55] = 1
                bp[:, 42:] = 0
                netList.append([ResNet1d([43 + extraInputDims] + mlpVector + [channels * 13], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
            else:
                b[:, 2:42] = 1
                bp[:, 2:42] = 0
                bp[:, 55:] = 0
                netList.append([ResNet1d([16 + extraInputDims] + mlpVector + [channels * 40], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
            boundaryList.append((torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1)), torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1))))
            maskList.append(b)
            maskConpList.append(bp)

        b = torch.zeros(1, 225).bool().to(device)
        bp = torch.ones(1, 225).bool().to(device)

        b[:, 55:] = 1
        bp[:, 55:] = 0
        netList.append([ResNet1d([56 + extraInputDims] + mlpVector + [channels * 170], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
        boundaryList.append((torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1)), torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1))))
        maskList.append(b)
        maskConpList.append(bp)

        for n in range(couplingLayer2):
            b = torch.zeros(1, 225).bool().to(device)
            bp = torch.ones(1, 225).bool().to(device)
            if n % 2 == 0:
                b[:, [2+n*2 for n in range(112)]] = 1
                bp[:, [2+n*2 for n in range(112)]] = 0
                netList.append([ResNet1d([114 + extraInputDims] + mlpVector + [channels * 112], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
            else:
                b[:, [3+n*2 for n in range(111)]] = 1
                bp[:, [3+n*2 for n in range(111)]] = 0
                netList.append([ResNet1d([115 + extraInputDims] + mlpVector + [channels * 111], channels, kernelSize, 2 * K + 2, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ELU())])
            boundaryList.append((torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1)), torch.masked_select(ranges[:, 0], b.reshape(-1)), torch.masked_select(ranges[:, 1], b.reshape(-1))))
            maskList.append(b)
            maskConpList.append(bp)

        maskList = torch.cat(maskList, 0).to(torch.uint8).reshape(-1, 225)
        maskConpList = torch.cat(maskConpList, 0).to(torch.uint8).reshape(-1, 225)

        sections = (K, K, 1, 1)
        spline = utils.spline.SteffenBernsteinSplineFn
        splineAllParams = utils.spline.SteffenBernsteinSplineFn.initalize()

        transformation = flow.SplineFlow
        transformationParams = flow.SplineFlow.initalize(maskList=maskList, maskConpList=maskConpList, networkList=netList, sections=sections, boundaryList=boundaryList, spline=spline, splineAllParams=splineAllParams)

        transformationList = [transformation]
        transformationParamList = [transformationParams]

        transformationParamList = utils.put(transformationParamList, device)
        priorParam = utils.put(priorParam, device)
        muNet = utils.put(muNet, device)
        sigmaNet = utils.put(sigmaNet, device)
    else:
        nvars = [223]
        prior = source.TruncatedGaussian
        transformationList = [flow.SplineFlow]
        priorParam, muNet, sigmaNet, transformationParamList = torch.load(os.path.join(args.retrain, "best_TrainLoss_joint.saving"), map_location=device, weights_only=False)

        lossLst = []
        energyLst = []
        energyMeanLst = []
        for idx in range(len(test1)):
            _cv1 = test1[idx].reshape(1, 1).to(device)
            _cv2 = test2[idx].reshape(1, 1).to(device)
            with torch.no_grad():
                _cv12 = torch.cat([_cv1, _cv2], dim=-1)
                _cv12 = _cv12.repeat(evalBatch, 1)
                beta_ = torch.Tensor([[beta]]).repeat(evalBatch, 1).to(_cv12)

                if betaCV:
                    beta_input = torch.cat([beta_, _cv12], dim=-1)
                else:
                    beta_input = beta_

                z = source.Uniform.sample(evalBatch, nvars=nvars, T=1, low=priorParam['low'], high=priorParam['high'])
                zlogProb = source.Uniform.logProbability(z, T=1, low=priorParam['low']-1e-5, high=priorParam['high']+1e-5)

                z = torch.cat([_cv12, z], dim=-1)

                _sample, logDet = source.TransformedDistribution.forward(z, T=beta_input, transformationList=transformationList, transformationParamList=transformationParamList)

                sample, ilogDet = ProteinConciseExpression.inverse(_sample, T=beta)
                es = energyFn(sample)
                loss = zlogProb - logDet - ilogDet + es * beta

            lossLst.append(loss.mean().detach().item())
            energyLst.append(es.min().item())
            energyMeanLst.append(es.mean().item())

        lossLst = np.array(lossLst)
        lossSum = lossLst.mean()
        energyLst = np.array(energyLst)
        energyMeanLst = np.array(energyMeanLst)

        if lossSum < bestTrainLoss:
            bestTrainLoss = lossSum.item()
            torch.save([priorParam, muNet, sigmaNet, transformationParamList], os.path.join(rootFolder, 'best_TrainLoss_joint.saving'))
            print("--> Updated best model: ", bestTrainLoss)

        printString = "epoch: {:d}, L: {:.5f}, "
        printString += "time: {:.2f}, best: {:.5f}"
        resultLst = [-1, lossSum]
        resultLst += [-1, bestTrainLoss]
        print(printString.format(*resultLst))

        for idx in range(len(test1)):
            printString = 'Protein >> @D1_{:.1f}'.format(test1[idx].item())
            resultLst = []
            printString += "@D2_{:.2f}".format(test2[idx].item()) + ":{:.2f}, e: {:.2f}, eb:{:.2f}"
            resultLst += [lossLst[idx].item(), energyLst[idx].item(), energyMeanLst[idx].item()]
            print(printString.format(*resultLst))

    paramsList = list(utils.yieldTrainable([muNet, sigmaNet, transformationParamList]))
    paramsList = list(filter(lambda p: p.requires_grad, paramsList))
    nparams = sum([np.prod(p.size()) for p in paramsList])
    print('total nubmer of trainable parameters:', nparams)

    if args.retrain and args.loadOpt:
        savedOpt = torch.load(os.path.join(args.retrain, 'best_Train_opt.saving'), map_location='cpu', weights_only=False)
        optimizer = torch.optim.Adamax(paramsList, lr=lr)
        optimizer.load_state_dict(savedOpt.state_dict())
        del savedOpt
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Loaded optimizer state from", os.path.join(args.retrain, 'best_Train_opt.saving'))
    else:
        optimizer = torch.optim.Adamax(paramsList, lr=lr)

    def lr_lambda(epoch):
        return min(1.,1.) * np.power(lrdecay, epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.8)

    if addMesh is not None:
        meshCv1 = torch.linspace(min1, max1, addMesh).to(device, dtype)
        meshCv2 = torch.linspace(min2, max2, addMesh).to(device, dtype)
        meshCv1, meshCv2 = torch.meshgrid(meshCv1, meshCv2, indexing='ij')
        meshCv1 = meshCv1.reshape(-1, 1)
        meshCv2 = meshCv2.reshape(-1, 1)
        meshCv12 = torch.cat([meshCv1, meshCv2], dim=-1)

    LOSS = []
    for e in range(epoch):
        tstart = time.time()
        for s in range(epochSteps):
            optimizer.zero_grad()
            beta_ = torch.rand(batch//2, 1).to(device, dtype) * (betahigh - betalow) + betalow
            betaPT_ = torch.rand(batch//2, 1).to(device, dtype) * (betaPThigh - betaPTlow) + betaPTlow
            beta_ = torch.cat([beta_, betaPT_], dim=0)

            cv1 = torch.rand(batch, 1).to(device, dtype) * (max1 - min1) + min1
            cv2 = torch.rand(batch, 1).to(device, dtype) * (max2 - min2) + min2
            cv12 = torch.cat([cv1, cv2], dim=-1)
            if addMesh is not None:
                cv12 = torch.cat([meshCv12, cv12], dim=0)
                meshBeta = beta.repeat(addMesh * addMesh, 1)
                beta_ = torch.cat([meshBeta, beta_], dim=0)

            if betaCV:
                beta_input = torch.cat([beta_, cv12], dim=-1)
            else:
                beta_input = beta_

            z = source.Uniform.sample(cv12.shape[0], nvars=nvars, T=1, low=priorParam['low'], high=priorParam['high'])
            zlogProb = source.Uniform.logProbability(z, T=1, low=priorParam['low']-1e-5, high=priorParam['high']+1e-5)

            z = torch.cat([cv12, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=beta_input, transformationList=transformationList, transformationParamList=transformationParamList)

            sample, ilogDet = ProteinConciseExpression.inverse(_sample, T=beta_)

            if torch.any(torch.isnan(sample)) or torch.any(torch.isnan(logDet)) or torch.any(torch.isinf(sample)) or torch.any(torch.isinf(logDet)):
                raise RuntimeError("Protein training produced invalid sample/Jacobian values")

            esContr = energyContrFn(sample)
            esBeta = esContr.sum(-1, keepdim=True) * beta_
            loss = zlogProb - logDet - ilogDet + esBeta
            loss = loss.mean()

            loss.backward()
            if clipGrad >= 0:
                nn.utils.clip_grad_norm_(paramsList, clipGrad)
            optimizer.step()

        trainTime = time.time() - tstart

        lossLst = []
        energyLst = []
        energyMeanLst = []
        entropyLst = []
        for idx in range(len(test1)):
            _cv1 = test1[idx].reshape(1, 1).to(device)
            _cv2 = test2[idx].reshape(1, 1).to(device)
            with torch.no_grad():
                _cv12 = torch.cat([_cv1, _cv2], dim=-1)
                _cv12 = _cv12.repeat(evalBatch, 1)
                beta_ = torch.Tensor([[beta]]).repeat(evalBatch, 1).to(_cv12)

                if betaCV:
                    beta_input = torch.cat([beta_, _cv12], dim=-1)
                else:
                    beta_input = beta_

                z = source.Uniform.sample(evalBatch, nvars=nvars, T=1, low=priorParam['low'], high=priorParam['high'])
                zlogProb = source.Uniform.logProbability(z, T=1, low=priorParam['low']-1e-5, high=priorParam['high']+1e-5)

                z = torch.cat([_cv12, z], dim=-1)

                _sample, logDet = source.TransformedDistribution.forward(z, T=beta_input, transformationList=transformationList, transformationParamList=transformationParamList)

                sample, ilogDet = ProteinConciseExpression.inverse(_sample, T=beta)
                es = energyFn(sample)
                entro = zlogProb - logDet - ilogDet
                loss = entro + es * beta

            lossLst.append(loss.mean().detach().item())
            energyLst.append(es.min().item())
            energyMeanLst.append(es.mean().item())
            entropyLst.append(entro.mean().item())

        lossLst = np.array(lossLst)
        lossSum = lossLst.mean()
        energyLst = np.array(energyLst)
        energyMeanLst = np.array(energyMeanLst)
        entropyLst = np.array(entropyLst)

        LOSS.append(lossSum)

        if noReg is not None and lossSum > bestTrainLoss * (1 - noReg):
            priorParam, muNet, sigmaNet, transformationParamList = torch.load(os.path.join(rootFolder, 'best_TrainLoss_joint.saving'), map_location=device, weights_only=False)
            paramsList = list(utils.yieldTrainable([muNet, sigmaNet, transformationParamList]))
            paramsList = list(filter(lambda p: p.requires_grad, paramsList))
            savedOpt = torch.load(os.path.join(rootFolder, 'best_Train_opt.saving'), map_location='cpu', weights_only=False)
            optimizer = torch.optim.Adamax(paramsList, lr=lr)
            optimizer.load_state_dict(savedOpt.state_dict())
            del savedOpt
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print(f"--> Rolled back to best model due to performance regression: {lossSum:.5f} > {bestTrainLoss * (1 - noReg):.5f}")
        elif lossSum < bestTrainLoss:
            bestTrainLoss = lossSum.item()
            torch.save([priorParam, muNet, sigmaNet, transformationParamList], os.path.join(rootFolder, 'best_TrainLoss_joint.saving'))
            torch.save(optimizer, os.path.join(rootFolder, 'best_Train_opt.saving'))
            print("--> Updated best model: ", bestTrainLoss)

        printString = "epoch: {:d}, L: {:.5f}, "
        printString += "time: {:.2f}, best: {:.5f}"
        resultLst = [e, lossSum]
        resultLst += [trainTime, bestTrainLoss]
        print(printString.format(*resultLst))

        for idx in range(len(test1)):
            printString = 'Protein >> @D1_{:.1f}'.format(test1[idx].item())
            resultLst = []
            printString += "@D2_{:.2f}".format(test2[idx].item()) + ":{:.2f}, e: {:.2f}, eb:{:.2f}, ep:{:.2f}"
            resultLst += [lossLst[idx].item(), energyLst[idx].item(), energyMeanLst[idx].item(), entropyLst[idx].item()]
            print(printString.format(*resultLst))

        scheduler.step()

        if e % saveStep == 0 or e == 0:
            torch.save([priorParam, muNet, sigmaNet, transformationParamList], os.path.join(rootFolder, 'savings', 'ALDPNF' + '_epoch_' + str(e) + ".saving"))
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
