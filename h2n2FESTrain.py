import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import argparse, json, h5py, glob

from h2n2Coordinate import energyCV, _coord2Cv
from h2n2CvTrain import SigmoidCoupling

from forceUtils.energy import energy
from forceUtils.twobody import fourthPowerBond, coulombPair
from forceUtils.threebody import harmonicAngle, harmonicCosine
from forceUtils.fourbody import periodicProperDihedral


if __name__ == '__main__':
    rngseed = torch.seed()
    torch.manual_seed(rngseed)
    print("Using torch seed:", rngseed)
    #torch.autograd.set_detect_anomaly(True)
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
    group.add_argument("-mlpVector", default=[20, 50, 100, 150], type=int, nargs="+", help="hidden dim of MLP used in spline flow")

    group = parser.add_argument_group("target parameters")
    parser.add_argument("-loadCV", default=None, help="path to load trained cv model")
    group.add_argument("-TList", default=[0.02, 0.05, 0.07, 0.1], type=float, nargs="+", help="temperature list used in evaluation")
    group.add_argument("-Tlow", type=float, default=0.02, help="low temperature start point in training")
    group.add_argument("-Thigh", type=float, default=1.5, help="high temperature end point in training")
    group.add_argument("-TPTlow", type=float, default=0.1, help="emphasis low temperature start point in training")
    group.add_argument("-TPThigh", type=float, default=1.5, help="emphasis high temperature end point in training")

    args = parser.parse_args()

    if args.folder is None:
        rootFolder = "./opt/h2n2fesTrain" + "_b" + str(args.K) + "_n" + str(args.layer) + "_" + str(args.mlpVector)
        print("No specified saving path, using", rootFolder)
    else:
        rootFolder = args.folder
        print("Using", rootFolder)
    utils.createWorkSpace(rootFolder)

    open(os.path.join(rootFolder, "output.log"), "w").close()

    def print_to_both(*args, **kwargs):
        # Print to file
        with open(os.path.join(rootFolder, 'output.log'), 'a') as file:
            original_print(*args, file=file)
        # Print on screen
        original_print(*args)

    # Save a reference to the original built-in print function
    original_print = __builtins__.print

    # Monkey-patch the built-in print function with our custom implementation
    __builtins__.print = print_to_both

    # save rng seed
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

    h2n2mass = torch.tensor([[[1.0080], [14.0067], [14.0067], [1.0080]]]).to(device, dtype)
    h2n2charge = torch.tensor([[[0.350], [-0.350], [-0.350], [0.350]]]).to(device, dtype)
    h2n2functs = [fourthPowerBond, coulombPair, harmonicCosine, periodicProperDihedral]
    h2n2idxs = [torch.tensor([[0, 1], [1, 2], [2, 3]]), torch.tensor([[0, 3]]), torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[0, 1, 2, 3]])]
    h2n2params= [torch.tensor([[2.2652e7, 0.1040], [2.0480e7, 0.1250], [2.2652e7, 0.1040]]), torch.tensor([[138.935458]]), torch.tensor([[503.00, torch.deg2rad(torch.tensor(106.75))], [503.00, torch.deg2rad(torch.tensor(106.75))]]), torch.tensor([[41.80, 2, torch.deg2rad(torch.tensor(180.0))]])]
    h2n2params = [term.to(device, dtype) for term in h2n2params]
    pos = torch.tensor([[[-.1121, 0.0763, 0.000], [-0.0607, -0.0140, -0.000], [0.0607, 0.0140, -0.000], [.1121, -0.0763, 0.000]]]).to(device, dtype)
    cv = _coord2Cv(pos)

    energyFn = lambda config: energyCV(config, h2n2mass, h2n2charge, h2n2functs, h2n2idxs, h2n2params)

    cvTransformationParamList = torch.load(
        os.path.join(args.loadCV, "best_TrainLoss_joint.saving"),
        map_location=device,
        weights_only=False,
    )[0]

    #ranges = torch.tensor([[-0.15, 0.15], [1e-5, 0.18], [0, 0.20], [-0.15, 0.15], [-0.18, 0.18], [0, 0.15]])
    ranges = torch.tensor([[-0.15, 0.15], [1e-5, 0.18], [0, 0.20], [-0.15, 0.15], [0, 1], [0, 0.15]])

    T = torch.tensor(1.0) # base temperature
    testY = torch.tensor([0.0995, 0, -0.0995])
    testZ = torch.tensor([0.0, 0.11])
    maxY = ranges[-2, 1]
    minY = ranges[-2, 0]
    maxZ = ranges[-1, 1]
    minZ = ranges[-1, 0]

    # define the prior
    nvars = [4]
    prior = source.Uniform
    priorParam = {'low': ranges[:4, 0], 'high': ranges[:4, 1]}
    priorParam = source.Uniform.initalize(priorParam)

    # init TpwLinearSpline
    maskList = []
    maskConpList = []
    netList = []
    boundaryList = []
    # checkboard
    for n in range(layer):
        if n % 2 == 0:
            b = torch.zeros(1, 6).bool()
            bp = torch.ones(1, 6).bool()
            b[:, [0, 2]] = 1
            bp[:, [0, 2]] = 0
            boundaryList.append((torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b), torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b)))
        else:
            b = torch.zeros(1, 6).bool()
            bp = torch.ones(1, 6).bool()
            b[:, [1, 3]] = 1
            bp[:, [1, 3]] = 0
            boundaryList.append((torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b), torch.masked_select(ranges[:, 0], b), torch.masked_select(ranges[:, 1], b)))
        netList.append([utils.layer.SimpleMLPreshape([5, *mlpVector, (2 * K + 2) * 2], (len(mlpVector)) * [nn.ELU()] + [None], reshapeBack=True, shape=[-1, 2 * K + 2, 2])])
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

    with torch.no_grad():
        cv = cv.repeat(3, 1)
        cv[:, -2] = testY
        cvf, _ = source.TransformedDistribution.forward(cv, T=1, transformationList=[SigmoidCoupling], transformationParamList=cvTransformationParamList)
        testY = cvf[:, -2].detach()

    bestTrainLoss = 99999999

    params = list(utils.yieldTrainable([priorParam, transformationParamList]))
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print('total nubmer of trainable parameters:', nparams)

    # init optimizer
    optimizer = torch.optim.Adamax(params, lr=lr)

    def lr_lambda(epoch):
        return min(1.,1.) * np.power(lrdecay, epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.8)

    # start optimize
    LOSS = []

    for e in range(epoch):
        tstart = time.time()
        # training
        for s in range(epochSteps):
            optimizer.zero_grad()
            Y = torch.rand(batch, 1) * (maxY - minY) + minY
            Y = Y.to(device)
            Z = torch.rand(batch, 1) * (maxZ - minZ) + minZ
            Z = Z.to(device)

            z = prior.sample(batch, nvars=nvars, T=1, **priorParam)
            zlogProb = prior.logProbability(z, T=1, **priorParam)

            z = torch.cat([z, Y, Z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=1, transformationList=transformationList, transformationParamList=transformationParamList)

            sample, ilogDet = source.TransformedDistribution.inverse(_sample, T=1, transformationList=[SigmoidCoupling], transformationParamList=cvTransformationParamList)

            if (
                torch.any(torch.isnan(sample))
                or torch.any(torch.isnan(logDet))
                or torch.any(torch.isinf(sample))
                or torch.any(torch.isinf(logDet))
                or torch.any(torch.isnan(ilogDet))
                or torch.any(torch.isinf(ilogDet))
            ):
                raise RuntimeError("H2N2 training produced invalid sample/Jacobian values")

            loss = zlogProb - logDet - ilogDet + energyFn(sample) / 1
            loss = loss.mean()

            loss.backward()
            optimizer.step()

        trainTime = time.time() - tstart

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

        LOSS.append(lossSum)
        if lossSum < bestTrainLoss:
            bestTrainLoss = lossSum.item()
            torch.save([priorParam, transformationParamList], os.path.join(rootFolder, 'best_TrainLoss_joint.saving'))
            print("--> Updated best model: ", bestTrainLoss)

        # feeddback
        printString = "epoch: {:d}, L: {:.5f}, "
        printString += "time: {:.2f}, best: {:.5f}"
        resultLst = [e, lossSum]
        resultLst += [trainTime, bestTrainLoss]
        print(printString.format(*resultLst))
        for idxT, _Y in enumerate(testY):
            printString = ' >> Y_{:.1f}'.format(_Y.item()) + "_h2n2"
            resultLst = []
            for idx, _Z in enumerate(testZ):
                printString += "@B" + str(_Z.item()) + ":{:.2f}  "
                resultLst += [lossLst[idx + len(testZ) * idxT].item()]
            print(printString.format(*resultLst))

        # step the schedular
        scheduler.step()

        if e % saveStep == 0 or e == 0:
            # save joint and opt
            torch.save([priorParam, transformationParamList], os.path.join(rootFolder, 'savings', 'h2n2NF' + '_epoch_' + str(e) + ".saving"))
            #torch.save(optimizer, rootFolder + 'savings/' + name + "_epoch_" + str(e) + "_opt.saving")
            # save loss values
            with h5py.File(os.path.join(rootFolder, "records", "LOSS"+'.hdf5'), 'w') as f:
                f.create_dataset("LOSS", data=np.array(LOSS))
            # plot loss curve
            lossfig = plt.figure(figsize=(8, 5))
            lossax = lossfig.add_subplot(111)
            epoch = len(LOSS)
            lossax.plot(np.arange(epoch), np.array(LOSS), 'go-', label="loss", markersize=2.5)
            lossax.set_xlim(0, epoch)
            lossax.legend()
            lossax.set_title("Loss Curve")
            plt.savefig(os.path.join(rootFolder, 'pic', 'lossCurve.png'), bbox_inches="tight", pad_inches=0)
            plt.close()
            # clean extra saving
            utils.cleanSaving(rootFolder, e, 6 * saveStep, 'h2n2NF')
