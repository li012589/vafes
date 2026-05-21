import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import argparse, json, h5py, glob

from h2n2Coordinate import _coord2Cv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CISSET = os.path.join(SCRIPT_DIR, 'etc', 'h2n2cis.npy')
DEFAULT_TRANSSET = os.path.join(SCRIPT_DIR, 'etc', 'h2n2trans.npy')


def resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path.removeprefix('./'))


class SigmoidCoupling(flow.CouplingBijector):
    r'''
    Sigmoid fn of F(x) = L / (1 + \exp(-k (x - t))) + b;
    k and t are learnable parameters, while L and b are inferred by boundary conditions.
    Forward: [-0.18, 0.18] -> [0, 1]
    '''
    @staticmethod
    def _parameterIter(inverse, maskList, kwargs, n):
        return kwargs

    @staticmethod
    def _preprocess(params):
        r'''
        params contains:
            k (ndarray of 1 element)
            t (ndarray of 1 element)
        '''
        params = params[0]
        k, t = params[:, 0:1], params[:, 1:]
        k = F.softplus(k) # k is positive
        t = torch.sigmoid(t) # t in [0, 1]
        _ek = torch.exp(k)
        _ekt = torch.exp(k * t)
        _ekmkt = torch.exp(k - k * t)
        L = ((1 + _ekt) * (1 + _ekmkt)) / (_ek - 1)
        b = (1 + _ekmkt) / (1 - _ek)
        return (k, t, L, b)

    @classmethod
    def _coupling(cls, inverse, x, params, **kwargs):
        k, t, L, b = cls._preprocess(params)
        if inverse:
            y = (x - b) / L
            y = torch.log(y / (1 - y))
            y = y / k + t
            ld = torch.log(k) + torch.log(L) + k * (y + t) - 2 * torch.log(torch.exp(k * t) + torch.exp(k * y))
            y = y * 0.36 - 0.18 # [0, 1] -> [-0.18, 0.18]
            ld = -ld + np.log(0.36)
        else:
            x = (x + 0.18) / (0.36) # [-0.18, 0.18] -> [0, 1]
            y = L / (1 + torch.exp(-k * (x - t))) + b
            ld = torch.log(k) + torch.log(L) + k * (x + t) - 2 * torch.log(torch.exp(k * t) + torch.exp(k * x))
            ld = ld - np.log(0.36)
        return y, ld


if __name__ == "__main__":
    rngseed = torch.seed()
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
    group.add_argument("-epoch", type=int, default=100, help="epoch/train steps")
    group.add_argument("-batch", type=int, default=256, help="batch size")
    group.add_argument("-saveStep", type=int, default=50, help="save model per steps")
    group.add_argument("-clipGrad", type=float, default=-1, help="maximum scale of gradient, negative for not activated")

    group = parser.add_argument_group("model parameters")
    group.add_argument("-K", type=int, default=10, help="bin number of nsf flow")
    group.add_argument("-layer", type=int, default=1, help="num of transformation layers")
    group.add_argument("-mlpVector", default=[10, 10], type=int, nargs="+", help="hidden dim of MLP used in spline flow")

    group = parser.add_argument_group("target parameters")
    group.add_argument("-cisset", default=DEFAULT_CISSET, help="path to load npy dataset of cis config")
    group.add_argument("-transset", default=DEFAULT_TRANSSET, help="path to load npy dataset of trans config")
    group.add_argument("-partition", type=float, default=0.8, help="the ratio of train to test size")
    group.add_argument("-seperate", type=int, default=50, help="sample every N-th sample from the original set")

    args = parser.parse_args()

    if args.folder is None:
        rootFolder = "./opt/h2n2CvTrain" + "_b" + str(args.K) + "_n" + str(args.layer) + "_" + str(args.mlpVector)
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
    cisset = resolve_repo_path(cisset)
    transset = resolve_repo_path(transset)

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

    # load datasets
    cisSet = torch.from_numpy(np.load(cisset)).to(device, dtype)[::seperate]
    transSet = torch.from_numpy(np.load(transset)).to(device, dtype)[::seperate]

    transSet = _coord2Cv(transSet)
    cisSet = _coord2Cv(cisSet)

    labelCis = torch.ones(cisSet.shape[0], 1).to(device, dtype)
    labelTrans = torch.zeros(transSet.shape[0], 1).to(device, dtype)

    permIdx = [torch.randperm(len(transSet) * 2)]
    data = torch.cat([transSet, cisSet], dim=0)[permIdx]
    labels = torch.cat([labelTrans, labelCis], dim=0)[permIdx]

    # train set and test set
    _trainSize = int(len(data) * partition)
    trainSet, testSet = torch.split(data, [_trainSize, len(data) - _trainSize])
    trainLabel, testLabel = torch.split(labels, [_trainSize, len(data) - _trainSize])

    epochSteps = cisSet.shape[0] // batch
    ranges = torch.tensor([[-0.15, 0.15], [1e-5, 0.18], [0, 0.20], [-0.15, 0.15], [-0.18, 0.18], [0, 0.15]])
    rangesCV = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0.15]])

    # init TpwLinearSpline
    maskList = []
    maskConpList = []
    netList = []
    # checkboard
    for n in range(layer):
        b = torch.zeros(1, 6).bool()
        bp = torch.ones(1, 6).bool()
        b[:, [4]] = 1
        bp[:, [4]] = 0
        netList.append([utils.layer.SimpleMLPreshape([6, *mlpVector, 2], (len(mlpVector)) * [nn.ELU()] + [None], reshapeBack=True, shape=[-1, 2])])
        maskList.append(b)
        maskConpList.append(bp)
    maskList = torch.cat(maskList, 0).to(torch.uint8)
    maskConpList = torch.cat(maskConpList, 0).to(torch.uint8)

    transformation = SigmoidCoupling
    transformationParams = SigmoidCoupling.initalize(maskList=maskList, maskConpList=maskConpList, networkList=netList)

    transformationList = [transformation]
    transformationParamList = [transformationParams]

    transformationParamList = utils.put(transformationParamList, device)

    bestTrainLoss = 99999999

    params = list(utils.yieldTrainable([transformationParamList]))
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
            #prepare data and label
            idx = torch.randint(high = len(trainSet), size = (batch,))
            dataBatch = trainSet[idx]
            labelBatch = trainLabel[idx]

            # use flow as discriminative model
            cvPred, logDet = source.TransformedDistribution.forward(dataBatch, T=1, transformationList=transformationList, transformationParamList=transformationParamList)

            cv = cvPred[:, -2:-1]

            #loss = F.binary_cross_entropy(cv, labelBatch)
            loss = ((cv - labelBatch)**2).mean()

            #print("s:", s, "/", epochSteps, "train loss:", loss.item())
            loss.backward()
            optimizer.step()

        trainTime = time.time() - tstart

        with torch.no_grad():
            # evaluation
            testPred, testlogDet = source.TransformedDistribution.forward(testSet, T=1, transformationList=transformationList, transformationParamList=transformationParamList)
            testcv = testPred[:, -2:-1]
            #testloss = F.binary_cross_entropy(testcv, testLabel)
            testloss = ((testcv - testLabel)**2).mean()

        LOSS.append(testloss.item())

        if testloss < bestTrainLoss:
            bestTrainLoss = testloss.item()
            torch.save([transformationParamList], os.path.join(rootFolder, 'best_TrainLoss_joint.saving'))
            print("--> Updated best model: ", bestTrainLoss)

        print("epoch:", e, "eval. loss:", testloss.item(), "time:", trainTime)
        print(testPred.max(0)[0])
        print(testPred.min(0)[0])

        # step the schedular
        scheduler.step()

        if e % saveStep == 0 or e == 0:
            # save joint and opt
            torch.save([transformationParamList], os.path.join(rootFolder, 'savings', 'h2n2CV' + '_epoch_' + str(e) + ".saving"))
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
            utils.cleanSaving(rootFolder, e, 6 * saveStep, 'h2n2CV')
