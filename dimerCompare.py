import argparse
import os

from dimerTrain import DimerBondLength, dimerVacuumSymWall
from dimerExact import F, Ur
from scope import flow, source

import torch
import numpy as np
from matplotlib import pyplot as plt


def add_subplot_axes(ax,rect,facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the dimer free-energy surface")
    parser.add_argument(
        "-load",
        default="./opt/dimer_fixed_b0_T0.3_1.6_B1_6",
        help="Path to the trained dimer checkpoint directory",
    )
    args = parser.parse_args()

    path = args.load
    batch = 512
    device = torch.device('cpu')
    T = 1.0
    base = 0

    energyFn = lambda sample: dimerVacuumSymWall(sample, base=base)


    Rrange = np.linspace(2, 5, 90)
    Trange = np.linspace(0.2, 2., 90)

    Fs = np.vectorize(lambda R: -np.log(F(R, T, base)[0]))(Rrange)
    errs = np.vectorize(lambda R: F(R, T, base)[1])(Rrange)

    with open(os.path.join(path, "exactCVfreeE.npy"), "wb") as f:
        np.save(f, Rrange)
        np.save(f, Fs)
        np.save(f, errs)

    T = 1.0
    nvars = [2]
    prior = source.Uniform
    transformationList = [flow.SplineFlow]
    priorParam, transformationParamList = torch.load(
        os.path.join(path, "best_TrainLoss_joint.saving"),
        map_location=device,
        weights_only=False,
    )
    transformationParamList[0]['linearBound'] = False
    lnZLst = []
    errsLst = []
    for _bondLen in Rrange:
        with torch.no_grad():
            _bondLen = torch.tensor([_bondLen], dtype=torch.float32).repeat(batch).unsqueeze(-1).to(device)
            z = prior.sample(batch, nvars=nvars, T=T, **priorParam)
            zlogProb = prior.logProbability(z, T=T, **priorParam)

            z = torch.cat([_bondLen, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

            sample, ilogDet = DimerBondLength.inverse(_sample, T=T)

            loss = zlogProb - logDet - ilogDet + energyFn(sample) / T
        lnZLst.append(loss.mean().detach().item())
        errsLst.append(loss.std().detach().item())

    lnZLst = np.array(lnZLst)
    errsLst = np.array(errsLst)

    with open(os.path.join(path, "estCVfreeE.npy"), "wb") as f:
        np.save(f, Rrange)
        np.save(f, lnZLst)
        np.save(f, errsLst)

    Es = np.vectorize(lambda R: Ur(R))(Rrange)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlabel(r'Bond length, $|\mathbf{x}|$', fontsize=27)
    ax.tick_params(labelsize=22)
    ax.set_ylabel('Free energy surface at $T=1.0$', fontsize=27)
    ax.errorbar(Rrange, lnZLst, yerr=errsLst, linewidth=3, elinewidth=2, alpha=0.5, color='xkcd:deep blue', linestyle='-', label='VaFES')
    ax.plot(Rrange, Fs, linewidth=3, color='xkcd:pinkish red', alpha=0.5, linestyle='--', label='Exact')
    plt.legend(loc='best', fontsize=22)

    subax = add_subplot_axes(ax, [0.25, 0.72, 0.45, 0.25])
    subax.plot(Rrange, Es, color='gray', linewidth=2)
    subax.tick_params(labelsize=17)
    subax.set_xlabel(r'Bond length, $|\mathbf{x}|$', fontsize=19)
    subax.set_ylabel('Energy', fontsize=19)

    plt.savefig(os.path.join(path, 'cvFreeEcompare.pdf'))

    Fs = []
    for _t in Trange:
        _Fs = []
        for _r in Rrange:
            _Fs.append(-np.log(F(_r, _t, base)[0]))
        Fs.append(_Fs)
    Fs = np.array(Fs)

    with open(os.path.join(path, "exactCVfreeEcont.npy"), "wb") as f:
        np.save(f, Rrange)
        np.save(f, Trange)
        np.save(f, Fs)

    lnZLst = []
    errsLst = []
    for _t in Trange:
        _t = torch.tensor([_t], dtype=torch.float32).repeat(batch).unsqueeze(-1).to(device)
        _lnZLst = []
        _errsLst = []
        for _bondLen in Rrange:
            with torch.no_grad():
                _bondLen = torch.tensor([_bondLen], dtype=torch.float32).repeat(batch).unsqueeze(-1).to(device)
                z = prior.sample(batch, nvars=nvars, T=_t, **priorParam)
                zlogProb = prior.logProbability(z, T=_t, **priorParam)

                z = torch.cat([_bondLen, z], dim=-1)

                _sample, logDet = source.TransformedDistribution.forward(z, T=_t, transformationList=transformationList, transformationParamList=transformationParamList)

                sample, ilogDet = DimerBondLength.inverse(_sample, T=_t)

                loss = zlogProb - logDet - ilogDet + energyFn(sample) / _t
            _lnZLst.append(loss.mean().detach().item())
            _errsLst.append(loss.std().detach().item())
        lnZLst.append(_lnZLst)
        errsLst.append(_errsLst)

    lnZLst = np.array(lnZLst)
    errsLst = np.array(errsLst)

    with open(os.path.join(path, "estCVfreeEcont.npy"), "wb") as f:
        np.save(f, Rrange)
        np.save(f, Trange)
        np.save(f, lnZLst)
        np.save(f, errsLst)

    _idx = [0, 49]
    cmap = 'viridis'

    coord1, coord2 = np.meshgrid(Rrange, Trange)

    plt.figure(figsize=(12, 9))
    plt.contourf(coord1, coord2, lnZLst, levels=50, cmap=cmap)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=22)
    plt.locator_params(axis='x', nbins=4)
    tMax = coord2.max()
    tMin = coord2.min()
    tSec = tMax - tMin
    plt.yticks([tMin, (tMax - tMin) / 2 + tMin , tMax])
    plt.xlabel(r'Bond length, $|\mathbf{x}|$', fontsize=27)
    plt.ylabel('Temperature, $T$', fontsize=27)
    plt.savefig(os.path.join(path, 'TcvFreeEest.pdf'))

    plt.figure(figsize=(12, 9))
    plt.contourf(coord1, coord2, Fs, levels=50, cmap=cmap)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(labelsize=22)
    plt.locator_params(axis='x', nbins=4)
    plt.yticks([tMin, (tMax - tMin) / 2 + tMin , tMax])
    plt.xlabel(r'Bond length, $|\mathbf{x}|$', fontsize=27)
    plt.ylabel('Temperature, $T$', fontsize=27)
    plt.savefig(os.path.join(path, 'TcvFreeEexact.pdf'))
    plt.close("all")
