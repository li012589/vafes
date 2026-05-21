import os

from scope import source, flow, utils

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse
import json

from forceUtils.energy import energy
from dipeptideEnergy import mass, charge, functs, idxs, params, concise2full
from dipeptideGeometry import alanineDipeptidePhiPsi


if __name__ == '__main__':
    device = torch.device('cpu')#torch.device('cuda:1')
    dtype = torch.float32
    batch = 512
    bins = 101
    cmap = 'viridis'

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-load", default=None, help="path to load free energy model")
    parser.add_argument("-loadV", default=None, help="path to load V matrix from TICA")
    parser.add_argument("-path", default=None, help="path to NEB path file (NEBpath.npy) for computing free energy along path")
    parser.add_argument("-noEval", action='store_true', help="skip 2D free energy evaluation, only process NEB path if -path is provided")
    parser.add_argument("-saveN", type=int, default=10, help="number of lowest energy configurations to save per NEB path point (default: 10)")
    args = parser.parse_args()

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
    test1 = torch.tensor([-0.112, -0.03, 0.1, -0.06, -0.05, 0.012, 0, 0])
    test2 = torch.tensor([-0.112, -0.05, 0.1, -0.05, -0.085, -0.05, 0.1, 0.05])
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

    lossLst = []
    for idx in range(len(test1)):
        _cv1 = test1[idx].repeat(batch).unsqueeze(-1).to(device)
        _cv2 = test2[idx].repeat(batch).unsqueeze(-1).to(device)
        with torch.no_grad():
            z = prior.sample(batch, nvars=nvars, T=T, **priorParam)
            zlogProb = prior.logProbability(z, T=T, **priorParam)

            z = torch.cat([_cv1, _cv2, z], dim=-1)

            _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

            sample = _sample @ invProjV + projMean

            loss = zlogProb - logDet + energyFn(sample) / T

        lossLst.append(loss.mean().detach().item())

    lossLst = np.array(lossLst)
    lossSum = lossLst.sum()

    printString = "epoch: {:d}, L: {:.5f}, "
    printString += "time: {:.2f}, best: {:.5f}"
    resultLst = [-1, lossSum]
    resultLst += [-1, -1]
    print(printString.format(*resultLst))
    for idx in range(len(test1)):
        printString = 'ALDP >> @A_{:.1f}'.format(test1[idx].item())
        resultLst = []
        printString += "@B_{:.1f}".format(test2[idx].item()) + ":{:.2f}  "
        resultLst += [lossLst[idx].item()]
        print(printString.format(*resultLst))

    if not args.noEval:
        print("Computing 2D free energy landscape...")
        cv1Range = np.linspace(-0.13, 0.13, bins)
        cv2Range = np.linspace(-0.13, 0.13, bins)
        cv1, cv2 = np.meshgrid(cv1Range, cv2Range)
        cv1, cv2 = torch.from_numpy(cv1).to(device, dtype).reshape(-1, 1), torch.from_numpy(cv2).to(device, dtype).reshape(-1, 1)
        cv12 = torch.cat([cv1, cv2], dim=-1)

        lnZLst = []
        errsLst = []
        logDetLst = []
        energyLst = []
        total_points = bins**2
        print(f"Total grid points to compute: {total_points}")
        for idx in range(total_points):
            _cv12 = cv12[idx].unsqueeze(0).repeat(batch, 1)
            with torch.no_grad():
                z = prior.sample(batch, nvars=nvars, T=T, **priorParam)
                zlogProb = prior.logProbability(z, T=T, **priorParam)

                z = torch.cat([_cv12, z], dim=-1)

                _sample, logDet = source.TransformedDistribution.forward(z, T=T, transformationList=transformationList, transformationParamList=transformationParamList)

                sample = _sample @ invProjV + projMean

                es = energyFn(sample)

                loss = zlogProb - logDet + es / T

            energyLst.append(es.mean().detach().item())
            logDetLst.append(logDet.mean().detach().item())
            lnZLst.append(loss.mean().detach().item())
            errsLst.append(loss.std().detach().item())
            
            if (idx + 1) % (total_points // 10) == 0 or idx == total_points - 1:
                progress = (idx + 1) / total_points * 100
                print(f"Progress: {progress:.1f}% ({idx + 1}/{total_points})")
        
        lnZLst = np.array(lnZLst).reshape(bins, bins)
        errsLst = np.array(errsLst).reshape(bins, bins)
        logDetLst = np.array(logDetLst).reshape(bins, bins)
        energyLst = np.array(energyLst).reshape(bins, bins)

        cv12 = cv12.detach().cpu().numpy()
        with open(os.path.join(args.load, "estCVfreeE.npy"), "wb") as f:
            np.save(f, cv12)
            np.save(f, lnZLst)
            np.save(f, errsLst)

        tickIdx = [0, bins//2, bins-1]
        plt.figure(figsize=(12, 9))
        plt.title('estimated CV Free energy of ALDP')
        plt.contourf(cv12[:, 0].reshape(lnZLst.shape), cv12[:, 1].reshape(lnZLst.shape), np.clip(lnZLst, min=0), levels=40, cmap=cmap)
        plt.colorbar()
        plt.xlabel('CV1')
        plt.ylabel('CV2')
        plt.savefig(os.path.join(args.load, 'cvFreeEest.pdf'))
        print("2D free energy landscape computation completed.")

    if args.path is not None:
        print(f"Processing NEB path from: {args.path}")

        neb_data = np.load(args.path, allow_pickle=True)
        final_path = neb_data[-1]
        n_images = final_path.shape[0]
        
        print(f"Number of images in path: {n_images}")
        print(f"Will save top {args.saveN} lowest energy configurations per point")
        
        configs_dir = os.path.join(args.load, "path_configs")
        os.makedirs(configs_dir, exist_ok=True)
        print(f"Configurations will be saved to: {configs_dir}")

        path_fe = []
        path_configs = []
        
        for i in range(n_images):
            cv1_val = final_path[i, 0]
            cv2_val = final_path[i, 1]
            
            _cv1 = torch.tensor([cv1_val]).repeat(batch).unsqueeze(-1).to(device)
            _cv2 = torch.tensor([cv2_val]).repeat(batch).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                z = prior.sample(batch, nvars=nvars, T=T, **priorParam)
                zlogProb = prior.logProbability(z, T=T, **priorParam)
                z = torch.cat([_cv1, _cv2, z], dim=-1)
                
                _sample, logDet = source.TransformedDistribution.forward(
                    z, T=T, transformationList=transformationList, 
                    transformationParamList=transformationParamList
                )
                
                sample = _sample @ invProjV + projMean
                es = energyFn(sample)
                loss = zlogProb - logDet + es / T
                
                fe_val = loss.mean().detach().cpu().item()
                path_fe.append(fe_val)

                es_np = es.detach().cpu().numpy().flatten()
                top_n_indices = np.argsort(es_np)[:args.saveN]

                top_configs = []
                atom_types = ['C', 'C', 'O', 'N', 'H', 'C', 'H', 'C', 'C', 'O', 'N', 'H', 'C']

                filename = f"{cv1_val:.4f}_{cv2_val:.4f}.xyz"
                filepath = os.path.join(configs_dir, filename)

                with open(filepath, 'w') as f:
                    for rank, idx in enumerate(top_n_indices):
                        best_config = sample[idx:idx+1]
                        full_xyz = concise2full(best_config).cpu().numpy() * 10
                        top_configs.append(full_xyz[0])

                        f.write(f"{len(atom_types)}\n")
                        f.write(f"CV1: {cv1_val:.6f}, CV2: {cv2_val:.6f}, Rank: {rank}, Energy: {es_np[idx]:.6f}\n")
                        for atom, coord in zip(atom_types, full_xyz[0]):
                            f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

                path_configs.append(top_configs[0])
            
            if (i + 1) % 10 == 0 or i == n_images - 1:
                print(f"Processed {i+1}/{n_images} images...")
        
        path_fe = np.array(path_fe)
        fe_path = os.path.join(args.load, "pathFEvalue.npy")
        np.save(fe_path, path_fe)
        print(f"Saved free energy along path to: {fe_path}")

        xyz_path = os.path.join(args.load, "configPath.xyz")
        atom_types = ['C', 'C', 'O', 'N', 'H', 'C', 'H', 'C', 'C', 'O', 'N', 'H', 'C']
        
        with open(xyz_path, 'w') as f:
            for i, (xyz, fe) in enumerate(zip(path_configs, path_fe)):
                f.write(f"{len(atom_types)}\n")
                f.write(f"Image {i+1}/{n_images}, Free Energy: {fe:.6f}, CV1: {final_path[i, 0]:.6f}, CV2: {final_path[i, 1]:.6f}\n")
                for atom, coord in zip(atom_types, xyz):
                    f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
        
        print(f"Saved path configurations to: {xyz_path}")
        print(f"XYZ file format: {len(atom_types)} atoms per frame, {n_images} frames")
        print(f"Individual top-{args.saveN} configurations saved to: {configs_dir}")
        print("File is compatible with UCSF ChimeraX and PyMOL")

        print("\nCalculating backbone dihedrals (phi, psi) for path configurations...")
        path_dihedrals = []
        for i, xyz in enumerate(path_configs):
            xyz_nm = xyz / 10.0
            xyz_tensor = torch.from_numpy(xyz_nm.reshape(1, -1))
            phi, psi = alanineDipeptidePhiPsi(xyz_tensor)
            path_dihedrals.append([phi.item(), psi.item()])
            if (i + 1) % 10 == 0 or i == n_images - 1:
                print(f"  Processed {i + 1}/{n_images} configurations...")

        path_dihedrals = np.array(path_dihedrals)
        dihedral_path = os.path.join(args.load, "pathDihedral.npy")
        np.save(dihedral_path, path_dihedrals)
        print(f"Saved backbone dihedrals to: {dihedral_path}")
        print(f"Phi range: [{path_dihedrals[:, 0].min():.3f}, {path_dihedrals[:, 0].max():.3f}] rad")
        print(f"Psi range: [{path_dihedrals[:, 1].min():.3f}, {path_dihedrals[:, 1].max():.3f}] rad")
