import numpy as np

import matplotlib.pyplot as plt
import argparse, os

plt.rcParams.update({
    "font.size": 18,
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150
})

cmap = 'viridis'
path_color1 = '#008060'
path_color2 = '#FF6B35'

parser = argparse.ArgumentParser(description="")
parser.add_argument("-load", default=None, help="path to load free energy model")
args = parser.parse_args()

pic_dir = os.path.join(args.load, 'pic')
os.makedirs(pic_dir, exist_ok=True)

with open(os.path.join(args.load, 'estCVfreeE.npy'), 'rb') as f:
    cv = np.load(f)
    lnZLst = np.load(f)

with open(os.path.join(args.load, 'NEBpath.npy'), 'rb') as f:
    pathList = np.load(f)
    VList = np.load(f)
    pathConfig = np.load(f)

with open(os.path.join(args.load, 'NEBpathNaive.npy'), 'rb') as f:
    pathListNaive = np.load(f)
    VListNaive = np.load(f)
    pathConfigNaive = np.load(f)

path = pathList[-1]
V = VList[-1]
pathConfig = pathConfig[-1]

fig, ax = plt.subplots(figsize=(8, 7))
plt.title(r'Free energy surface of diazene', fontsize=22)
plt.contourf(cv[:, 0].reshape(lnZLst.shape), cv[:, 1].reshape(lnZLst.shape), np.clip(lnZLst, max=700), levels=50, cmap=cmap)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)

plt.tick_params(labelsize=20)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.xticks([0.0, 0.5, 1.0])
plt.yticks(np.arange(0.02, 0.13, 0.02))

plt.plot(path[:, 0], path[:, 1],  alpha=0.5, color='white', linewidth=4.5, zorder=2)
plt.plot(path[:, 0], path[:, 1],  alpha=1, color=path_color1, linewidth=3, zorder=2, label=r'Torsion')

_pathListNaive = pathListNaive[:, 1] + 0.0005 # avoid overlap with x-axis
plt.plot(pathListNaive[:, 0], _pathListNaive, alpha=0.5, color='white', linewidth=4.5, zorder=2)
plt.plot(pathListNaive[:, 0], _pathListNaive, alpha=1, color=path_color2, linewidth=3, zorder=2, label=r'Inversion')
plt.legend(loc='best', fontsize=18)

plt.xlabel(r'Machine-learning CV', fontsize=22)
plt.ylabel(r'Z coordinate (nm)', fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(pic_dir, 'cvPath.pdf'), dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 2.5))
plt.tick_params(labelsize=20)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.plot(path[:, 0], V, linewidth=3, alpha=1, color=path_color1, linestyle='-')
plt.xlabel(r'Machine-learning CV', fontsize=22)
plt.ylabel(r'Free energy ($k_B T$)', fontsize=22)
ax.yaxis.set_label_coords(-0.12, 0.35)
plt.savefig(os.path.join(pic_dir, 'fePath.pdf'), dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(8, 2.5))
plt.tick_params(labelsize=20)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)
plt.plot(path[:, 0], VListNaive, linewidth=3, alpha=1, color=path_color2, linestyle='-')
plt.xlabel(r'Machine-learning CV', fontsize=22)
plt.ylabel(r'Free energy ($k_B T$)', fontsize=22)
ax.yaxis.set_label_coords(-0.12, 0.35)
plt.savefig(os.path.join(pic_dir, 'fePathNaive.pdf'), dpi=300, bbox_inches='tight')
