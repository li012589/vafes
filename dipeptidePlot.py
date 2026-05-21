import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import argparse, json, h5py, glob, os

plt.rcParams.update({
    "font.size": 18,
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150
})

cmap = 'viridis'
path_color1 = '#008060'  # teal green
path_color2 = '#FF6B35'  # terracotta red

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

with open(os.path.join(args.load, 'pathFEvalue.npy'), 'rb') as f:
    pathFE = np.load(f)

# Load dihedral angles if available
dihedral_file = os.path.join(args.load, 'pathDihedral.npy')
if os.path.exists(dihedral_file):
    with open(dihedral_file, 'rb') as f:
        pathDihedral = np.load(f)
    has_dihedrals = True
    print(f"Loaded dihedral angles: phi and psi for {len(pathDihedral)} configurations")
else:
    has_dihedrals = False
    print("Warning: pathDihedral.npy not found, skipping dihedral plot")

path = pathList[-1]

fig, ax = plt.subplots(figsize=(8, 5))
plt.title(r'Free energy surface of alanine dipeptide', fontsize=22)
#plt.pcolormesh(cv[:, 0].reshape(lnZLst.shape), cv[:, 1].reshape(lnZLst.shape), lnZLst, cmap=cmap)
plt.contourf(cv[:, 0].reshape(lnZLst.shape), cv[:, 1].reshape(lnZLst.shape), lnZLst, levels=50, cmap=cmap)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.ax.locator_params(nbins=5)
plt.tick_params(labelsize=20)
plt.locator_params(axis='x', nbins=4)
plt.locator_params(axis='y', nbins=4)

plt.plot(path[:, 0], path[:, 1], alpha=0.5, color='white', linewidth=4.5, zorder=2)
plt.plot(path[:, 0], path[:, 1], alpha=1, color=path_color2, linewidth=3, zorder=2)
#plt.legend(loc='best', fontsize=18)

plt.xlabel(r'TICA CV1', fontsize=22)
plt.ylabel(r'TICA CV2', fontsize=22)

plt.tight_layout()
plt.savefig(os.path.join(pic_dir, 'cvPath.pdf'), dpi=300, bbox_inches='tight')

fig, ax1 = plt.subplots(figsize=(8, 4))
#plt.tick_params(labelsize=20)
plt.locator_params(axis='y', nbins=4)

# Plot free energy on primary y-axis
ax1.plot(pathFE, linewidth=3, alpha=1, color=path_color2, linestyle='-', label='Free Energy')
ax1.set_xlabel(r'Reaction Coordinate', fontsize=22)
ax1.set_ylabel(r'Free Energy ($k_B T$)', fontsize=22, color=path_color2)
ax1.tick_params(axis='y', labelcolor=path_color2, labelsize=18)
ax1.tick_params(axis='x', bottom=False, labelbottom=False)  # Remove x-axis ticks but keep label
ax1.grid(True, alpha=0.3)

# Add twin axis for dihedral angles if available
if has_dihedrals:
    ax2 = ax1.twinx()
    # Use dihedrals in radians directly
    phi_rad = pathDihedral[:, 0]
    psi_rad = pathDihedral[:, 1]
    
    ax2.plot(phi_rad, linewidth=2, alpha=0.7, color='xkcd:deep blue', linestyle='--', label=r'$\phi$')
    ax2.plot(psi_rad, linewidth=2, alpha=0.7, color='xkcd:magenta', linestyle='-.', label=r'$\psi$')
    ax2.set_ylabel(r'Backbone Dihedrals (rad)', fontsize=22)
    ax2.tick_params(axis='y', labelsize=18)
    # Set y-ticks to show -π, 0, π
    ax2.set_yticks([-np.pi, 0, np.pi])
    ax2.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    
    # Combine legends from both axes and place above the plot to avoid blocking curves
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', 
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=16, frameon=False)
else:
    ax1.legend(loc='upper right', fontsize=16, framealpha=0.9)

plt.savefig(os.path.join(pic_dir, 'fePath.pdf'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saved to {pic_dir}")
print("  - cvPath.pdf: Free energy surface with NEB path")
print("  - fePath.pdf: Free energy profile along path with dihedral angles")
