import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patheffects import withStroke
import argparse, os
import MDAnalysis as mda
import warnings
from chignolinGeometry import compute_chignolin_geometry

warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size": 18,
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150
})

cmap = 'viridis'
PRESET_COLORS = ['#FF6B35', '#00BFFF', '#32CD32', '#E60000', '#BF7FFF', '#00FFCC',
                 '#FFD700', '#FF1493', '#1E90FF', '#ADFF2F']

def parse_region(region_str):
    """Parse region string in format: label:xmin:xmax:ymin:ymax:color
    Color is optional, will use preset if not provided."""
    parts = region_str.split(':')
    if len(parts) < 5:
        raise ValueError(f"Region must have at least 5 parts (label:xmin:xmax:ymin:ymax), got: {region_str}")
    
    label = parts[0]
    xmin, xmax, ymin, ymax = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    color = parts[5] if len(parts) > 5 else None
    
    return {'label': label, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'color': color}

parser = argparse.ArgumentParser(description="Plot free energy surface with optional region markers")
parser.add_argument("-load", default=None, help="path to load free energy model")
parser.add_argument("-region", action='append', dest='regions',
                   help="Region to mark on plot. Format: 'label:xmin:xmax:ymin:ymax[:color]'. "
                        "Can be used multiple times (max 10). Example: 'Native:3.4:4.0:4.0:4.5:red'")
parser.add_argument("-outputPDB", action='store_true', 
                   help="Output lowest energy PDB for each region")
parser.add_argument("-refPDB", default=os.path.join(SCRIPT_DIR, 'etc', 'geoOpt.pdb'),
                   help="Reference PDB file for output structure (default: codeRepo/etc/geoOpt.pdb)")
args = parser.parse_args()

if args.load is None:
    print("Error: Please provide -load argument with the path to the saving directory")
    exit(1)

pic_dir = os.path.join(args.load, 'pic')
os.makedirs(pic_dir, exist_ok=True)

regions = []
if args.regions:
    if len(args.regions) > 10:
        print(f"Warning: More than 10 regions provided, only using first 10.")
        args.regions = args.regions[:10]
    
    for i, region_str in enumerate(args.regions):
        try:
            region = parse_region(region_str)
            if region['color'] is None:
                region['color'] = PRESET_COLORS[i % len(PRESET_COLORS)]
            regions.append(region)
        except ValueError as e:
            print(f"Error parsing region {i+1}: {e}")
            continue

try:
    with open(os.path.join(args.load, 'estCVfreeE.npy'), 'rb') as f:
        cv_grid = np.load(f)
        lnZLst = np.load(f)

    grid_size = int(np.sqrt(cv_grid.shape[0]))
    cv_x = cv_grid[:, 0].reshape(grid_size, grid_size)
    cv_y = cv_grid[:, 1].reshape(grid_size, grid_size)
    lnZLst = lnZLst.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(r'Free energy surface of Chignolin', fontsize=22)

    cont = ax.contourf(
        cv_x,
        cv_y,
        lnZLst,
        levels=22,
        cmap=cmap)

    cbar = plt.colorbar(cont, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.locator_params(nbins=7)

    ax.tick_params(labelsize=20)

    if regions:
        for region in regions:
            width = region['xmax'] - region['xmin']
            height = region['ymax'] - region['ymin']

            rect = patches.FancyBboxPatch((region['xmin'], region['ymin']), width, height,
                                           boxstyle="round,pad=0,rounding_size=0.08",
                                           linewidth=1.2, edgecolor=region['color'],
                                           facecolor='none', linestyle='--', alpha=1.0, zorder=2)
            rect.set_path_effects([withStroke(linewidth=2, foreground='white', alpha=0.6)])
            ax.add_patch(rect)

            label_x = region['xmin'] - 0.12
            label_y = region['ymax']

            txt = ax.text(label_x, label_y, region['label'], 
                         fontsize=22, fontweight='bold', color=region['color'],
                         ha='right', va='top', zorder=4)
            txt.set_path_effects([withStroke(linewidth=1.2, foreground='white', alpha=0.7)])

    ax.set_xlabel(r'$C_{\alpha}^{1}{-}C_{\alpha}^{10}$ ($\AA$)', fontsize=22)
    ax.set_ylabel(r'$C_{\alpha}^{3}{-}C_{\alpha}^{8}$ ($\AA$)', fontsize=22)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, 'cvFreeEnergy.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cvFreeEnergy.pdf to {pic_dir}")
    
    if regions:
        print(f"Marked {len(regions)} region(s) on the plot")
    
except FileNotFoundError as e:
    print(f"Could not generate cvFreeEnergy.pdf: {e}")
except Exception as e:
    print(f"Error generating cvFreeEnergy.pdf: {e}")

if args.outputPDB and regions:
    try:
        with open(os.path.join(args.load, 'samplesCV.npy'), 'rb') as f:
            cv_samples = np.load(f)
            samples = np.load(f)

        num_cv_points, num_configs_per_cv, num_atoms, _ = samples.shape
        print(f"Loaded {num_cv_points} CV points, each with {num_configs_per_cv} configurations")

        if not os.path.exists(args.refPDB):
            print(f"Warning: Reference PDB file not found: {args.refPDB}")
            print("PDB output skipped. Please provide correct -refPDB path.")
        else:
            u_ref = mda.Universe(args.refPDB)

            energy_file = os.path.join(args.load, 'sampleEnergies.npy')
            if os.path.exists(energy_file):
                with open(energy_file, 'rb') as f:
                    energies = np.load(f)
                energies = energies.reshape(num_cv_points, num_configs_per_cv)
                has_energies = True
                print(f"Loaded energies from {energy_file}")
            else:
                has_energies = False
                print(f"Warning: Energy file not found at {energy_file}")
                print("Will select first configuration in each region instead of minimum energy")

            for region in regions:
                mask_x = (cv_samples[:, 0] >= region['xmin']) & (cv_samples[:, 0] <= region['xmax'])
                mask_y = (cv_samples[:, 1] >= region['ymin']) & (cv_samples[:, 1] <= region['ymax'])
                cv_mask = mask_x & mask_y
                
                if not cv_mask.any():
                    print(f"Warning: No CV points found in region '{region['label']}'")
                    continue

                cv_indices = np.where(cv_mask)[0]
                print(f"Region '{region['label']}': Found {len(cv_indices)} CV points in range")

                min_energy = float('inf')
                min_config = None

                for cv_idx in cv_indices:
                    if has_energies:
                        config_energies = energies[cv_idx]
                        min_config_idx = np.argmin(config_energies)
                        config_energy = config_energies[min_config_idx]

                        if config_energy < min_energy:
                            min_energy = config_energy
                            min_config = samples[cv_idx, min_config_idx]
                    else:
                        if min_config is None:
                            min_config = samples[cv_idx, 0]

                if has_energies:
                    print(f"  Minimum energy found: {min_energy:.3f}")

                config_pos = min_config
                cv_x_calc = np.linalg.norm(config_pos[1] - config_pos[131])
                cv_y_calc = np.linalg.norm(config_pos[31] - config_pos[93])

                print(f"  Selected config CV: X={cv_x_calc:.3f}, Y={cv_y_calc:.3f}")
                print(f"  Expected range: X=[{region['xmin']}, {region['xmax']}], Y=[{region['ymin']}, {region['ymax']}]")
                print(f"  X in range: {region['xmin'] <= cv_x_calc <= region['xmax']}")
                print(f"  Y in range: {region['ymin'] <= cv_y_calc <= region['ymax']}")

                geom = compute_chignolin_geometry(min_config)
                print(f"  d(Asp3 N - Gly7 O)  = {geom['d_Asp3N_Gly7O']:.3f} A")
                print(f"  d(Asp3 N - Thr8 O)  = {geom['d_Asp3N_Thr8O']:.3f} A")
                print(f"  Gly7 phi            = {geom['phi_Gly7']:.1f} deg")
                print(f"  Gly7 psi            = {geom['psi_Gly7']:.1f} deg")

                output_filename = f"{region['label']}.pdb"
                output_path = os.path.join(pic_dir, output_filename)

                u_ref.atoms.positions = min_config
                with mda.Writer(output_path) as w:
                    w.write(u_ref.atoms)
                
                print(f"  Saved {output_filename}\n")
                
    except FileNotFoundError as e:
        print(f"Could not output PDB files: {e}")
        print("Make sure samplesCV.npy exists in the load directory")
    except Exception as e:
        print(f"Error generating PDB files: {e}")
        import traceback
        traceback.print_exc()

print(f"All plots saved to {pic_dir}")
