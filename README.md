# Variational Free Energy Surface (VaFES)

PyTorch implementation for the paper _Differentiable free energy surface: a variational approach to directly observing rare events using generative deep-learning models_ ([arXiv:2604.09769](https://arxiv.org/abs/2604.09769)). The repository contains necessary scripts to reproduce the paper experiments.

## Overview

Included applications:
- dimer
- H2N2
- alanine dipeptide
- chignolin

## Requirements

The code runs on standard CPUs and benefits from GPUs when available. 
(scripts are tested on Apple Silicon, Intel CPUs, and NVIDIA GPUs)

The package has been tested on:
- macOS (Tahoe 26.5) 
- Linux (Ubuntu 22.04.5 LTS)

Python dependencies are listed in `requirements.txt`:

```text
numpy
torch
matplotlib
h5py
scipy
openmm
pdbfixer
MDAnalysis
```

## Setup

Use `python3`.

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```
- `pdbfixer` is sometimes easier to install from `conda-forge` than from `pip`.
- Installation time depends on network conditions.

## Application Experiments

### Dimer

Train:

```bash
python3 dimerTrain.py
```
- Expected run time: about 15 minutes (on Apple M4 Pro, 24GB)

Evaluate and compare against the analytic result:

```bash
python3 dimerCompare.py -load <dimer-checkpoint-dir>
```

### H2N2

Train the CV model:

```bash
python3 h2n2CvTrain.py
```
- Expected run time: about 0.5 minutes (on Apple M4 Pro, 24GB)

Train the VaFES model:

```bash
python3 h2n2FESTrain.py -loadCV <h2n2-cv-dir>
```
- Expected run time: about 18 minutes (on Apple M4 Pro, 24GB)

Evaluate the free energy surface:

```bash
python3 h2n2Compare.py -load <h2n2-dir> -loadCV <h2n2-cv-dir>
```

Compute NEB paths and make the plots:

```bash
python3 h2n2Path.py -load <h2n2-dir> -loadCV <h2n2-cv-dir>
python3 h2n2Plot.py -load <h2n2-dir>
```

### Alanine Dipeptide

Train:

```bash
python3 dipeptideTrain.py -loadV <path-to-TICA-projection-npy>
```
- `-loadV` defaultly uses `etc/dipeptideMeta.npz` for demonstration.
- Expected run time: about 5 hours (on one Nvidia RTX4090, 24GB)

Evaluate the free energy surface:

```bash
python3 dipeptideCompare.py -load <dipeptide-dir>
```

Compute the NEB path, then evaluate free energy along that path and plot it:

```bash
python3 dipeptidePath.py -load <dipeptide-dir>
python3 dipeptideCompare.py -load <dipeptide-dir> -path <dipeptide-dir>/NEBpath.npy
python3 dipeptidePlot.py -load <dipeptide-dir>
```

### Chignolin

Train:

```bash
python3 chignolinTrain.py
```
- Expected run time: about 76 hours (on one Nvidia RTX4090, 24GB)

Evaluate the free energy surface and sample configurations:

```bash
python3 chignolinCompare.py -load <protein-dir>
```

Make the plot:

```bash
python3 chignolinPlot.py -load <protein-dir>
```

Optional structure extraction examples:

```bash
python3 chignolinPlot.py -load <protein-dir> -region "Native:3.4:4.0:4.0:4.6" -outputPDB
```

Compute CA-RMSD against an experimental/reference PDB after exporting `Native.pdb`:

```bash
python3 carmsd.py -load1 /path/to/1uao.pdb -load2 <protein-dir>/pic/Native.pdb -core 2 9
```
- `carmsd.py` supports multi-model PDBs and prints the full pairwise CA-RMSD matrix.
- `-core 2 9` core-residue comparison excluding the two terminal residues.

## Outputs

Scripts write outputs either into the directory passed with `-load` or into an automatically created training directory.

Typical outputs:
- dimer: `exactCVfreeE.npy`, `estCVfreeE.npy`, `cvFreeEcompare.pdf`
- H2N2: `estCVfreeE.npy`, `NEBpath.npy`, `NEBpathNaive.npy`, `pic/*.pdf`
- alanine dipeptide: `estCVfreeE.npy`, `NEBpath.npy`, `pathFEvalue.npy`, `pathDihedral.npy`, `configPath.xyz`, `path_configs/`
- chignolin: `estCVfreeE.npy`, `samplesCV.npy`, `sampleEnergies.npy`, `pic/*.pdf`

## Bundled Assets and Datasets

- `etc/h2n2cis.npy`: reference H2N2 cis conformations used to train the H2N2 CV model.
- `etc/h2n2trans.npy`: reference H2N2 trans conformations used together with the cis set for H2N2 CV training.
- `etc/dipeptideMeta.npz`: alanine dipeptide demonstration rotation matrix and coordinate ranges.
- `etc/chignolinMeta.npz`: helper arrays for the chignolin local-coordinate parameterization, including ranges and hydrogen-placement metadata.
- `etc/geoOpt.pdb`: reference chignolin structure used by the OpenMM frontend.

## Citation

```bibtex
@misc{li2026differentiablefreeenergysurface,
      title={Differentiable free energy surface: a variational approach to directly observing rare events using generative deep-learning models},
      author={Shuo-Hui Li and Chen Chen and Yao-Wen Zhang and Ding Pan},
      year={2026},
      eprint={2604.09769},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2604.09769},
}
```
