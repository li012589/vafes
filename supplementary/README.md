# Supplementary analysis scripts

This directory contains the supplementary scripts and the parameter files they read. Script paths in each application entry are relative to that application's `scripts/` directory; repo-root scripts are identified explicitly.

## Outputs

- dimer: the repo-root `dimerTrain.py` trains each seed, and `dimerValidation.py` reproduces SI Figs. S2–S4 and the accompanying ESS values.
- diazene: the repo-root `h2n2FESTrain.py` trains each seed; `h2n2ExactIntegral.py` and `h2n2PathValidation.py` reproduce the order-65 barrier comparison in SI Table S3 and the multi-seed torsion and inversion profiles in SI Figs. S5 and S6.
- alanine: `metad/run_dipeptide_metad_cv.py`, `metad/sum_hills.sh`, and `analysis/analyze_dipeptide_fes.py` reproduce the metadynamics and leave-one-out validation reported with SI Table S5. The repo-root `dipeptideTrain.py` trains each seed; `multiseed/dipeptide_multiseed_eval.py` and `analysis/analyze_dipeptide_fes.py` reproduce SI Table S6. `tica/md/gen_md.py` and `tica/tica.py` reproduce the TICA fit reported in the SI technical details and used for Fig. 4.
- chignolin: the repo-root `chignolinTrain.py` trains each seed; `sample_fixed_cv.py` and `analyze_native.py` reproduce the native-basin RMSD analysis in SI Fig. S7 and the hydrogen-bond occupancies in SI Table S8.

Alanine metadynamics requires `conda install -c conda-forge openmm-plumed plumed`.

## Training seeds

- dimer: paper model `14498636577045679552`; additional training `1101, 1102, 1103, 1104, 1105`.
- diazene: CV model `12296939786608200374`; paper model `14741652158638001041`; additional training `2101, 2102, 2103`.
- alanine: paper model `13240477824351584163`; additional training `3102, 3103, 3104`.
- chignolin: paper model `1725444404668423095`; additional training `2026071601, 2026071602`.
