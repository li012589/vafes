#!/usr/bin/env bash
set -euo pipefail

walker_dir="${1:-.}"
plumed sum_hills \
  --hills "$walker_dir/HILLS" \
  --outfile "$walker_dir/fes.dat" \
  --min -0.2,-0.2 \
  --max 0.2,0.2 \
  --bin 100,100 \
  --kt 2.4943395 \
  --mintozero
