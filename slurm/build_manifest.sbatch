#!/bin/bash
#SBATCH --job-name=rnog_ml_manifest
#SBATCH --account=PAS2608
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/manifest_%j.out
#SBATCH --error=logs/manifest_%j.err

set -euo pipefail

cd /fs/ess/PAS2608/alisa/rno_g/TS_ML

set +u
source /users/PCON0003/anozdrina/venvs/py312-env/bin/activate
set -u

python build_training_manifest.py \
  'ml_dataset/data/station23_2022/*.h5' \
  'ml_dataset/sim/nc_cr_proxy/*.h5' \
  --output ml_dataset/training_manifest.csv \
  --relative-to .
