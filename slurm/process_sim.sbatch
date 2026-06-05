#!/bin/bash
#SBATCH --job-name=rnog_ml_sim
#SBATCH --account=PAS2608
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err

set -euo pipefail

cd /fs/ess/PAS2608/alisa/rno_g/TS_ML

set +u
source /users/PCON0003/anozdrina/venvs/py312-env/bin/activate
set -u

SIM_LIST=${SIM_LIST:-slurm/lists/nc_cr_proxy_nur.txt}
SIM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SIM_LIST}")
if [[ -z "${SIM_FILE}" ]]; then
  echo "No simulation file on line ${SLURM_ARRAY_TASK_ID} of ${SIM_LIST}; exiting."
  exit 0
fi

echo "Processing simulation file ${SIM_FILE}"

python build_ml_run_h5.py \
  --config build_ml_sim.example.yaml \
  --input-file "${SIM_FILE}"
