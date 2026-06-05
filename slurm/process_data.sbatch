#!/bin/bash
#SBATCH --job-name=rnog_ml_data
#SBATCH --account=PAS2608
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/data_%A_%a.out
#SBATCH --error=logs/data_%A_%a.err

set -euo pipefail

cd /fs/ess/PAS2608/alisa/rno_g/TS_ML

set +u
source /users/PCON0003/anozdrina/venvs/py312-env/bin/activate
set -u

if [[ -n "${RUN_LIST:-}" ]]; then
  RUN_NUMBER=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${RUN_LIST}")
else
  : "${RUN_START:?Set RUN_START or RUN_LIST. Example: sbatch --array=0-99 --export=ALL,RUN_START=1000 slurm/process_data.sbatch}"
  RUN_NUMBER=$((RUN_START + SLURM_ARRAY_TASK_ID))
  if [[ -n "${RUN_END:-}" ]] && (( RUN_NUMBER > RUN_END )); then
    echo "RUN_NUMBER=${RUN_NUMBER} is larger than RUN_END=${RUN_END}; exiting."
    exit 0
  fi
fi

echo "Processing station 23 run ${RUN_NUMBER}"

python build_ml_run_h5.py \
  --config build_ml_run_h5.example.yaml \
  --run "${RUN_NUMBER}"
