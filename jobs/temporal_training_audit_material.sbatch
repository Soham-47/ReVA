#!/bin/bash
# Render the frozen blind temporal audit sample with the existing CPU runtime.
# The job writes only review sheets and a manifest under the new audit run.
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J reva_temporal_audit
#SBATCH -o runs/temporal_training_data_audit/review_material/slurm-%j.out
set -euo pipefail

work_root="${REVA_WORK_ROOT:-/scratch/ambpdc/soham/ReVA}"
python_bin="$work_root/envs/reva/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "missing HPC Python runtime: $python_bin" >&2
  exit 2
fi

cd "$work_root"
exec "$python_bin" jobs/build_temporal_training_review_material.py --work-root "$work_root"
