#!/bin/bash
# Run only the Phase 3A two-video SigLIP2 retrieval stage on one H200.
# Replicate SAM3 artifacts must already exist under RUN_RELATIVE; this job
# never contacts Replicate, Modal, Hugging Face, or any other provider.
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=reva_phase3a_hpc
#SBATCH --open-mode=append
#SBATCH --output=reva_phase3a_hpc_%j.out
set -euo pipefail

: "${WORK:?set WORK to the prepared ReVA runtime tree}"
cd "$WORK"

# Prepare the environment before submission, for example:
#   module load python-3.10.8 cuda/12.1
#   sbatch --export=ALL,WORK=/path/to/ReVA,REVA_ENV=/path/to/env \
#     jobs/hpc_phase3a_retrieval_smoke.sbatch
env_activate="${REVA_ENV:-$WORK/envs/reva/bin/activate}"
if [[ -d "$env_activate" ]]; then
  env_activate="$env_activate/bin/activate"
fi
if [[ ! -f "$env_activate" ]]; then
  echo "missing prepared Python environment: $env_activate" >&2
  exit 2
fi
source "$env_activate"

python_bin="${REVA_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 2
fi

export PHASE3A_RUN="${PHASE3A_RUN:-runs/phase3a_retrieval_smoke}"
export SIGLIP2_DIR="${SIGLIP2_DIR:-hf_cache/siglip2-base-patch16-224-022b6f7}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$WORK/logs"
job_id="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "$WORK/logs/phase3a_hpc_${job_id}.log") 2>&1

"$python_bin" jobs/modal_phase3a_retrieval_smoke.py \
  --backend hpc \
  --work-root "$WORK"
