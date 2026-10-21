#!/bin/bash
# Build offline SigLIP2 feature caches for selected ReVA validation splits.
# Cache files are atomic and reusable, so re-running this job resumes safely.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=96G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_cache
#SBATCH -o reva_adaptive_cache_%j.out
set -euo pipefail

: "${WORK:?set WORK to the runtime Lustre tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$WORK"
source jobs/wait_child.sh

if (($#)); then
  splits=("$@")
else
  splits=(valQ valT val)
fi

model_dir="$WORK/hf_cache/siglip2-base-patch16-224-022b6f7"
cache_dir="${REVA_FEATURE_CACHE_DIR:-runs/feature_cache}"
mkdir -p "$WORK/logs"
graceful=0
rc=0
on_usr1() {
  echo "$(date -Is) received USR1; will requeue after the current split"
  graceful=1
}
trap on_usr1 USR1
for split in "${splits[@]}"; do
  case "$split" in
    valQ|valT|val) ;;
    *) echo "unsupported validation split: $split" >&2; exit 2 ;;
  esac
  python code/build_feature_cache.py \
    --split "$split" \
    --model-dir "$model_dir" \
    --feature-cache-dir "$cache_dir" \
    --device cuda &
  pid=$!
  rc=0
  wait_child "$pid" || rc=$?
  if ((rc != 0)) || ((graceful)); then
    break
  fi
done

if ((graceful)); then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if ((rc != 0)); then
  exit "$rc"
fi
echo "adaptive feature caches complete for: ${splits[*]}"
