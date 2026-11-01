#!/bin/bash
# Score adaptive frames with the existing adapter, then evaluate each split.
# Outputs are versioned evidence files and do not overwrite baseline scores.
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
#SBATCH -J reva_adaptive
#SBATCH -o reva_adaptive_score_%j.out
set -euo pipefail

if (($# < 2)); then
  echo "usage: sbatch jobs/adaptive_score.sbatch runs/X/final_adapter valQ valT val" >&2
  exit 2
fi
: "${WORK:?set WORK to the runtime Lustre tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HOME="$WORK/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$WORK"
source jobs/wait_child.sh

adapter=$1
shift
cache_dir="${REVA_FEATURE_CACHE_DIR:-runs/feature_cache}"
output_dir="${REVA_ADAPTIVE_OUTPUT_DIR:-$(dirname "$adapter")}" 
mkdir -p "$output_dir"
for split in "$@"; do
  case "$split" in
    valQ|valT|val) ;;
    *) echo "unsupported validation split: $split" >&2; exit 2 ;;
  esac
done

mkdir -p "$WORK/logs"
stop_file="$WORK/logs/reva_adaptive_score_${SLURM_JOB_ID}.stop"
rm -f "$stop_file"
export REVA_STOP_FILE="$stop_file"
graceful=0
rc=0
on_usr1() {
  echo "$(date -Is) received USR1; requesting score checkpoint and graceful requeue"
  graceful=1
  touch "$stop_file"
}
trap on_usr1 USR1

for split in "$@"; do
  out="$output_dir/scores_${split}_adaptive.jsonl"
  python code/score.py \
    --adapter "$adapter" \
    --split "$split" \
    --adaptive-frames \
    --feature-cache-dir "$cache_dir" \
    --out "$out" &
  pid=$!
  rc=0
  wait_child "$pid" || rc=$?
  if ((rc != 0)) || ((graceful)); then
    break
  fi
  evidence="${out%.jsonl}.evidence.selector-v2.grounding-v1.jsonl"
  python code/evaluate.py "$evidence" "$split"
done

if ((graceful)); then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if ((rc != 0)); then
  exit "$rc"
fi
rm -f "$stop_file"
echo "adaptive scoring complete for: $*"
