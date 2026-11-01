#!/bin/bash
# Build and score the fixed 32-question Phase 3 grounding-only pilot.
# Detector caches and score JSONL files are atomic/resumable; requeueing this
# job reuses completed per-video cache entries and completed QA rows.
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
#SBATCH -J reva_grounding_pilot
#SBATCH -o reva_grounding_pilot_%j.out
set -euo pipefail

: "${WORK:?set WORK to the runtime Lustre tree}"
source "$WORK/envs/reva/bin/activate"
cd "$WORK"
source jobs/wait_child.sh

pilot_ids_file="$WORK/tasks/phase3_grounding_pilot_ids.json"
model_dir="${GROUNDING_DINO_DIR:-$WORK/hf_cache/grounding-dino-tiny-a2bb814}"
adapter_dir="$WORK/runs/stageA_q3vl8b_r16/final_adapter"
run_dir="${REVA_GROUNDING_PILOT_DIR:-$WORK/runs/grounding_pilot_stageA}"
cache_dir="$run_dir/cache"
output_dir="$run_dir/outputs"
log_dir="$run_dir/logs"
mkdir -p "$cache_dir" "$output_dir" "$log_dir"

job_id="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "$log_dir/job-${job_id}.log") 2>&1

if [[ ! -f "$pilot_ids_file" ]]; then
  echo "missing fixed pilot manifest: $pilot_ids_file" >&2
  exit 2
fi
if [[ ! -d "$adapter_dir" ]]; then
  echo "missing Stage A adapter: $adapter_dir" >&2
  exit 2
fi
if [[ ! -d "$model_dir" || ! -f "$model_dir/.reva_detector_revision" ]]; then
  echo "staged Grounding DINO snapshot and revision marker are required: $model_dir" >&2
  exit 2
fi

# The snapshot must be staged before entering offline mode. The cache builder
# performs the immutable file-size/digest validation before model loading.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

splits=(valQ valT)
graceful=0
rc=0
stop_file="$log_dir/job-${job_id}.stop"
rm -f "$stop_file"
export REVA_STOP_FILE="$stop_file"

on_usr1() {
  echo "$(date -Is) received USR1; finish the current split and requeue"
  graceful=1
  touch "$stop_file"
}
trap on_usr1 USR1

requeue_after_checkpoint() {
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol requeue "$SLURM_JOB_ID"
  else
    echo "requeue requested outside Slurm; completed work remains resumable" >&2
  fi
}

# Generate all detector caches first. The split-aware manifest limits each
# invocation to exactly its 16 fixed QA IDs.
for split in "${splits[@]}"; do
  python code/build_grounding_cache.py \
    --split "$split" \
    --model-dir "$model_dir" \
    --grounding-cache-dir "$cache_dir" \
    --qa-ids-file "$pilot_ids_file" \
    --offsets 0,0.5 \
    --frame-sampling legacy \
    --device cuda \
    --batch-size 4 &
  pid=$!
  rc=0
  wait_child "$pid" || rc=$?
  if ((rc != 0)); then
    exit "$rc"
  fi
  if ((graceful)); then
    requeue_after_checkpoint
    exit 0
  fi
done

# Score the same baseline frames with the existing Stage A adapter. The output
# directory is unique to this opt-in arm and is never reused by baseline or
# adaptive scoring.
for split in "${splits[@]}"; do
  out="$output_dir/scores_${split}_grounding_pilot32.jsonl"
  python code/score.py \
    --adapter "$adapter_dir" \
    --split "$split" \
    --grounding-context \
    --grounding-cache-dir "$cache_dir" \
    --qa-ids-file "$pilot_ids_file" \
    --offsets 0,0.5 \
    --rots 2 \
    --frame-sampling legacy \
    --out "$out" &
  pid=$!
  rc=0
  wait_child "$pid" || rc=$?
  if ((rc != 0)); then
    exit "$rc"
  fi
  if ((graceful)); then
    requeue_after_checkpoint
    exit 0
  fi
done

rm -f "$stop_file"
echo "grounding pilot complete: ${splits[*]} (16 fixed QA IDs per split)"
