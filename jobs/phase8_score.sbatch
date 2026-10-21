#!/bin/bash
# Resumable Phase 8 pretrained-backbone scoring on one H200.
# This wrapper writes only under a phase8_* run directory and never invokes
# the frozen Stage A scorer or reuses its score paths.
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH --time=11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --job-name=reva_phase8
#SBATCH --output=reva_phase8_%j.out
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: PHASE8_MODE=pilot|full sbatch jobs/phase8_score.sbatch qwen38 valQ [valT val]" >&2
  echo "       or: PHASE8_MODE=pilot|full sbatch jobs/phase8_score.sbatch qwen3vl32b valQ [valT val]" >&2
  exit 2
fi

mode="${PHASE8_MODE:-full}"
case "$mode" in
  pilot|full) ;;
  *) echo "PHASE8_MODE must be pilot or full: $mode" >&2; exit 2 ;;
esac

family="$1"
shift
case "$family" in
  qwen38|qwen3.8|qwen3.8-27b) family_key=qwen38 ;;
  qwen3vl32|qwen3vl32b|qwen3-vl-32b|qwen3-vl-32b-instruct) family_key=qwen3vl32b ;;
  *) echo "unsupported Phase 8 family: $family" >&2; exit 2 ;;
esac
for split in "$@"; do
  case "$split" in
    valQ|valT|val|test) ;;
    *) echo "unsupported split: $split" >&2; exit 2 ;;
  esac
  if [[ "$mode" == pilot && "$split" != valQ && "$split" != valT ]]; then
    echo "Phase 8 pilot mode supports only valQ and valT: $split" >&2
    exit 2
  fi
done

: "${WORK:?set WORK to the prepared ReVA runtime tree}"
env_activate="${REVA_ENV:-$WORK/envs/reva}"
if [[ -d "$env_activate" ]]; then
  env_activate="$env_activate/bin/activate"
fi
if [[ ! -f "$env_activate" ]]; then
  echo "missing prepared Python environment: $env_activate" >&2
  exit 2
fi
source "$env_activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "$WORK"
source jobs/wait_child.sh

python_bin="${REVA_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable is unavailable: $python_bin" >&2
  exit 2
fi

if [[ "$mode" == pilot ]]; then
  pilot_selection="${PHASE8_PILOT_SELECTION:-$WORK/runs/phase8_backbone_comparison/pilot_selection.json}"
  if [[ ! -f "$pilot_selection" ]]; then
    echo "missing frozen Phase 8 pilot selection: $pilot_selection" >&2
    exit 2
  fi
  selection_args=(--qa-ids-file "$pilot_selection")
  run_dir="${PHASE8_RUN:-$WORK/runs/phase8_pilot_${family_key}}"
else
  selection_args=()
  run_dir="${PHASE8_RUN:-$WORK/runs/phase8_${family_key}}"
fi
run_name="$(basename "$run_dir")"
if [[ "$mode" == pilot && "$run_name" != phase8_pilot_* ]]; then
  echo "pilot PHASE8_RUN must be a phase8_pilot_* directory: $run_dir" >&2
  exit 2
fi
if [[ "$mode" == full && ("$run_name" != phase8_* || "$run_name" == phase8_pilot_*) ]]; then
  echo "full PHASE8_RUN must be a phase8_* directory (not phase8_pilot_*): $run_dir" >&2
  exit 2
fi
if [[ "$run_dir" == *stageA* || "$run_dir" == *stageB* ]]; then
  echo "PHASE8_RUN must be separate from Stage A/B: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir/logs"
job_id="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "$run_dir/logs/phase8_${job_id}.log") 2>&1

stop_file="$run_dir/STOP"
rm -f "$stop_file"
export REVA_STOP_FILE="$stop_file"

extra_args=()
if [[ -n ${PHASE8_REVISION:-} ]]; then
  extra_args+=(--revision "$PHASE8_REVISION")
fi
if [[ -n ${PHASE8_ADAPTER:-} ]]; then
  extra_args+=(--adapter "$PHASE8_ADAPTER")
fi
if [[ -n ${PHASE8_FRAME_MANIFEST:-} ]]; then
  extra_args+=(--frame-manifest "$PHASE8_FRAME_MANIFEST")
fi

graceful=0
pid=""
on_usr1() {
  echo "$(date -Is) received USR1; requesting Phase 8 score checkpoint and graceful stop"
  graceful=1
  if [[ -n ${pid:-} ]]; then
    touch "$stop_file"
  fi
}
trap on_usr1 USR1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$python_bin" code/phase8_scorer.py \
  --family "$family_key" \
  --split "$@" \
  --out "$run_dir/scores_${family_key}_{split}.jsonl" \
  "${extra_args[@]}" \
  "${selection_args[@]}" &
pid=$!
rc=0
wait_child "$pid" || rc=$?
pid=""
if ((rc != 0)); then
  echo "Phase 8 $mode scoring failed for $family_key ($*) (rc=$rc)" >&2
  exit "$rc"
fi

if ((graceful)); then
  echo "requeueing Phase 8 scoring from resumable JSONL outputs"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
rm -f "$stop_file"
echo "Phase 8 $mode scoring complete: $family_key ($*)"

