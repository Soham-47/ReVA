#!/bin/bash
# Opt-in A.I.R.-style ReVA scoring. Writes only under runs/air_*.
# The legacy score.sbatch path and all EXP-001/004 outputs remain untouched.
# Usage: sbatch jobs/air_score.sbatch air16 val
#        sbatch jobs/air_score.sbatch air32 val
#        sbatch jobs/air_score.sbatch uniform32 val
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
#SBATCH --job-name=reva_air
#SBATCH --output=reva_air_%j.out
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch jobs/air_score.sbatch air16|air32|uniform32 val|valQ|valT" >&2
  exit 2
fi
arm="$1"
split="$2"
case "$arm" in
  air16) frame_budget=16; mode=air ;;
  air32) frame_budget=32; mode=air ;;
  uniform32) frame_budget=32; mode=uniform ;;
  *) echo "unknown AIR arm: $arm" >&2; exit 2 ;;
esac
case "$split" in
  val|valQ|valT) ;;
  test) echo "AIR experiment is val-only; refusing test scoring" >&2; exit 2 ;;
  *) echo "unsupported split: $split" >&2; exit 2 ;;
esac

: "${WORK:?set WORK to the prepared ReVA runtime tree}"
env_activate="${REVA_ENV:-$WORK/envs/reva}"
if [[ -d "$env_activate" ]]; then env_activate="$env_activate/bin/activate"; fi
if [[ ! -f "$env_activate" ]]; then echo "missing prepared environment: $env_activate" >&2; exit 2; fi
source "$env_activate"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$WORK"
source jobs/wait_child.sh

adapter="${AIR_ADAPTER:-$WORK/runs/stageA_q3vl8b_r16/final_adapter}"
feature_cache="${AIR_FEATURE_CACHE:-$WORK/runs/air_feature_cache}"
run_dir="${AIR_RUN_ROOT:-$WORK/runs/air_${arm}}"
mkdir -p "$run_dir/logs"
if [[ "$run_dir" == *stageA* || "$run_dir" == *stageB* || "$run_dir" == *phase8* ]]; then
  echo "AIR output directory is not isolated: $run_dir" >&2
  exit 2
fi
job_id="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "$run_dir/logs/air_${job_id}.log") 2>&1
stop_file="$run_dir/STOP"
rm -f "$stop_file"
export REVA_STOP_FILE="$stop_file"
graceful=0
pid=""
on_usr1() {
  echo "$(date -Is) received USR1; requesting AIR checkpoint"
  graceful=1
  touch "$stop_file"
}
trap on_usr1 USR1

extra=(--adapter "$adapter" --split "$split" --offsets 0.0,0.5 --rots 2)
if [[ "$mode" == air ]]; then
  extra+=(
    --air --frame-budget "$frame_budget"
    --feature-cache-dir "$feature_cache"
    --air-candidate-count 64
    --air-max-iterations "${AIR_MAX_ITERATIONS:-3}"
    --air-vlm-batch-size "${AIR_VLM_BATCH_SIZE:-6}"
    --out "$run_dir/scores_${split}.jsonl"
  )
else
  extra+=(--frame-sampling endpoints --frame-budget "$frame_budget" --out "$run_dir/scores_${split}.jsonl")
fi
if [[ -n "${AIR_QA_IDS_FILE:-}" ]]; then
  extra+=(--qa-ids-file "$AIR_QA_IDS_FILE")
fi
if [[ -n "${AIR_VIDEO_PATHS_FILE:-}" ]]; then
  extra+=(--video-paths-file "$AIR_VIDEO_PATHS_FILE")
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python code/score.py "${extra[@]}" &
pid=$!
rc=0
wait_child "$pid" || rc=$?
pid=""
if ((rc != 0)); then echo "AIR $arm scoring failed (rc=$rc)" >&2; exit "$rc"; fi
if ((graceful)); then
  echo "requeueing AIR scoring from resumable JSONL"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
rm -f "$stop_file"
echo "AIR $arm scoring complete: $run_dir/scores_${split}.jsonl"
