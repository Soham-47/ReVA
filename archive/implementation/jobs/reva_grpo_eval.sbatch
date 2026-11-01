#!/bin/bash
# Paired, frozen validation of a completed ReVA answer-only GRPO run.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_rs_grpo_eval
#SBATCH -o reva_rs_grpo_eval_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$WORK/hf_cache" HF_HUB_CACHE="$WORK/hf_cache/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"
RUN="$WORK/runs/reva_rs_video_grpo_answer_20261023"
A0="$WORK/runs/stageA_q3vl8b_r16/final_adapter"
ADAPTER="$RUN/final_adapter"
STOP_FILE="$RUN/STOP_EVAL"
mkdir -p "$WORK/logs"
exec > >(tee -a "$WORK/logs/reva_rs_grpo_eval_${SLURM_JOB_ID:-manual}.log") 2>&1
if [[ ! -f "$RUN/DONE" || ! -d "$ADAPTER" ]]; then
  echo "GRPO run is incomplete: $RUN" >&2
  exit 2
fi
rm -f "$STOP_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
graceful=0
on_usr1() {
  echo "$(date -Is) received USR1; requesting scorer checkpoint and graceful stop"
  graceful=1
  touch "$STOP_FILE"
}
trap on_usr1 USR1
for split in valQ valT; do
  baseline="$RUN/scores_A0_${split}.jsonl"
  candidate="$RUN/scores_${split}.jsonl"
  if [[ ! -f "${baseline%.jsonl}_summary.json" ]]; then
    export REVA_STOP_FILE="$STOP_FILE"
    python code/score.py --adapter "$A0" --split "$split" \
      --offsets 0,0.5 --rots 2 --out "$baseline"
    unset REVA_STOP_FILE
    if ((graceful)); then
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
    python code/evaluate.py "$baseline" "$split"
    if ((graceful)); then
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
  fi
  if [[ ! -f "${candidate%.jsonl}_summary.json" ]]; then
    export REVA_STOP_FILE="$STOP_FILE"
    python code/score.py --adapter "$ADAPTER" --split "$split" \
      --offsets 0,0.5 --rots 2 --out "$candidate"
    unset REVA_STOP_FILE
    if ((graceful)); then
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
    python code/evaluate.py "$candidate" "$split"
    if ((graceful)); then
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
  fi
  python code/reva_grpo.py compare --work "$WORK" --split "$split" \
    --baseline "$baseline" --candidate "$candidate" --output "$RUN/paired_${split}.json"
  if ((graceful)); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
done
echo "paired ReVA validation complete: $RUN"
