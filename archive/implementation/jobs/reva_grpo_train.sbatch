#!/bin/bash
# One complete shuffled pass of answer-only GRPO on the ReVA fit_A split.
# Checkpoints and graceful Slurm requeues preserve the run across allocations.
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
#SBATCH -J reva_rs_grpo
#SBATCH -o reva_rs_grpo_%j.out
set -euo pipefail

export WORK="${WORK:-/scratch/ambpdc/soham/ReVA}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$WORK/hf_cache" HF_HUB_CACHE="$WORK/hf_cache/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"
source jobs/wait_child.sh

RUN="$WORK/runs/reva_rs_video_grpo_answer_20261023"
SMOKE_RUN="$WORK/runs/reva_rs_video_grpo_answer_20261023_smoke"
TRAIN_JSON="$WORK/data/splits/fit_A.json"
STAGE_A="$WORK/runs/stageA_q3vl8b_r16/final_adapter"
STOP_FILE="$RUN/STOP"
mkdir -p "$RUN" "$SMOKE_RUN" "$WORK/logs"
exec > >(tee -a "$WORK/logs/reva_rs_grpo_${SLURM_JOB_ID:-manual}.log") 2>&1
if [[ -f "$RUN/DONE" ]]; then
  echo "already complete: $RUN"
  exit 0
fi
if [[ ! -s "$TRAIN_JSON" || ! -d "$STAGE_A" ]]; then
  echo "missing ReVA fit_A or Stage A adapter" >&2
  exit 2
fi
rm -f "$STOP_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python code/reva_grpo.py self-check

if [[ ! -f "$SMOKE_RUN/SMOKE_OK" ]]; then
  python code/reva_grpo.py preflight \
    --work "$WORK" --train-json "$TRAIN_JSON" \
    --init-adapter "$STAGE_A" --reference-adapter "$STAGE_A" \
    --output-dir "$RUN" --group-size 4 --temperature 0.7 \
    --max-new-tokens 4 --learning-rate 1e-5 --kl-beta 0.03
  touch "$SMOKE_RUN/SMOKE_OK"
fi

graceful=0
pid=""
on_usr1() {
  echo "$(date -Is) received USR1; requesting checkpoint and graceful stop"
  graceful=1
  touch "$STOP_FILE"
}
trap on_usr1 USR1
export REVA_STOP_FILE="$STOP_FILE"
python code/reva_grpo.py train \
  --work "$WORK" --train-json "$TRAIN_JSON" \
  --init-adapter "$STAGE_A" --reference-adapter "$STAGE_A" \
  --output-dir "$RUN" --group-size 4 --temperature 0.7 \
  --max-new-tokens 4 --learning-rate 1e-5 --kl-beta 0.03 \
  --clip-range 0.2 --warmup-ratio 0.03 --max-tokens 12000 \
  --save-steps 500 --log-steps 10 &
pid=$!
rc=0
wait_child "$pid" || rc=$?
pid=""
if [[ -f "$RUN/DONE" ]]; then
  exit 0
fi
if ((graceful)) || ((rc == 75)); then
  echo "requeueing from latest valid GRPO checkpoint (rc=$rc)"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "GRPO training failed (rc=$rc); not requeueing" >&2
if ((rc == 0)); then rc=1; fi
exit "$rc"
