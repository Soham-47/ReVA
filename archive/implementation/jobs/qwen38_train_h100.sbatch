#!/bin/bash
# Isolated Qwen3.8 Stage A-equivalent full training on H100.
# Requires the completed finite smoke marker from the H100 smoke config.
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_qwen38_h100
#SBATCH -o reva_qwen38_h100_%j.out
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: sbatch jobs/qwen38_train_h100.sbatch code/configs/stageA_qwen38_h100.yaml [train]" >&2
  exit 2
fi
CFG=$1
MODE=${2:-train}
if [[ "$MODE" != train ]]; then
  echo "H100 training wrapper accepts only train mode" >&2
  exit 2
fi
if [[ "$CFG" != *qwen38_h100.yaml ]]; then
  echo "H100 wrapper requires the isolated H100 Qwen3.8 training config: $CFG" >&2
  exit 2
fi
: "${WORK:?set WORK to the runtime Lustre tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WORK"
source jobs/wait_child.sh

OUT="$WORK/$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CFG")"
SMOKE_MARKER="$WORK/runs/stageA_qwen38_27b_r16_h100_smoke/SMOKE_OK"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ -f "$OUT/DONE" ]]; then
  echo "already complete: $OUT"
  exit 0
fi
if [[ ! -f "$SMOKE_MARKER" ]]; then
  echo "refusing training: missing completed smoke marker $SMOKE_MARKER" >&2
  exit 2
fi

pid=""
graceful=0
on_usr1() {
  echo "$(date -Is) received USR1; requesting checkpoint and graceful stop"
  graceful=1
  if [[ -n ${pid:-} ]]; then
    touch "$OUT/STOP"
  fi
}
trap on_usr1 USR1

python code/train.py "$CFG" &
pid=$!
rc=0
wait_child "$pid" || rc=$?
pid=""
if [[ -f "$OUT/DONE" ]]; then
  exit 0
fi
if ((graceful)); then
  echo "training stopped by timeout (rc=$rc); requeueing from latest checkpoint"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "training failed (rc=$rc); not requeueing" >&2
if ((rc == 0)); then
  rc=1
fi
exit "$rc"
