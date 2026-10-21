#!/bin/bash
# Controlled Phase 6 A1/A2 pilot. The default mode is smoke-only. Run the
# explicit `train` mode only after reviewing the completed smoke allocation.
#SBATCH -p gpu_h200_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_phase6
#SBATCH -o reva_phase6_%j.out
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: sbatch jobs/phase6_train.sbatch code/configs/phase6_A1.yaml [smoke|train]" >&2
  exit 2
fi
CFG=$1
MODE=${2:-smoke}
if [[ "$MODE" != smoke && "$MODE" != train ]]; then
  echo "mode must be smoke or train (default: smoke)" >&2
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
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ -f "$OUT/DONE" ]]; then
  echo "already complete: $OUT"
  exit 0
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

SMOKE_MARKER="$OUT/SMOKE_OK"
if [[ "$MODE" == smoke ]]; then
  if [[ ! -f "$SMOKE_MARKER" ]]; then
    smoke_args=()
    if [[ -n ${SMOKE_QA_ID:-} ]]; then
      smoke_args+=(--qa-id "$SMOKE_QA_ID")
    fi
    python code/smoke.py --config "$CFG" "${smoke_args[@]}"
    touch "$SMOKE_MARKER"
  else
    echo "smoke already passed: $SMOKE_MARKER"
  fi
  if ((graceful)); then
    echo "timeout requested during smoke; requeueing smoke-only allocation"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  echo "smoke-only mode complete; training was not started"
  exit 0
fi

if [[ ! -f "$SMOKE_MARKER" ]]; then
  echo "refusing training: run this config in smoke mode first and review $SMOKE_MARKER" >&2
  exit 2
fi

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
