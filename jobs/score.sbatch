#!/bin/bash
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
#SBATCH -J reva_score
#SBATCH -o reva_score_%j.out
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch jobs/score.sbatch runs/X/final_adapter valQ valT val" >&2
  exit 2
fi
: "${WORK:?set WORK to the runtime Lustre tree}"
source "$WORK/envs/reva/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$WORK/hf_cache"
cd "$WORK"
source jobs/wait_child.sh
adapter=$1
shift
stop_file="$(dirname "$adapter")/SCORE_STOP"
rm -f "$stop_file"
export REVA_STOP_FILE="$stop_file"
graceful=0
rc=0
on_usr1() {
  echo "$(date -Is) received USR1; requesting score checkpoint and graceful stop"
  graceful=1
  touch "$stop_file"
}
trap on_usr1 USR1
for split in "$@"; do
  python code/score.py --adapter "$adapter" --split "$split" --out "$(dirname "$adapter")/scores_${split}.jsonl" &
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
