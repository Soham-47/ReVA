#!/bin/bash
# Score one completed specialist run. Training rows are never scored here.
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 11:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J reva_a100_score
#SBATCH -o reva_a100_score_%j.out
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: sbatch jobs/a100_score_specialist.sbatch RUN_DIR valT val" >&2
  exit 2
fi
RUN_DIR=$1
shift
: "${WORK:?set WORK to the runtime Lustre tree}"
cd "$WORK"
# shellcheck disable=SC1091
source jobs/a100_prepare.sh
a100_prepare_env
# shellcheck disable=SC1091
source jobs/wait_child.sh

case "$RUN_DIR" in
  /*) OUT="$RUN_DIR" ;;
  *) OUT="$WORK/$RUN_DIR" ;;
esac
RESOLVED="$OUT/resolved_config.yaml"
if [[ ! -f "$OUT/DONE" || ! -d "$OUT/final_adapter" || ! -f "$RESOLVED" ]]; then
  echo "specialist run is incomplete: $OUT" >&2
  exit 2
fi
for split in "$@"; do
  case "$split" in
    valQ|valT|val) ;;
    *)
      echo "refusing split $split; this wrapper scores valQ, valT, or official val only" >&2
      exit 2
      ;;
  esac
done

mapfile -t TASKS < <(python -c 'import sys,yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["train_tasks"]))' "$RESOLVED")
TASK_CSV=$(IFS=,; echo "${TASKS[*]}")
FRAME_BUDGET=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["frame_budget"])' "$RESOLVED")
MOTION=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("motion_hints") or "")' "$RESOLVED")
stop_file="$OUT/SCORE_STOP"
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
  cmd=(python code/score.py --adapter "$OUT/final_adapter" --split "$split" --out "$OUT/scores_${split}.jsonl" --frame-budget "$FRAME_BUDGET" --tasks "$TASK_CSV")
  if [[ -n "$MOTION" ]]; then
    cmd+=(--motion-hints "$MOTION")
  fi
  "${cmd[@]}" &
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
exit "$rc"
