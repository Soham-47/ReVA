#!/bin/bash
# One-epoch A100 specialist. Requires the matching kernel smoke marker. Does not score.
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
#SBATCH -J reva_a100_spec
#SBATCH -o reva_a100_spec_%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch jobs/a100_train_specialist.sbatch code/configs/b1_hint16.yaml" >&2
  exit 2
fi
CFG=$1
case "$(basename "$CFG")" in
  b1_hint16.yaml|b2_frames32.yaml|b1_hint16_mirror.yaml) ;;
  *)
    echo "specialist wrapper only accepts b1_hint16.yaml, b2_frames32.yaml, or b1_hint16_mirror.yaml" >&2
    exit 2
    ;;
esac

: "${WORK:?set WORK to the runtime Lustre tree}"
cd "$WORK"
# shellcheck disable=SC1091
source jobs/a100_prepare.sh
a100_prepare_env
# shellcheck disable=SC1091
source jobs/wait_child.sh
python code/specialist_data.py assert-training-allowed "$CFG"
python code/specialist_data.py validate-config "$CFG" --allow-unresolved >/dev/null

SMOKE_RUN=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["requires_smoke_run"])' "$CFG")
SMOKE_CONTRACT=$(python -c 'import sys,yaml; print({"b1":"smoke_b1","b2":"smoke_b2","b1_mirror":"smoke_b1"}[yaml.safe_load(open(sys.argv[1]))["specialist_contract"]])' "$CFG")
export REVA_SMOKE_OK
REVA_SMOKE_OK=$(python code/specialist_data.py assert-smoke-marker --work "$WORK" --run-name "$SMOKE_RUN" --contract "$SMOKE_CONTRACT")

RESOLVED=$(a100_resolve_output "$CFG")
OUT=$(dirname "$RESOLVED")
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python code/specialist_data.py audit-eligibility --work "$WORK" --out "$OUT/eligibility_audit.json"
if python -c 'import sys,yaml; raise SystemExit(0 if yaml.safe_load(open(sys.argv[1])).get("motion_hints_required") else 1)' "$RESOLVED"; then
  mapfile -t TASKS < <(python -c 'import sys,yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["train_tasks"]))' "$RESOLVED")
  python code/specialist_data.py audit-hints --work "$WORK" --out "$OUT/hint_audit.json" --tasks "${TASKS[@]}"
fi
if [[ -f "$OUT/DONE" ]]; then
  echo "specialist already complete: $OUT"
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
python code/train.py "$RESOLVED" &
pid=$!
rc=0
wait_child "$pid" || rc=$?
pid=""
if [[ -f "$OUT/DONE" ]]; then
  echo "specialist training complete: $OUT"
  exit 0
fi
if ((graceful)); then
  echo "timeout during specialist training; requeueing from the latest checkpoint"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
echo "specialist training failed (rc=$rc); not requeueing" >&2
if ((rc == 0)); then
  rc=1
fi
exit "$rc"
