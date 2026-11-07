#!/bin/bash
# A100 kernel diagnostic: 5 forward/backward/AdamW steps at grad_accum 1.
# Wall limit is a kill cap, not an ETA. Reported inference startup is about 6 minutes
# and the forward probe was 9m15s. Backward duration is unmeasured. No resubmit.
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 00:20:00
#SBATCH --open-mode=append
#SBATCH -J reva_a100_smoke
#SBATCH -o reva_a100_smoke_%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch jobs/a100_train_smoke.sbatch code/configs/smoke_b1_hint16.yaml" >&2
  exit 2
fi
CFG=$1
case "$(basename "$CFG")" in
  smoke_b1_hint16.yaml|smoke_b2_frames32.yaml) ;;
  *)
    echo "smoke wrapper only accepts smoke_b1_hint16.yaml or smoke_b2_frames32.yaml" >&2
    exit 2
    ;;
esac

: "${WORK:?set WORK to the runtime Lustre tree}"
cd "$WORK"
# shellcheck disable=SC1091
source jobs/a100_prepare.sh
a100_prepare_env
python code/specialist_data.py assert-training-allowed "$CFG"
python code/specialist_data.py validate-config "$CFG" --allow-unresolved >/dev/null

RESOLVED=$(a100_resolve_output "$CFG")
OUT=$(dirname "$RESOLVED")
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python code/specialist_data.py audit-eligibility --work "$WORK" --out "$OUT/eligibility_audit.json"
if python -c 'import sys,yaml; raise SystemExit(0 if yaml.safe_load(open(sys.argv[1])).get("motion_hints_required") else 1)' "$RESOLVED"; then
  mapfile -t TASKS < <(python -c 'import sys,yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["train_tasks"]))' "$RESOLVED")
  python code/specialist_data.py audit-hints --work "$WORK" --out "$OUT/hint_audit.json" --tasks "${TASKS[@]}"
fi
if [[ -f "$OUT/DONE" && -f "$OUT/SMOKE_OK" && -f "$OUT/KERNEL_SMOKE_STEP_OK" ]]; then
  echo "smoke already complete: $OUT"
  exit 0
fi

set +e
python code/train.py "$RESOLVED"
rc=$?
set -e
if [[ -f "$OUT/DONE" && -f "$OUT/KERNEL_SMOKE_STEP_OK" ]]; then
  python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
states = sorted(out.glob("checkpoint-*/trainer_state.json"))
if not states:
    raise SystemExit("smoke finished without a Trainer checkpoint")
state = json.loads(states[-1].read_text())
step = int(state["global_step"])
if not 5 <= step <= 20:
    raise SystemExit(f"smoke global_step {step} is outside 5..20")
PY
  touch "$OUT/SMOKE_OK"
  echo "kernel smoke complete: $OUT"
  exit 0
fi
echo "kernel smoke stopped (rc=$rc) before KERNEL_SMOKE_STEP_OK. A wall-clock stop is inconclusive; backward time was not measured." >&2
if ((rc == 0)); then
  rc=1
fi
exit "$rc"
