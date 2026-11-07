#!/bin/bash
# CPU checks before the motion diagnostic. Does not submit a job.
set -euo pipefail

WORK_FROM_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work)
      WORK_FROM_FLAG=$2
      shift 2
      ;;
    *)
      echo "usage: bash preflight_motion.sh --work /scratch/ambpdc/soham/ReVA" >&2
      exit 2
      ;;
  esac
done
if [[ -n "$WORK_FROM_FLAG" ]]; then
  WORK=$WORK_FROM_FLAG
elif [[ -z "${WORK:-}" ]]; then
  WORK=/scratch/ambpdc/soham/ReVA
fi
export WORK
# shellcheck disable=SC1091
source "$WORK/jobs/activate_reva.sh"
cd "$WORK"

python code/specialist_data.py assert-training-allowed code/configs/smoke_b1_hint16.yaml
python code/specialist_data.py assert-training-allowed code/configs/b1_hint16.yaml
set +e
python code/specialist_data.py assert-training-allowed code/configs/b2_frames32.yaml
blocked=$?
set -e
if [[ "$blocked" -eq 0 ]]; then
  echo "REFUSE: temporal config is launchable" >&2
  exit 1
fi
python code/specialist_data.py assert-adapter-sha --work "$WORK"
python - <<'PY'
from pathlib import Path
import common as C
text = Path("code/score.py").read_text()
if text.index("model.to(device)") > text.index("load_peft_for_inference"):
    raise SystemExit("score.py still merges before CUDA")
if C.video_size_for(16, None)["longest_edge"] != 16 * 640 * 640:
    raise SystemExit("16-frame pixel cap drifted")
print("score_cuda_order_ok video_size_16_ok")
PY
audit=$(mktemp)
hints=$(mktemp)
python code/specialist_data.py audit-eligibility --work "$WORK" --out "$audit"
python code/specialist_data.py audit-hints --work "$WORK" --out "$hints" --tasks "Perspective and Viewpoint" "Change Detection"
python code/specialist_data.py assert-live-counts "$audit" --hints "$hints"
rm -f "$audit" "$hints"
if [[ -d "$WORK/runs/b1_q3vl8b_hint16" ]]; then
  echo "preserved existing run: $WORK/runs/b1_q3vl8b_hint16"
fi
echo "PREFLIGHT_OK"
echo "Submit the motion diagnostic only:"
echo "  cd $WORK && sbatch --export=ALL,WORK=$WORK jobs/a100_train_smoke.sbatch code/configs/smoke_b1_hint16.yaml"
echo "After that directory contains SMOKE_OK, submit the one-epoch motion specialist:"
echo "  cd $WORK && sbatch --export=ALL,WORK=$WORK jobs/a100_train_specialist.sbatch code/configs/b1_hint16.yaml"
echo "Do not submit smoke_b2_frames32.yaml or b2_frames32.yaml."
