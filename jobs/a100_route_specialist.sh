#!/bin/bash
# CPU routing gate. Writes a routed JSONL only when the predeclared gate passes.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: bash jobs/a100_route_specialist.sh RUN_DIR SPLIT BASELINE_JSONL BASELINE_FAMILY" >&2
  echo "BASELINE_FAMILY is stageA, stageB, or ensemble_ab. ensemble_ab and stageB require SPLIT=val." >&2
  exit 2
fi
RUN_DIR=$1
SPLIT=$2
BASELINE=$3
FAMILY=$4
: "${WORK:?set WORK to the runtime Lustre tree}"
cd "$WORK"
case "$SPLIT" in
  val) GOLD="data/ReVA/val.json" ;;
  valQ) GOLD="data/splits/valQ.json" ;;
  valT) GOLD="data/splits/valT.json" ;;
  *)
    echo "refusing split $SPLIT" >&2
    exit 2
    ;;
esac
case "$RUN_DIR" in
  /*) OUT="$RUN_DIR" ;;
  *) OUT="$WORK/$RUN_DIR" ;;
esac
RESOLVED="$OUT/resolved_config.yaml"
ELIGIBILITY="$OUT/eligibility_audit.json"
SCORES="$OUT/scores_${SPLIT}.jsonl"
if [[ ! -f "$RESOLVED" || ! -f "$ELIGIBILITY" || ! -f "$SCORES" ]]; then
  echo "missing resolved config, eligibility audit, or scores under $OUT" >&2
  exit 2
fi
mapfile -t TASKS < <(python -c 'import sys,yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["train_tasks"]))' "$RESOLVED")
python code/route_specialist.py \
  --gold "$WORK/$GOLD" \
  --split "$SPLIT" \
  --baseline "$BASELINE" \
  --specialist "$SCORES" \
  --tasks "${TASKS[@]}" \
  --train-json "$WORK/data/splits/fit_A.json" \
  --baseline-family "$FAMILY" \
  --eligibility "$ELIGIBILITY" \
  --report "$OUT/gate_${SPLIT}_${FAMILY}.json" \
  --routed-out "$OUT/routed_${SPLIT}_${FAMILY}.jsonl"
