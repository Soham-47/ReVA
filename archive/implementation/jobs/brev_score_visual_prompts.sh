#!/usr/bin/env bash
# Frozen Stage A 2x2 visual-prompt pilot on Brev.
# Usage:
#   export WORK=/home/ubuntu/reva
#   bash jobs/brev_score_visual_prompts.sh
set -euo pipefail
WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"
ADAPTER="${ADAPTER:-$WORK/runs/stageA_q3vl8b_r16/final_adapter}"
OUT_ROOT="${OUT_ROOT:-$WORK/runs/visual_prompt_pilot}"
IDS="${IDS:-$REPO/results/visual_prompt_pilot/valT_phase0_ids.json}"

export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
PY="$VENV/bin/python"
cd "$REPO"

if [[ ! -x "$PY" ]]; then
  echo "missing interpreter $PY" >&2
  exit 1
fi
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "missing Stage A adapter at $ADAPTER" >&2
  exit 1
fi

"$PY" -c "import cv2, numpy" >/dev/null

score() {
  local name="$1"
  shift
  local out="$OUT_ROOT/${name}.jsonl"
  echo "scoring $name -> $out"
  WORK="$WORK" "$PY" code/score.py \
    --adapter "$ADAPTER" \
    --split valT \
    --qa-ids-file "$IDS" \
    --out "$out" \
    "$@"
}

score bar --progress-bar
score motion --camera-motion-hint
score both --progress-bar --camera-motion-hint

BASELINE="${BASELINE:-$WORK/runs/stageA_q3vl8b_r16/legacy/scores_valT.jsonl}"
if [[ -f "$BASELINE" ]]; then
  WORK="$WORK" "$PY" code/analyze_visual_prompt_pilot.py \
    --split valT \
    --ids "$IDS" \
    --baseline "$BASELINE" \
    --arm bar "$OUT_ROOT/bar.jsonl" \
    --arm motion "$OUT_ROOT/motion.jsonl" \
    --arm both "$OUT_ROOT/both.jsonl" \
    --out "$OUT_ROOT/report.json"
fi
echo "visual prompt pilot finished"
