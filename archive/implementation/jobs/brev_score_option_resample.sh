#!/usr/bin/env bash
# Option-conditioned two-pass resampling on valQ and valT (frozen Stage A on Brev).
set -euo pipefail

WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"
ADAPTER="${ADAPTER:-$WORK/runs/stageA_q3vl8b_r16/final_adapter}"
OUT_ROOT="${OUT_ROOT:-$WORK/runs/option_resample_tg}"
SCORE="${SCORE:-$REPO/archive/implementation/code/score_stageA_option_resample.py}"

export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
PY="$VENV/bin/python"

cd "$REPO"
test -f "$ADAPTER/adapter_config.json"
test -f "$SCORE"
mkdir -p "$OUT_ROOT"

for split in valQ valT; do
  out="$OUT_ROOT/scores_${split}.jsonl"
  echo "Scoring Option-Resample $split -> $out"
  WORK="$WORK" "$PY" "$SCORE" \
    --adapter "$ADAPTER" \
    --split "$split" \
    --fusion-mode "confidence_weighted" \
    --out "$out"
  
  echo "Evaluating $split..."
  "$PY" "$REPO/code/evaluate.py" "$out" "$split" | tee "$OUT_ROOT/${split}_eval.txt"
done

echo "Option-resampling scoring complete across valQ and valT!"
