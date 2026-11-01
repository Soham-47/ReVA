#!/usr/bin/env bash
# Launch Temporal Grounding dual-stream training on Brev (foreground).
set -euo pipefail
WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"
export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WORK/InternVideo/InternVideo2/multi_modality:${PYTHONPATH:-}"
cd "$REPO"
exec "$VENV/bin/python" code/train.py code/configs/dualstream_temporal_grounding_brev.yaml
