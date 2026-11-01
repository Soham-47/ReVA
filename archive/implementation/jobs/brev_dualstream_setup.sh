#!/usr/bin/env bash
# One-shot runtime bootstrap for dual-stream temporal grounding on Brev.
# Usage (on the instance):
#   export WORK=/home/ubuntu/reva
#   bash jobs/brev_dualstream_setup.sh
set -euo pipefail

WORK="${WORK:-/home/ubuntu/reva}"
REPO="${REPO:-$WORK/repo}"
VENV="${VENV:-$WORK/venv312}"

export WORK HF_HOME="$WORK/hf_cache" HUGGINGFACE_HUB_CACHE="$WORK/hf_cache/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WORK/InternVideo/InternVideo2/multi_modality:${PYTHONPATH:-}"

PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv at $VENV; run the Brev bootstrap first." >&2
  exit 1
fi

mkdir -p "$WORK/models" "$WORK/runs" "$HF_HOME/hub"

if [[ ! -d "$WORK/data/ReVA" ]]; then
  echo "Extract data/ReVA under WORK first (HPC reva_data.tar)." >&2
  exit 1
fi

if [[ ! -d "$WORK/InternVideo/.git" ]]; then
  git clone --depth 1 https://github.com/OpenGVLab/InternVideo.git "$WORK/InternVideo"
fi

"$PY" -m pip install -q timm einops decord

IV_PKG="$WORK/InternVideo/InternVideo2/multi_modality/models/backbones/internvideo2"
MODELS_INIT="$WORK/InternVideo/InternVideo2/multi_modality/models/__init__.py"
if [[ -d "$IV_PKG" ]]; then
  cat > "$IV_PKG/flash_attention_class.py" <<'PY'
"""Stub for ReVA temporal branch when use_flash_attn=False."""
import torch.nn as nn

class FlashAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("FlashAttention stub; keep temporal_stream.use_flash_attn false")
PY
  cat > "$IV_PKG/__init__.py" <<'PY'
from .internvideo2 import pretrain_internvideo2_1b_patch14_224, pretrain_internvideo2_6b_patch14_224

__all__ = [
    "pretrain_internvideo2_1b_patch14_224",
    "pretrain_internvideo2_6b_patch14_224",
]
PY
  cat > "$MODELS_INIT" <<'PY'
# ReVA temporal branch: skip CLIP imports that pull optional flash_attn stacks.
__all__ = []
PY
fi

IV_CKPT="$WORK/models/InternVideo2-stage2_1b-224p-f4.pt"
if [[ ! -f "$IV_CKPT" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "InternVideo2 checkpoint missing and HF_TOKEN unset." >&2
    echo "Accept https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4 then:" >&2
    echo "  export HF_TOKEN=hf_..." >&2
    echo "  huggingface-cli login --token \"\$HF_TOKEN\"" >&2
    exit 1
  fi
  "$PY" - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
work = Path("/home/ubuntu/reva/models")
path = hf_hub_download(
    repo_id="OpenGVLab/InternVideo2-Stage2_1B-224p-f4",
    filename="InternVideo2-stage2_1b-224p-f4.pt",
    local_dir=str(work),
)
print("internvideo2", path)
PY
fi

QWEN_MARKER="$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct"
if [[ ! -d "$QWEN_MARKER" ]]; then
  "$PY" - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download(
    "Qwen/Qwen3-VL-8B-Instruct",
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    cache_dir=os.environ["HF_HOME"] + "/hub",
)
PY
fi

echo "WORK=$WORK ready. Train with:"
echo "  cd $REPO && WORK=$WORK $PY code/train.py code/configs/dualstream_temporal_grounding_brev.yaml"
