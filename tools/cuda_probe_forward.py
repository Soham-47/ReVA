#!/usr/bin/env python3
"""Fail-fast Qwen3-VL video forward probe for the HPC CUDA cuModuleLoadData bug.

DRAFT ONLY — do not sbatch until the user asks. Intended for a ~15 min interactive
or short job: load venv312, print versions, run ONE video forward with
CUDA_LAUNCH_BLOCKING=1 and CUDA_MODULE_LOADING=EAGER, write SMOKE_OK or exit 1.

This does not cancel Slurm job 432730 (reva_b2_h100, PENDING). Leave it alone
unless the user explicitly says to cancel.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def main() -> int:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    work = Path(os.environ.get("WORK", str(ROOT))).expanduser().resolve()
    out_dir = Path(os.environ.get("REVA_PROBE_OUT", str(work / "tmp_cuda_probe")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_path = out_dir / "SMOKE_OK"
    fail_path = out_dir / "SMOKE_FAIL.txt"
    if ok_path.exists():
        ok_path.unlink()
    if fail_path.exists():
        fail_path.unlink()

    print("=== ReVA CUDA probe (draft) ===", flush=True)
    print(f"WORK={work}", flush=True)
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}", flush=True)
    print(f"CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING')}", flush=True)

    import torch
    import transformers
    import peft

    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"transformers={transformers.__version__} peft={peft.__version__}", flush=True)
    print(
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}",
        flush=True,
    )
    if not torch.cuda.is_available():
        fail_path.write_text("torch.cuda.is_available() is False\n")
        return 1
    print(f"gpu0={torch.cuda.get_device_name(0)}", flush=True)

    import common as C

    qa_path = work / "data/splits/fit_A.json"
    if not qa_path.is_file():
        qa_path = work / "data/ReVA/val.json"
    with qa_path.open() as handle:
        rows = json.load(handle)["QA"]
    row = rows[0]
    print(f"probe_qa_id={row['qa_id']} video={row['video_path']}", flush=True)

    try:
        processor = C.load_processor()
        model = C.load_model(dtype=torch.bfloat16)
        model.eval()
        frames, meta = C.get_video(row["video_path"], 0.0, frame_budget=16)
        video_size = C.video_size_for(16, None) if hasattr(C, "video_size_for") else None
        encoded, _ = C.encode(processor, row, frames, meta, C.LETTERS, video_size=video_size)
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in encoded.items()}
        with torch.inference_mode():
            out = model(**batch)
        logits = out.logits
        print(f"forward_ok logits_shape={tuple(logits.shape)}", flush=True)
        ok_path.write_text(
            "SMOKE_OK\n"
            f"qa_id={row['qa_id']}\n"
            f"torch={torch.__version__}\n"
            f"cuda={torch.version.cuda}\n"
            f"gpu={torch.cuda.get_device_name(0)}\n"
        )
        print(f"wrote {ok_path}", flush=True)
        return 0
    except Exception:
        tb = traceback.format_exc()
        fail_path.write_text(tb)
        print(tb, file=sys.stderr, flush=True)
        print(f"wrote {fail_path}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
