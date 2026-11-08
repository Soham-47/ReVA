#!/usr/bin/env python3
"""Fail-fast Qwen3-VL-8B (+LoRA) video forward probe for the venv312 cuModuleLoadData bug.

Mirrors code/score.py exactly: C.load_processor -> Qwen3VLForConditionalGeneration.from_pretrained
-> C.load_peft_for_inference -> C.get_video -> C.encode -> model(**inputs).logits[0,-1,ltr_ids].
Writes SMOKE_OK or SMOKE_FAIL.txt in $REVA_PROBE_OUT and prints SMOKE_OK / SMOKE_FAIL.
`--preflight` checks imports/attributes/signatures/data paths without a GPU.
Never touches Slurm job 432730.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

REQUIRED_C = [
    "MODEL_ID", "MODEL_REV", "LETTERS", "load_processor", "letter_ids", "configure_attention_backend",
    "load_peft_for_inference", "get_video", "encode", "video_size_for", "work_path",
]
DEFAULT_ADAPTERS = ["runs/stageB_q3vl8b_r16_h100/final_adapter", "runs/stageA_q3vl8b_r16/final_adapter"]


T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def pick_rows(work: Path):
    for rel in ("data/splits/fit_A.json", "data/ReVA/val.json"):
        p = work / rel
        if p.is_file():
            with p.open() as h:
                data = json.load(h)
            rows = data["QA"] if isinstance(data, dict) else data
            return p, rows
    raise FileNotFoundError("no fit_A.json or val.json found")


def pick_adapter(work: Path):
    env = os.environ.get("REVA_PROBE_ADAPTER")
    cands = [env] if env else DEFAULT_ADAPTERS
    for c in cands:
        if c and c != "none" and (work / c / "adapter_config.json").is_file():
            return work / c
    return None


def preflight(work: Path) -> list[str]:
    errs = []
    import common as C
    import transformers, peft  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    for name in REQUIRED_C:
        if not hasattr(C, name):
            errs.append(f"common.{name} missing")
    if not errs:
        sig = inspect.signature(C.get_video).parameters
        if "frame_budget" not in sig:
            errs.append("get_video lacks frame_budget")
        if "video_size" not in inspect.signature(C.encode).parameters:
            errs.append("encode lacks video_size")
        if "merge" not in inspect.signature(C.load_peft_for_inference).parameters:
            errs.append("load_peft_for_inference lacks merge")
    try:
        qa, rows = pick_rows(work)
        row = rows[0]
        for k in ("qa_id", "video_path"):
            if k not in row:
                errs.append(f"row missing {k}")
        if not errs:
            if row["video_path"] not in C.load_manifest(work):
                errs.append(f"{row['video_path']} not in manifest")
            print(f"preflight qa={qa} qa_id={row['qa_id']}", flush=True)
    except Exception as e:
        errs.append(f"data: {e!r}")
    ad = pick_adapter(work)
    print(f"preflight adapter={ad}", flush=True)
    if os.environ.get("REVA_PROBE_ADAPTER") not in (None, "none") and ad is None:
        errs.append("REVA_PROBE_ADAPTER set but adapter_config.json not found")
    return errs


def main() -> int:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work = Path(os.environ.get("WORK", str(ROOT))).expanduser().resolve()

    if "--preflight" in sys.argv:
        try:
            errs = preflight(work)
        except Exception:
            errs = [traceback.format_exc()]
        if errs:
            print("PREFLIGHT_FAIL\n" + "\n".join(errs), flush=True)
            return 1
        print("PREFLIGHT_OK", flush=True)
        return 0

    out_dir = Path(os.environ.get("REVA_PROBE_OUT", str(work / "tmp_cuda_probe")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_path, fail_path = out_dir / "SMOKE_OK", out_dir / "SMOKE_FAIL.txt"
    for p in (ok_path, fail_path):
        if p.exists():
            p.unlink()

    print("=== ReVA CUDA probe ===", flush=True)
    print(f"WORK={work} CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']} "
          f"CUDA_MODULE_LOADING={os.environ['CUDA_MODULE_LOADING']}", flush=True)
    stage = "preflight"
    try:
        errs = preflight(work)
        if errs:
            raise RuntimeError("preflight: " + "; ".join(errs))
        import torch, transformers, peft
        from transformers import Qwen3VLForConditionalGeneration
        import common as C
        print(f"python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} "
              f"transformers={transformers.__version__} peft={peft.__version__}", flush=True)
        stage = "cuda_init"
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        print(f"gpu0={torch.cuda.get_device_name(0)}", flush=True)
        device = torch.device("cuda")

        _, rows = pick_rows(work)
        row = rows[0]
        print(f"probe_qa_id={row['qa_id']} video={row['video_path']}", flush=True)
        stage = "load_processor"
        processor = C.load_processor()
        ltr_ids = C.letter_ids(processor.tokenizer)
        stage = "load_model"
        C.configure_attention_backend("sdpa")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            C.MODEL_ID, revision=C.MODEL_REV, dtype=torch.bfloat16, attn_implementation="sdpa")
        log("weights loaded on cpu")
        stage = "to_cuda"
        model.to(device).eval()
        torch.cuda.synchronize()
        log(f"model on cuda mem={torch.cuda.memory_allocated()/2**30:.1f}GiB")
        adapter = pick_adapter(work)
        if adapter is not None:
            stage = "load_lora"
            model = C.load_peft_for_inference(model, adapter, merge=True)
            model.to(device).eval()
            torch.cuda.synchronize()
        log(f"adapter={adapter} merged")
        stage = "video+encode"
        frames, meta = C.get_video(row["video_path"], 0.0, frame_budget=16)
        encoded, perm = C.encode(processor, row, frames, meta, list(C.LETTERS),
                                 video_size=C.video_size_for(16, None))
        log("video+encode done")
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in encoded.items()}
        stage = "forward"
        with torch.inference_mode():
            logits = model(**batch).logits[0, -1, ltr_ids].float()
            probs = torch.softmax(logits, -1).cpu().tolist()
        torch.cuda.synchronize()
        log("forward done")
        print(f"forward_ok probs={probs}", flush=True)
        ok_path.write_text(f"SMOKE_OK\nqa_id={row['qa_id']}\nadapter={adapter}\ntorch={torch.__version__}\n"
                           f"cuda={torch.version.cuda}\ngpu={torch.cuda.get_device_name(0)}\n")
        print("SMOKE_OK", flush=True)
        return 0
    except BaseException:
        tb = f"stage={stage}\n" + traceback.format_exc()
        fail_path.write_text(tb)
        print(tb, file=sys.stderr, flush=True)
        print(f"SMOKE_FAIL stage={stage}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
