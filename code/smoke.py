"""Run a small GPU forward/backward check before submitting long Slurm jobs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from train import make_lora


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("smoke requires a CUDA GPU")
    with (Path(__file__).resolve().parent / "configs" / "stageA.yaml").open() as f:
        config = yaml.safe_load(f)
    processor = C.load_processor()
    tokenizer = processor.tokenizer
    ltr_ids = C.letter_ids(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with C.work_path("data/splits/fit_A.json").open() as f:
        rows = json.load(f)["QA"]
    picks = []
    for source in ("VisDrone", "Hawk_UAV", "ERA", "UAVDT"):
        picks.append(next(row for row in rows if row["dataset_name"].startswith(source)))
    manifest = C.load_manifest()
    worst = max(
        rows,
        key=lambda row: (
            manifest[row["video_path"]]["width"] * manifest[row["video_path"]]["height"],
            manifest[row["video_path"]]["n_frames"],
        ),
    )
    if worst["qa_id"] not in {row["qa_id"] for row in picks}:
        picks.append(worst)
        print("including worst native visual sample:", worst["qa_id"], worst["video_path"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    adapted = make_lora(model, config)
    adapted.enable_input_require_grads()
    if config["train"].get("gradient_checkpointing", True):
        adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapted = adapted.cuda()
    adapted.train()
    for row in picks:
        frames, meta = C.get_video(row["video_path"], 0.0)
        encoded, _ = C.encode(processor, row, frames, meta, ["B", "D", "A", "C"], row["correct_answer"], im_end_id, ltr_ids)
        decoded = tokenizer.decode(encoded["input_ids"][0])
        if "seconds>" not in decoded or "<|im_start|>assistant\n" not in decoded:
            raise AssertionError("prompt is missing Qwen timestamps or assistant generation tag")
        print(row["dataset_name"], frames.shape, encoded["video_grid_thw"].tolist(), encoded["input_ids"].shape[1])
        inputs = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in encoded.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        output = adapted(**inputs)
        if not torch.isfinite(output.loss):
            raise AssertionError("smoke loss is not finite")
        output.loss.backward()
        torch.cuda.synchronize()
        print(f"loss={output.loss.item():.3f} seconds={time.monotonic() - started:.2f} peak_gb={torch.cuda.max_memory_allocated()/2**30:.1f}")
        adapted.zero_grad(set_to_none=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
