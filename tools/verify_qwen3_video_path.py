#!/usr/bin/env python3
"""Inspect Qwen3-VL native video timestamps, temporal patch size, and visual token counts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C


def visual_tokens(encoded: dict) -> int:
    grid = encoded["video_grid_thw"]
    if grid.ndim == 2:
        return int(grid.prod(dim=1).sum().item())
    return int(grid.prod().item())


def decode_prompt(processor, encoded) -> str:
    ids = encoded["input_ids"][0].tolist()
    return processor.tokenizer.decode(ids, skip_special_tokens=False)


def inspect_timestamps(processor, row, frame_budget: int, video_size: dict) -> dict:
    frames, meta = C.get_video(row["video_path"], 0.0, frame_budget=frame_budget)
    encoded, _ = C.encode(processor, row, frames, meta, C.LETTERS, video_size=video_size)
    text = decode_prompt(processor, encoded)
    seconds_tags = re.findall(r"<\d+(?:\.\d+)?\s*seconds?>", text, flags=re.IGNORECASE)
    cotr_like = re.findall(r"\[\d+(?:\.\d+)?s\]", text)
    return {
        "qa_id": row["qa_id"],
        "frames": len(meta.frames_indices),
        "seconds_tags": seconds_tags[:8],
        "cotr_brackets": cotr_like[:8],
        "visual_tokens": visual_tokens(encoded),
        "snippet": text[text.find("<|im_start|>user") : text.find("<|im_start|>user") + 800],
    }


def token_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "mean": sum(values) / len(values),
        "p95": float(ordered[p95_index]),
        "max": float(max(values)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import transformers

    print(f"transformers={transformers.__version__}", flush=True)

    processor = C.load_processor()
    model_id = getattr(processor, "model_info", None) or C.MODEL_ID
    temporal_patch = getattr(processor, "temporal_patch_size", None)
    if temporal_patch is None and hasattr(processor, "video_processor"):
        temporal_patch = getattr(processor.video_processor, "temporal_patch_size", None)
    print(f"temporal_patch_size={temporal_patch}", flush=True)

    fit_path = C.work_path("data/splits/fit_A.json")
    with fit_path.open() as handle:
        rows = json.load(handle)["QA"]
    rng = random.Random(args.seed)
    sample_rows = rows if len(rows) <= args.samples else rng.sample(rows, args.samples)

    probe = sample_rows[0]
    for budget in (16, 32):
        video_size = C.video_size_for(budget, C.DEFAULT_VIDEO_MAX_PIXELS)
        detail = inspect_timestamps(processor, probe, budget, video_size)
        print(f"== timestamp probe frame_budget={budget} qa={detail['qa_id']}", flush=True)
        print(f"   seconds_tags={detail['seconds_tags']}", flush=True)
        print(f"   cotr_brackets={detail['cotr_brackets']}", flush=True)

    configs = [
        ("16f_default", 16, None),
        ("32f_fullres_cap", 32, C.DEFAULT_VIDEO_MAX_PIXELS),
    ]
    report = {"transformers": transformers.__version__, "temporal_patch_size": temporal_patch, "arms": {}}
    for name, budget, pixels in configs:
        video_size = C.video_size_for(budget, pixels)
        counts = []
        for row in sample_rows:
            frames, meta = C.get_video(row["video_path"], 0.0, frame_budget=budget)
            encoded, _ = C.encode(processor, row, frames, meta, C.LETTERS, video_size=video_size)
            counts.append(visual_tokens(encoded))
        stats = token_stats(counts)
        report["arms"][name] = {"frame_budget": budget, "video_size": video_size, **stats}
        print(
            f"== visual tokens {name} n={len(counts)} mean={stats['mean']:.1f} "
            f"p95={stats['p95']:.0f} max={stats['max']:.0f}",
            flush=True,
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
