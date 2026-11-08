#!/usr/bin/env python3
"""Print five training prompts for Round-1 config verification."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C
import train as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    processor = C.load_processor()
    tokenizer = processor.tokenizer
    ltr_ids = C.letter_ids(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    dataset = T.ReVATrain(
        C.work_path(config["train_json"]),
        processor,
        config,
        ltr_ids,
        im_end_id,
    )
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.samples, len(dataset)))

    motion_path = config.get("motion_hints")
    hints = None
    if motion_path:
        with C.work_path(motion_path).open() as handle:
            hints = json.load(handle)

    video_cfg = config.get("video", {})
    frame_budget = int(video_cfg.get("frame_budget", config.get("frame_budget", C.MAX_FRAMES)))

    for index in indices:
        row = dataset.rows[index]
        rng_i = random.Random(int(config["seed"]) + index)
        permutation = list(C.LETTERS)
        rng_i.shuffle(permutation)
        offset = rng_i.random()
        frames, meta = C.get_video(row["video_path"], offset, frame_budget=dataset.frame_budget)
        motion_context = C.motion_hint_text(dataset.motion_hints, row["video_path"])
        encoded, _ = C.encode(
            processor,
            row,
            frames,
            meta,
            permutation,
            enable_thinking=T.chat_template_enable_thinking(config),
            temporal_context=motion_context,
            video_size=dataset.video_size,
        )
        text = tokenizer.decode(encoded["input_ids"][0].tolist(), skip_special_tokens=False)
        seconds_tags = re.findall(r"<\d+(?:\.\d+)?\s*seconds?>", text, flags=re.IGNORECASE)
        hint_line = hints.get(row["video_path"]) if hints else None
        print(f"qa_id={row['qa_id']} video={row['video_path']}")
        print(f"  frames={len(meta.frames_indices)} seconds_tags={len(seconds_tags)} first_tags={seconds_tags[:4]}")
        if hint_line:
            in_prompt = hint_line in text
            print(f"  motion_hint_in_prompt={in_prompt} hint={hint_line[:120]}")
        print(f"  prompt_snippet={text[text.find('user'):text.find('user')+500]!r}")
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
