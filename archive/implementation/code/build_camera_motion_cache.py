"""Precompute ORB+RANSAC camera-motion sentences for ReVA clips.

Saves a lookup table {video_path: "Camera motion: ..."} so both SFT training
(Track B) and evaluation (Track A) can reuse deterministic motion hints without
runtime overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "archive" / "implementation" / "code"))

import common as C
import visual_prompts as VP

SPLIT_FILES = {
    "fit_A": "data/splits/fit_A.json",
    "valQ": "data/splits/valQ.json",
    "valT": "data/splits/valT.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def load_split_qa(split: str, work) -> list[dict]:
    rel = SPLIT_FILES.get(split)
    if rel is None:
        raise ValueError(f"unknown split: {split}")
    path = C.work_path(rel, work)
    if not path.is_file():
        return []
    with path.open() as handle:
        payload = json.load(handle)
    return payload["QA"] if "QA" in payload else payload


def motion_hint_fixed_grid(video_path: str, work, *, frame_budget: int) -> str:
    """Deterministic endpoints grid at offset 0 (independent of train/score offsets)."""

    frames, meta = C.get_video(
        video_path,
        0.0,
        work=work,
        mode="endpoints",
        frame_budget=frame_budget,
    )
    hint = VP.camera_motion_hint(frames, meta)
    if hint in {VP.MOTION_UNAVAILABLE, VP.MOTION_UNCLEAR}:
        return VP.MOTION_UNCLEAR
    return hint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/camera_motion_hints.json", help="output cache path relative to WORK")
    ap.add_argument(
        "--splits",
        default="fit_A,valQ,valT,val,test",
        help="comma-separated splits to cover",
    )
    ap.add_argument(
        "--frame-budget",
        type=int,
        default=C.MAX_FRAMES,
        help="fixed endpoints grid size for hint extraction",
    )
    args = ap.parse_args()

    work = os.environ.get("WORK")
    out_path = C.work_path(args.out, work)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, str] = {}
    if out_path.is_file():
        try:
            with out_path.open() as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    unique_videos: set[str] = set()

    for name in split_names:
        for row in load_split_qa(name, work):
            if "video_path" in row:
                unique_videos.add(row["video_path"])

    print(f"Total unique videos across {split_names}: {len(unique_videos)}", flush=True)
    needed = [v for v in sorted(unique_videos) if v not in cache]
    print(f"Already cached: {len(cache)}, Remaining to compute: {len(needed)}", flush=True)

    t0 = time.time()
    for idx, vpath in enumerate(needed, 1):
        try:
            cache[vpath] = motion_hint_fixed_grid(vpath, work, frame_budget=args.frame_budget)
        except Exception as exc:
            print(f"[{idx}/{len(needed)}] Error processing {vpath}: {exc}", flush=True)
            cache[vpath] = VP.MOTION_UNCLEAR

        if idx % 50 == 0 or idx == len(needed):
            elapsed = time.time() - t0
            rate = idx / max(1.0, elapsed)
            print(f"[{idx}/{len(needed)}] cached ({rate:.2f} videos/s)... saving checkpoint", flush=True)
            with out_path.open("w") as f:
                json.dump(cache, f, indent=2, sort_keys=True)
                f.write("\n")

    with out_path.open("w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")

    unclear = sum(1 for value in cache.values() if value == VP.MOTION_UNCLEAR)
    fraction = unclear / max(len(cache), 1)
    print(f"Saved {len(cache)} camera motion hints to {out_path}", flush=True)
    print(f"unclear={unclear}/{len(cache)} ({fraction:.2%})", flush=True)
    examples = [value for value in cache.values() if value != VP.MOTION_UNCLEAR][:10]
    for index, sentence in enumerate(examples, 1):
        print(f"example_{index}: {sentence}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
