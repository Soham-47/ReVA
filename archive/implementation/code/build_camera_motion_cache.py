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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/camera_motion_hints.json", help="output cache path relative to WORK")
    ap.add_argument("--splits", default="fit_A,valQ,valT", help="comma-separated splits to cover")
    args = ap.parse_args()

    work = os.environ.get("WORK")
    manifest = C.load_manifest(work)
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

    for s in split_names:
        split_file = C.work_path(f"data/splits/{s}.json", work)
        if not split_file.is_file():
            continue
        with split_file.open() as sf:
            d = json.load(sf)
        qa = d["QA"] if "QA" in d else d
        for r in qa:
            if "video_path" in r:
                unique_videos.add(r["video_path"])

    print(f"Total unique videos across {split_names}: {len(unique_videos)}", flush=True)
    needed = [v for v in sorted(unique_videos) if v not in cache]
    print(f"Already cached: {len(cache)}, Remaining to compute: {len(needed)}", flush=True)

    t0 = time.time()
    for idx, vpath in enumerate(needed, 1):
        try:
            frames, meta = C.get_video(vpath, 0.0, work=work)
            hint = VP.camera_motion_hint(frames, meta)
            cache[vpath] = hint
        except Exception as exc:
            print(f"[{idx}/{len(needed)}] Error processing {vpath}: {exc}", flush=True)
            cache[vpath] = VP.MOTION_UNAVAILABLE

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

    print(f"Saved {len(cache)} camera motion hints to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
