"""Create deterministic video-grouped and question-grouped holdouts from train only."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path


def write_split(path: Path, rows: list[dict], name: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({"metadata": {"split": name, "seed": seed}, "QA": rows}, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("WORK"), help="runtime tree (defaults to $WORK)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-fraction", type=float, default=0.06)
    ap.add_argument("--question-fraction", type=float, default=0.04)
    args = ap.parse_args()
    if not args.work:
        ap.error("--work or WORK is required")
    if not (0 < args.video_fraction < 1 and 0 < args.question_fraction < 1):
        ap.error("fractions must be in (0, 1)")
    work = Path(args.work).expanduser().resolve()
    with (work / "data" / "ReVA" / "train.json").open() as f:
        train = json.load(f)["QA"]
    if not train or any(not row["qa_id"].startswith("train_") for row in train):
        raise ValueError("split source must be train.json with train_* QA ids")
    if len({row["qa_id"] for row in train}) != len(train):
        raise ValueError("train QA ids are not unique")

    rng = random.Random(args.seed)
    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    for row in train:
        by_video[row["video_path"]].append(row)
    strata: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for video, rows in by_video.items():
        dominant_task = collections.Counter(row["task"] for row in rows).most_common(1)[0][0]
        strata[(rows[0]["dataset_name"], dominant_task)].append(video)
    val_t_videos: set[str] = set()
    for videos in strata.values():
        rng.shuffle(videos)
        val_t_videos.update(videos[: max(1, round(args.video_fraction * len(videos)))])
    val_t = [row for row in train if row["video_path"] in val_t_videos]
    rest = [row for row in train if row["video_path"] not in val_t_videos]

    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rest:
        groups[(row["video_path"], row["question"])].append(row)
    question_strata: dict[tuple[str, str], list[tuple[str, str]]] = collections.defaultdict(list)
    for key, rows in groups.items():
        question_strata[(rows[0]["task"], rows[0]["dataset_name"])].append(key)
    val_q_keys: set[tuple[str, str]] = set()
    for keys in question_strata.values():
        rng.shuffle(keys)
        val_q_keys.update(keys[: max(1, round(args.question_fraction * len(keys)))])
    val_q = [row for key in val_q_keys for row in groups[key]]
    fit_a = [row for key, rows in groups.items() if key not in val_q_keys for row in rows]

    def ids(rows: list[dict]) -> set[str]:
        return {row["qa_id"] for row in rows}

    if ids(fit_a) & (ids(val_t) | ids(val_q)):
        raise AssertionError("holdout QA ids overlap fit_A")
    if {row["video_path"] for row in fit_a} & val_t_videos:
        raise AssertionError("fit_A contains a val-T video")
    if {(row["video_path"], row["question"]) for row in fit_a} & val_q_keys:
        raise AssertionError("fit_A contains a val-Q question group")
    if len(fit_a) + len(val_t) + len(val_q) != len(train):
        raise AssertionError("split sizes do not add up")

    out = work / "data" / "splits"
    for name, rows in (("fit_A", fit_a), ("valT", val_t), ("valQ", val_q), ("full_train", train)):
        write_split(out / f"{name}.json", rows, name, args.seed)
        print(name, len(rows), "questions", len({row["video_path"] for row in rows}), "videos")
    with (work / "data" / "ReVA" / "test.json").open() as f:
        test = json.load(f)["QA"]
    fit_videos = {row["video_path"] for row in fit_a}
    print("test videos seen in fit_A:", sum(row["video_path"] in fit_videos for row in test), "/", len(test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
