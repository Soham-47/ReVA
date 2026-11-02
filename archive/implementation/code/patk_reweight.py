"""PATK-style TTA view reweighting from Stage A JSONL (CPU-only, no rescore)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import reze as R  # noqa: E402


def _load_ids(path: Path, split: str) -> list[str]:
    payload = json.loads(path.read_text())
    section = payload["splits"][split]
    values = section["ids"] if isinstance(section, dict) else section
    return [str(value) for value in values]


def _option_anchors(row: dict, duration: float) -> list[float]:
    duration_q = R.is_duration_question(row["question"])
    anchors: list[float] = []
    for letter in C.LETTERS:
        parsed = R.parse_option(row["options"][letter], duration=duration, duration_question=duration_q)
        kind = parsed["kind"]
        if kind == "point":
            anchors.append(float(parsed["time"]))
        elif kind == "range":
            anchors.extend([float(parsed["start"]), float(parsed["end"])])
        elif kind == "duration":
            anchors.append(float(parsed["seconds"]))
    return anchors or [0.0, duration * 0.5, duration]


def _view_weight(row: dict, offset: float, frame_budget: int, sampling: str) -> float:
    frames, meta = C.get_video(row["video_path"], offset, mode=sampling, frame_budget=frame_budget)
    times = R.frame_times(meta)
    duration = R.clip_duration(meta)
    anchors = _option_anchors(row, duration)
    if not times:
        return 1.0
    best = min(min(abs(time_s - anchor) for anchor in anchors) for time_s in times)
    return 1.0 / (1.0 + best)


def reweight_row(row: dict, score_row: dict, *, frame_budget: int, sampling: str) -> dict:
    views = score_row["views"]
    weights = []
    probs = []
    for view in views:
        weight = _view_weight(row, float(view["offset"]), frame_budget, sampling)
        weights.append(weight)
        probs.append(np.asarray(view["p"], dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    stacked = np.stack(probs, axis=0)
    mean = (w[:, None] * stacked).sum(axis=0)
    mean = mean / mean.sum()
    out = dict(score_row)
    out["p"] = [float(value) for value in mean]
    out["pred"] = C.LETTERS[int(np.argmax(mean))]
    out["patk_weights"] = [float(value) for value in w]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("valQ", "valT", "val"))
    parser.add_argument("--baseline", required=True, help="Stage A score JSONL with views")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids", help="optional cohort JSON (splits.<split>.ids)")
    parser.add_argument("--task", default="Temporal Grounding")
    parser.add_argument("--frame-budget", type=int, default=C.MAX_FRAMES)
    parser.add_argument("--frame-sampling", default=C.DEFAULT_FRAME_SAMPLING)
    args = parser.parse_args()

    split_path = C.work_path(f"data/splits/{args.split}.json" if args.split != "val" else "data/ReVA/val.json")
    gold_rows = {row["qa_id"]: row for row in json.loads(split_path.read_text())["QA"]}
    baseline_path = C.work_path(args.baseline)
    scores = {}
    with baseline_path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                scores[row["qa_id"]] = row

    if args.ids:
        qa_ids = _load_ids(C.work_path(args.ids), args.split)
    else:
        qa_ids = [
            qa_id
            for qa_id, row in gold_rows.items()
            if row.get("task") == args.task and qa_id in scores
        ]

    out_path = C.work_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with out_path.open("w") as stream:
        for qa_id in qa_ids:
            gold = gold_rows[qa_id]
            if "views" not in scores[qa_id]:
                raise SystemExit(f"baseline row {qa_id} missing views (need full Stage A JSONL)")
            rew = reweight_row(
                gold,
                scores[qa_id],
                frame_budget=args.frame_budget,
                sampling=args.frame_sampling,
            )
            correct += int(rew["pred"] == gold["correct_answer"])
            stream.write(json.dumps(rew, separators=(",", ":")) + "\n")

    report = {
        "split": args.split,
        "task": args.task,
        "n": len(qa_ids),
        "correct": correct,
        "accuracy": correct / len(qa_ids) if qa_ids else math.nan,
        "baseline": str(baseline_path),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
