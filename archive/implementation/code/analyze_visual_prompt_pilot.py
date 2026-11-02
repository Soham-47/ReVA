"""Compare overlay-arm JSONL files against Stage A on a fixed ID cohort."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def _load_ids(path: Path, split: str) -> list[str]:
    payload = json.loads(path.read_text())
    section = payload["splits"][split]
    values = section["ids"] if isinstance(section, dict) else section
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} splits.{split} must be a non-empty id list")
    return [str(value) for value in values]


def _load_scores(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["qa_id"]] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valT")
    parser.add_argument("--ids", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "JSONL"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ids_path = C.work_path(args.ids)
    ids = _load_ids(ids_path, args.split)
    roles = json.loads(ids_path.read_text()).get("roles") or {}
    gold_rows = {
        row["qa_id"]: row
        for row in json.loads(C.work_path(f"data/splits/{args.split}.json").read_text())["QA"]
    }
    gold = {qa_id: gold_rows[qa_id] for qa_id in ids}
    baseline = _load_scores(C.work_path(args.baseline))
    missing_baseline = [qa_id for qa_id in ids if qa_id not in baseline]
    if missing_baseline:
        raise SystemExit(f"baseline missing {len(missing_baseline)} cohort ids")

    def summarize(scores: dict[str, dict]) -> dict:
        by_task: dict[str, list[int]] = defaultdict(list)
        by_role_task: dict[str, list[int]] = defaultdict(list)
        ok = 0
        for qa_id in ids:
            item = gold[qa_id]
            correct = int(scores[qa_id]["pred"] == item["correct_answer"])
            ok += correct
            by_task[item["task"]].append(correct)
            meta = roles.get(qa_id) or {}
            role_key = f"{item['task']}|{meta.get('review_role', 'unknown')}"
            by_role_task[role_key].append(correct)
        return {
            "n": len(ids),
            "correct": ok,
            "accuracy": ok / len(ids),
            "by_task": {
                task: {"n": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
                for task, values in sorted(by_task.items())
            },
            "by_role_task": {
                key: {"n": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
                for key, values in sorted(by_role_task.items())
            },
        }

    report = {
        "split": args.split,
        "n": len(ids),
        "baseline": summarize(baseline),
        "arms": {},
        "paired_vs_baseline": {},
    }
    for name, path in args.arm:
        scores = _load_scores(C.work_path(path))
        missing = [qa_id for qa_id in ids if qa_id not in scores]
        if missing:
            raise SystemExit(f"{name} missing {len(missing)} cohort ids")
        summary = summarize(scores)
        report["arms"][name] = summary
        deltas = []
        for qa_id in ids:
            base_ok = int(baseline[qa_id]["pred"] == gold[qa_id]["correct_answer"])
            arm_ok = int(scores[qa_id]["pred"] == gold[qa_id]["correct_answer"])
            deltas.append(arm_ok - base_ok)
        tg = [qa_id for qa_id in ids if gold[qa_id]["task"] == "Temporal Grounding"]
        geo = [qa_id for qa_id in ids if gold[qa_id]["task"] == "Geometric Relation"]
        spatial = [
            qa_id
            for qa_id in ids
            if gold[qa_id]["task"]
            in {"Geometric Relation", "Structural Layout", "Perspective and Viewpoint"}
        ]
        def slice_acc(score_map, subset):
            return sum(score_map[qa_id]["pred"] == gold[qa_id]["correct_answer"] for qa_id in subset) / len(subset)
        def slice_correct(score_map, subset):
            return sum(score_map[qa_id]["pred"] == gold[qa_id]["correct_answer"] for qa_id in subset)
        spatial_base = slice_correct(baseline, spatial) if spatial else None
        spatial_arm = slice_correct(scores, spatial) if spatial else None
        tg_base = slice_correct(baseline, tg) if tg else None
        tg_arm = slice_correct(scores, tg) if tg else None
        report["paired_vs_baseline"][name] = {
            "net_delta": sum(deltas),
            "tg_baseline": slice_acc(baseline, tg) if tg else None,
            "tg_arm": slice_acc(scores, tg) if tg else None,
            "tg_correct_baseline": tg_base,
            "tg_correct_arm": tg_arm,
            "geometry_baseline": slice_acc(baseline, geo) if geo else None,
            "geometry_arm": slice_acc(scores, geo) if geo else None,
            "spatial_n": len(spatial) or None,
            "spatial_correct_baseline": spatial_base,
            "spatial_correct_arm": spatial_arm,
            "spatial_baseline": slice_acc(baseline, spatial) if spatial else None,
            "spatial_arm": slice_acc(scores, spatial) if spatial else None,
            "gate_tg_and_geometry": bool(
                tg
                and geo
                and slice_acc(scores, tg) > slice_acc(baseline, tg)
                and slice_acc(scores, geo) > slice_acc(baseline, geo)
            ),
            "gate_spatial_up_tg_not_down": bool(
                spatial
                and spatial_arm is not None
                and spatial_base is not None
                and spatial_arm > spatial_base
                and (not tg or (tg_arm is not None and tg_base is not None and tg_arm >= tg_base))
            ),
        }

    out = C.work_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
