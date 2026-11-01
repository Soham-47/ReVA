"""Evaluate score JSONL files by overall, task, category, and source accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
}


def accuracy(rows: list[dict]) -> float:
    return sum(row["ok"] for row in rows) / len(rows) if rows else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("split", choices=sorted(SPLITS))
    ap.add_argument("--dump", type=int, default=40)
    args = ap.parse_args()
    scores_path = C.work_path(args.scores)
    with C.work_path(SPLITS[args.split]).open() as f:
        gold = {row["qa_id"]: row for row in json.load(f)["QA"]}
    rows = []
    seen: set[str] = set()
    with scores_path.open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            score = json.loads(line)
            qa_id = score.get("qa_id")
            if qa_id not in gold or qa_id in seen:
                raise ValueError(f"unknown or duplicate qa_id at {scores_path}:{line_number}")
            if score.get("pred") not in C.LETTERS or len(score.get("p", [])) != 4:
                raise ValueError(f"invalid prediction record at {scores_path}:{line_number}")
            views = score.get("views") or []
            if not views:
                raise ValueError(f"score record has no views at {scores_path}:{line_number}")
            question = gold[qa_id]
            single = views[0]
            p = [float(value) for value in score["p"]]
            row = {
                "qa_id": qa_id,
                "task": question["task"],
                "category": question["category"],
                "source": question["dataset_name"],
                "gold": question["correct_answer"],
                "pred": score["pred"],
                "conf": max(p),
                "pred_single": C.LETTERS[max(range(4), key=lambda i: float(single["p"][i]))],
                "shown0": single.get("argmax_shown", ""),
                "question": question["question"],
                "options": json.dumps(question["options"], ensure_ascii=False),
                "video": question["video_path"],
            }
            row["ok"] = row["pred"] == row["gold"]
            row["ok_single"] = row["pred_single"] == row["gold"]
            rows.append(row)
            seen.add(qa_id)
    if seen != set(gold):
        raise ValueError(f"scores are incomplete: missing {len(set(gold) - seen)} records")

    print(f"== {args.split}: n={len(rows)} ACC(TTA)={accuracy(rows):.4f} ACC(single)={sum(r['ok_single'] for r in rows)/len(rows):.4f}")
    for field in ("category", "task", "source"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        for value, group in sorted(grouped.items(), key=lambda item: accuracy(item[1])):
            print(f"{field:8s} {value}: n={len(group)} acc={accuracy(group):.4f}")
    confusion = {gold_letter: {pred: 0 for pred in C.LETTERS} for gold_letter in C.LETTERS}
    for row in rows:
        confusion[row["gold"]][row["pred"]] += 1
    print("confusion rows=gold cols=pred:", confusion)
    print("single-view shown-letter histogram:", dict(Counter(row["shown0"] for row in rows)))

    stem = scores_path.with_suffix("")
    csv_path = stem.with_name(stem.name + "_eval.csv")
    columns = [
        "qa_id", "task", "category", "source", "gold", "pred", "conf", "pred_single", "shown0",
        "question", "options", "video", "ok", "ok_single",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    worst_tasks = sorted({row["task"] for row in rows}, key=lambda task: accuracy([r for r in rows if r["task"] == task]))[:3]
    errors_path = stem.with_name(stem.name + "_errors_worst3.csv")
    errors = [row for row in rows if not row["ok"] and row["task"] in worst_tasks]
    errors.sort(key=lambda row: row["conf"], reverse=True)
    with errors_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(errors[: args.dump * 3])
    summary = {
        "acc": accuracy(rows),
        "acc_single": sum(row["ok_single"] for row in rows) / len(rows),
        "per_task": {
            task: accuracy([row for row in rows if row["task"] == task])
            for task in sorted({row["task"] for row in rows})
        },
    }
    with stem.with_name(stem.name + "_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"wrote {csv_path}, {errors_path}, and summary JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
