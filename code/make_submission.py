"""Validate test predictions and package predictions.json in a submission zip."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import zipfile
from pathlib import Path

import common as C


def read_scores(paths: list[str], ids: set[str]) -> dict[str, list[float]]:
    probabilities = collections.defaultdict(lambda: [0.0] * 4)
    total_weight = 0.0
    for spec in paths:
        if ":" in spec:
            path_text, weight_text = spec.rsplit(":", 1)
            try:
                weight = float(weight_text)
            except ValueError as exc:
                raise ValueError(f"invalid score weight in {spec!r}") from exc
        else:
            path_text, weight = spec, 1.0
        if weight <= 0 or not math.isfinite(weight):
            raise ValueError(f"score weight must be finite and positive: {spec!r}")
        seen: set[str] = set()
        with C.work_path(path_text).open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                qa_id = row.get("qa_id")
                probs = row.get("p")
                if qa_id not in ids or qa_id in seen or not isinstance(probs, list) or len(probs) != 4:
                    raise ValueError(f"invalid score row at {path_text}:{line_number}")
                if any(not math.isfinite(float(value)) or float(value) < 0 for value in probs):
                    raise ValueError(f"invalid probabilities at {path_text}:{line_number}")
                if not math.isclose(sum(float(value) for value in probs), 1.0, rel_tol=1e-4, abs_tol=1e-4):
                    raise ValueError(f"probabilities must sum to 1 at {path_text}:{line_number}")
                for index, value in enumerate(probs):
                    probabilities[qa_id][index] += weight * float(value)
                seen.add(qa_id)
        if seen != ids:
            raise ValueError(f"{path_text}: expected {len(ids)} ids, got {len(seen)}")
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("at least one score file is required")
    return {qa_id: [value / total_weight for value in values] for qa_id, values in probabilities.items()}


def write_zip(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("predictions.json", json.dumps(payload, ensure_ascii=False))
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["predictions.json"]:
            raise AssertionError(f"unexpected zip contents: {archive.namelist()}")
        json.loads(archive.read("predictions.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--scores", nargs="+", help="score.jsonl[:weight] files")
    source.add_argument("--all", choices=list(C.LETTERS), help="constant prediction for format probing")
    ap.add_argument("--out", required=True, help="output path without .zip")
    args = ap.parse_args()
    with C.work_path("data/ReVA/test.json").open() as f:
        test = json.load(f)
    rows = test.get("QA", [])
    ids = [row.get("qa_id") for row in rows]
    if len(ids) != 4000 or len(set(ids)) != len(ids):
        raise ValueError(f"test.json must contain 4,000 unique QA ids, got {len(ids)}")
    if args.all:
        predictions = {qa_id: args.all for qa_id in ids}
    else:
        probability_map = read_scores(args.scores, set(ids))
        predictions = {qa_id: C.LETTERS[max(range(4), key=lambda i: probability_map[qa_id][i])] for qa_id in ids}
    if set(predictions) != set(ids) or any(value not in C.LETTERS for value in predictions.values()):
        raise AssertionError("submission predictions failed schema validation")
    histogram = collections.Counter(predictions.values())
    print("letter histogram:", dict(histogram))
    if not args.all and max(histogram.values()) / len(ids) >= 0.35:
        raise ValueError("suspicious letter bias (>35%); inspect score files before submitting")

    primary = {
        "metadata": test.get("metadata", {}),
        "QA": [dict(row, correct_answer=predictions[row["qa_id"]]) for row in rows],
    }
    base = C.work_path(args.out)
    write_zip(base.with_suffix(".zip"), primary)
    # The Codabench schema is undocumented; keep two tiny alternatives for a format probe.
    write_zip(base.with_name(base.name + "_alt_list").with_suffix(".zip"), [{"qa_id": qa_id, "correct_answer": predictions[qa_id]} for qa_id in ids])
    write_zip(base.with_name(base.name + "_alt_dict").with_suffix(".zip"), predictions)
    print(f"wrote {base.with_suffix('.zip')} (plus _alt_list and _alt_dict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
