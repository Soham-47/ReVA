#!/usr/bin/env python3
"""Paired promotion gates for ReVA score JSONL sets (baseline vs candidate).

Gate definitions (remediation plan §5.3 + subset S screening):
  G1: valQ overall — candidate correct >= baseline correct (no valQ regression).
  G2: valT overall — candidate correct >= baseline correct.
  G3: val overall — candidate correct >= baseline correct.
  G4: Subset S (Temporal Grounding, Change Detection, Trend and Pattern on valQ+valT)
      — paired net >= 0 and exact two-sided McNemar p <= 0.05 when net > 0, else net >= 0 only.
  G5: Saturated deductive cluster G_sat on valQ+valT — valQ >= 82/86 and valT >= 155/166 for candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C


def default_work() -> Path:
    import os

    value = os.environ.get("WORK")
    return Path(value).expanduser().resolve() if value else ROOT

SPLITS = {
    "valQ": "data/splits/valQ.json",
    "valT": "data/splits/valT.json",
    "val": "data/ReVA/val.json",
}

SUBSET_S_TASKS = frozenset({"Temporal Grounding", "Change Detection", "Trend and Pattern"})
G_SAT_TASKS = frozenset(
    {
        "General Understanding",
        "Consequence Reasoning",
        "Hypothetical Reasoning",
        "Causation Reasoning",
    }
)
G_SAT_FLOORS = {"valQ": (82, 86), "valT": (155, 166)}


def load_gold(split: str, work: Path) -> dict[str, dict]:
    path = work / SPLITS[split] if not Path(SPLITS[split]).is_absolute() else Path(SPLITS[split])
    if not path.is_file():
        path = C.work_path(SPLITS[split], str(work))
    with path.open() as handle:
        return {row["qa_id"]: row for row in json.load(handle)["QA"]}


def load_scores(path: Path) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qa_id = row["qa_id"]
            if qa_id in scores:
                raise ValueError(f"duplicate qa_id {qa_id} at {path}:{line_number}")
            scores[qa_id] = row
    return scores


def correct_total(scores: dict[str, dict], gold: dict[str, dict]) -> tuple[int, int]:
    ok = sum(1 for qa_id, question in gold.items() if scores[qa_id]["pred"] == question["correct_answer"])
    return ok, len(gold)


def paired_table(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    gold: dict[str, dict],
    *,
    task_filter: set[str] | None = None,
    category_filter: set[str] | None = None,
) -> dict[str, int]:
    fixed = broken = net = 0
    for qa_id, question in gold.items():
        if task_filter and question["task"] not in task_filter:
            continue
        if category_filter and question["category"] not in category_filter:
            continue
        base_ok = baseline[qa_id]["pred"] == question["correct_answer"]
        cand_ok = candidate[qa_id]["pred"] == question["correct_answer"]
        if not base_ok and cand_ok:
            fixed += 1
        elif base_ok and not cand_ok:
            broken += 1
        net += int(cand_ok) - int(base_ok)
    return {"fixed": fixed, "broken": broken, "net": net}


def mcnemar_exact_p(fixed: int, broken: int) -> float:
    """Two-sided exact binomial test on discordant pairs (fixed vs broken)."""

    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2**n))


def task_table(scores: dict[str, dict], gold: dict[str, dict]) -> list[tuple[str, int, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for qa_id, question in gold.items():
        grouped[question["task"]].append(qa_id)
    rows = []
    for task in sorted(grouped):
        ids = grouped[task]
        ok = sum(1 for qa_id in ids if scores[qa_id]["pred"] == gold[qa_id]["correct_answer"])
        rows.append((task, ok, len(ids)))
    return rows


def category_table(scores: dict[str, dict], gold: dict[str, dict]) -> list[tuple[str, int, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for qa_id, question in gold.items():
        grouped[question["category"]].append(qa_id)
    rows = []
    for category in sorted(grouped):
        ids = grouped[category]
        ok = sum(1 for qa_id in ids if scores[qa_id]["pred"] == gold[qa_id]["correct_answer"])
        rows.append((category, ok, len(ids)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", nargs="+", help="baseline score JSONL paths (valQ valT val order optional)")
    ap.add_argument("--candidate", nargs="+", required=True, help="candidate score JSONL paths")
    ap.add_argument("--work", default=None, help="WORK root (defaults to WORK env or repo root)")
    args = ap.parse_args()
    work = Path(args.work).expanduser().resolve() if args.work else default_work()

    if len(args.baseline) != len(args.candidate):
        ap.error("baseline and candidate must have the same number of score files")
    if len(args.baseline) not in {1, 3}:
        ap.error("provide 1 combined file or 3 files in valQ valT val order")

    split_names = ["valQ", "valT", "val"] if len(args.baseline) == 3 else ["val"]
    baseline_by_split = {name: load_scores(Path(path)) for name, path in zip(split_names, args.baseline)}
    candidate_by_split = {name: load_scores(Path(path)) for name, path in zip(split_names, args.candidate)}

    baseline_correct: dict[str, tuple[int, int]] = {}
    candidate_correct: dict[str, tuple[int, int]] = {}

    for split in split_names:
        gold = load_gold(split, work)
        base = baseline_by_split[split]
        cand = candidate_by_split[split]
        if set(base) != set(gold) or set(cand) != set(gold):
            raise ValueError(f"incomplete scores for split {split}")
        b_ok, n = correct_total(base, gold)
        c_ok, _ = correct_total(cand, gold)
        baseline_correct[split] = (b_ok, n)
        candidate_correct[split] = (c_ok, n)
        print(f"== {split} baseline {b_ok}/{n} candidate {c_ok}/{n} net {c_ok - b_ok:+d}")
        for task, ok, total in task_table(cand, gold):
            base_ok = sum(1 for qa_id in gold if gold[qa_id]["task"] == task and base[qa_id]["pred"] == gold[qa_id]["correct_answer"])
            print(f"  task {task}: base {base_ok}/{total} cand {ok}/{total}")
        for category, ok, total in category_table(cand, gold):
            base_ok = sum(
                1
                for qa_id in gold
                if gold[qa_id]["category"] == category and base[qa_id]["pred"] == gold[qa_id]["correct_answer"]
            )
            print(f"  category {category}: base {base_ok}/{total} cand {ok}/{total}")

    subset_paired = {"fixed": 0, "broken": 0, "net": 0}
    for split in ("valQ", "valT"):
        if split not in split_names:
            continue
        gold = load_gold(split, work)
        part = paired_table(
            baseline_by_split[split],
            candidate_by_split[split],
            gold,
            task_filter=SUBSET_S_TASKS,
        )
        for key in subset_paired:
            subset_paired[key] += part[key]
    s_p = mcnemar_exact_p(subset_paired["fixed"], subset_paired["broken"])
    ratio = (
        f"{subset_paired['fixed']}:{subset_paired['broken']}"
        if subset_paired["broken"]
        else f"{subset_paired['fixed']}:0"
    )
    print(
        f"== subset S (TG+CD+TP on valQ+valT): paired fixed/broken/net="
        f"{subset_paired['fixed']}/{subset_paired['broken']}/{subset_paired['net']:+d} "
        f"ratio={ratio} mcnemar_p={s_p:.6f}"
    )

    g_sat_valq = (0, 0)
    g_sat_valt = (0, 0)
    if "valQ" in split_names:
        gold = load_gold("valQ", work)
        g_sat_valq = (
            sum(
                1
                for qa_id, q in gold.items()
                if q["task"] in G_SAT_TASKS
                and candidate_by_split["valQ"][qa_id]["pred"] == q["correct_answer"]
            ),
            sum(1 for q in gold.values() if q["task"] in G_SAT_TASKS),
        )
    if "valT" in split_names:
        gold = load_gold("valT", work)
        g_sat_valt = (
            sum(
                1
                for qa_id, q in gold.items()
                if q["task"] in G_SAT_TASKS
                and candidate_by_split["valT"][qa_id]["pred"] == q["correct_answer"]
            ),
            sum(1 for q in gold.values() if q["task"] in G_SAT_TASKS),
        )
    print(f"== G_sat candidate valQ {g_sat_valq[0]}/{g_sat_valq[1]} valT {g_sat_valt[0]}/{g_sat_valt[1]}")

    gates = {
        "G1": "PASS" if candidate_correct.get("valQ", (0, 0))[0] >= baseline_correct.get("valQ", (0, 0))[0] else "FAIL",
        "G2": "PASS" if candidate_correct.get("valT", (0, 0))[0] >= baseline_correct.get("valT", (0, 0))[0] else "FAIL",
        "G3": "PASS" if candidate_correct.get("val", (0, 0))[0] >= baseline_correct.get("val", (0, 0))[0] else "FAIL",
        "G4": (
            "FAIL"
            if subset_paired["net"] < 0
            else "PASS"
            if subset_paired["net"] == 0 or s_p <= 0.05
            else "FAIL"
        ),
        "G5": (
            "PASS"
            if g_sat_valq[0] >= G_SAT_FLOORS["valQ"][0] and g_sat_valt[0] >= G_SAT_FLOORS["valT"][0]
            else "FAIL"
        ),
    }
    print("== gates " + " ".join(f"{name}={status}" for name, status in gates.items()))
    return 0 if all(status == "PASS" for status in gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
