"""Held-out routing gate for one task specialist against an existing score file.

The gate does not train and does not rewrite either score file. A routed JSONL
is written only when the predeclared rule passes. Thresholds are a routing
policy, not a measured improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common as C
import specialist_data as S

MIN_NET = 2
MAX_REGRESSIONS = 1
_STAGE_B_FAMILIES = {"stageB", "ensemble_ab"}
# The Nov 5 A+B result is a directory. The HPC scripts are generators, not that result.
ENSEMBLE_ARTIFACT_DIR = "results/ensemble_ab_20261105"
ENSEMBLE_GENERATOR = "tools/ensemble_scores.py"
PAIRED_GATE_SCRIPT = "tools/paired_gate.py"
# Letter-logit files this repo's scorer writes. Stage A legacy copies use this shape.
SCORE_RECORD_FIELDS = ("qa_id", "p", "pred", "views")


def load_predictions(path: Path) -> dict[str, dict]:
    """Load letter predictions and preserve the original record when present."""

    text = path.read_text()
    records: list[dict] = []
    stripped = text.lstrip()
    if path.suffix == ".jsonl" or (stripped.startswith("{") and "\n{" in text):
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not a prediction object") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} is not a prediction object")
            records.append(item)
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            records.extend(payload)
        elif isinstance(payload, dict) and isinstance(payload.get("predictions"), dict):
            return {
                str(qa_id): {"qa_id": str(qa_id), "pred": _letter(pred)}
                for qa_id, pred in payload["predictions"].items()
            }
        elif isinstance(payload, dict) and payload and all(isinstance(value, str) for value in payload.values()):
            return {str(qa_id): {"qa_id": str(qa_id), "pred": _letter(pred)} for qa_id, pred in payload.items()}
        else:
            keys = sorted(payload)[:8] if isinstance(payload, dict) else [type(payload).__name__]
            raise ValueError(
                f"{path} is not a prediction record this router can read. "
                "Accepted shapes are a JSONL or list of objects with qa_id and pred, "
                "a qa_id-to-letter object, or {\"predictions\": {qa_id: letter}}. "
                f"A+B artifacts belong under {ENSEMBLE_ARTIFACT_DIR}/; "
                f"{ENSEMBLE_GENERATOR} is the HPC generator, not the artifact. keys={keys}"
            )
    predictions: dict[str, dict] = {}
    for item in records:
        if not isinstance(item, dict) or "qa_id" not in item or "pred" not in item:
            raise ValueError(f"{path} record is missing qa_id or pred")
        qa_id = str(item["qa_id"])
        if qa_id in predictions:
            raise ValueError(f"{path} repeats {qa_id}")
        stored = dict(item)
        stored["pred"] = _letter(item["pred"])
        predictions[qa_id] = stored
    if not predictions:
        raise ValueError(f"{path} has no predictions")
    return predictions


def _letter(value) -> str:
    if value not in C.LETTERS:
        raise ValueError(f"prediction must be one of A-D, got {value!r}")
    return value


def _training_ids(train_json: Path, tasks: tuple[str, ...]) -> set[str]:
    rows = S.filter_rows_by_tasks(S.load_qa_payload(train_json)["QA"], tasks)
    S.assert_train_only_rows(rows)
    return {row["qa_id"] for row in rows}


def evaluate_gate(
    gold_rows: list[dict],
    baseline: dict[str, dict],
    specialist: dict[str, dict],
    tasks: tuple[str, ...],
    training_ids: set[str],
    *,
    split_name: str,
    baseline_family: str,
    eligibility: dict,
) -> dict:
    if split_name == "test":
        raise ValueError("the routing gate does not read the hidden test split")
    if baseline_family in _STAGE_B_FAMILIES and split_name != "val":
        raise ValueError("Stage B-family comparisons require the official val split")
    if baseline_family in _STAGE_B_FAMILIES and eligibility.get("stageB_family_leaks_into_official_val"):
        raise ValueError("official val overlaps full_train, so an A+B comparison is not a holdout")
    if split_name in {"valQ", "valT"} and not eligibility.get("eligible_for_specialist_holdout"):
        raise ValueError("valQ/valT are not eligible until the fit_A disjointness audit passes")
    task_rows = [row for row in gold_rows if row.get("task") in tasks]
    if not task_rows:
        raise ValueError(f"{split_name} has no rows for tasks {list(tasks)}")
    gold_ids = {row["qa_id"] for row in gold_rows}
    leaked = sorted(gold_ids & training_ids)
    if leaked:
        raise ValueError(f"{split_name} contains {len(leaked)} qa ids from the specialist training tasks")
    missing_specialist = [row["qa_id"] for row in task_rows if row["qa_id"] not in specialist]
    missing_baseline = [row["qa_id"] for row in task_rows if row["qa_id"] not in baseline]
    if missing_specialist or missing_baseline:
        raise ValueError(
            "task slice is incomplete: "
            f"specialist_missing={len(missing_specialist)} baseline_missing={len(missing_baseline)}"
        )
    fixes: list[str] = []
    regressions: list[str] = []
    baseline_correct = 0
    specialist_correct = 0
    for row in task_rows:
        gold = row["correct_answer"]
        base_ok = baseline[row["qa_id"]]["pred"] == gold
        spec_ok = specialist[row["qa_id"]]["pred"] == gold
        baseline_correct += int(base_ok)
        specialist_correct += int(spec_ok)
        if spec_ok and not base_ok:
            fixes.append(row["qa_id"])
        elif base_ok and not spec_ok:
            regressions.append(row["qa_id"])
    net = len(fixes) - len(regressions)
    passed = net >= MIN_NET and len(regressions) <= MAX_REGRESSIONS
    return {
        "split": split_name,
        "tasks": list(tasks),
        "n": len(task_rows),
        "baseline_correct": baseline_correct,
        "specialist_correct": specialist_correct,
        "fixes": len(fixes),
        "regressions": len(regressions),
        "net": net,
        "fix_ids": fixes,
        "regression_ids": regressions,
        "threshold": {"min_net": MIN_NET, "max_regressions": MAX_REGRESSIONS},
        "passed": passed,
        "baseline_family": baseline_family,
        "video_held_out": bool(eligibility.get("valT_video_held_out")) if split_name == "valT" else None,
        "note": "Routing threshold only. This report is not a claim that the specialist improved.",
    }


def write_routed(path: Path, gold_rows: list[dict], baseline: dict[str, dict], specialist: dict[str, dict], tasks: tuple[str, ...]) -> None:
    """Write a new JSONL. Target tasks come from the specialist; other tasks stay on the baseline."""

    lines: list[str] = []
    for row in gold_rows:
        source = specialist if row.get("task") in tasks else baseline
        if row["qa_id"] not in source:
            raise ValueError(f"cannot route {row['qa_id']}: selected arm has no prediction")
        record = dict(source[row["qa_id"]])
        record["route"] = "specialist" if row.get("task") in tasks else "baseline"
        record["task"] = row.get("task")
        lines.append(json.dumps(record, separators=(",", ":"), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a specialist with a baseline and route only if the gate passes")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--split", required=True, choices=("valQ", "valT", "val"))
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--baseline-family", required=True, choices=("stageA", "stageB", "ensemble_ab"))
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--routed-out")
    args = parser.parse_args(argv)
    tasks = tuple(args.tasks)
    eligibility = json.loads(Path(args.eligibility).read_text())
    gold_rows = S.load_qa_payload(Path(args.gold))["QA"]
    report = evaluate_gate(
        gold_rows,
        load_predictions(Path(args.baseline)),
        load_predictions(Path(args.specialist)),
        tasks,
        _training_ids(Path(args.train_json), tasks),
        split_name=args.split,
        baseline_family=args.baseline_family,
        eligibility=eligibility,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps({key: report[key] for key in ("split", "n", "baseline_correct", "specialist_correct", "fixes", "regressions", "passed")}, sort_keys=True))
    if report["passed"]:
        if not args.routed_out:
            raise SystemExit("gate passed but --routed-out was not set; refusing to leave routing implicit")
        write_routed(
            Path(args.routed_out),
            gold_rows,
            load_predictions(Path(args.baseline)),
            load_predictions(Path(args.specialist)),
            tasks,
        )
        print(f"wrote {args.routed_out}")
    elif args.routed_out and Path(args.routed_out).exists():
        raise SystemExit(f"gate failed and refusing to overwrite existing routed file {args.routed_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
