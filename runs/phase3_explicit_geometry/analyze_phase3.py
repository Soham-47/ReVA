"""Audit aligned Phase 3 score/context artifacts without changing predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("valQ", "valT")
CONDITIONS = ("G0", "G1", "G2")


def _jsonl(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row.get("qa_id"), str) or row["qa_id"] in result:
                raise RuntimeError(f"duplicate or invalid qa_id in {path}")
            result[row["qa_id"]] = row
    return result


def _validate_score(rows: dict[str, dict[str, Any]], expected: set[str], path: Path) -> None:
    if set(rows) != expected:
        raise RuntimeError(f"{path} IDs do not match selected scope: {sorted(set(rows) ^ expected)[:5]}")
    for qa_id, row in rows.items():
        vectors = [row.get("p", [])] + [view.get("p", []) for view in row.get("views", [])]
        if len(row.get("views", [])) != 4 or any(len(vector) != 4 for vector in vectors):
            raise RuntimeError(f"{path} has incomplete vectors/views for {qa_id}")
        for vector in vectors:
            if any(not math.isfinite(float(value)) for value in vector) or abs(sum(map(float, vector)) - 1.0) > 2e-4:
                raise RuntimeError(f"{path} has invalid probability vector for {qa_id}")


def _transitions(base: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]], gold: dict[str, str]) -> dict[str, int]:
    counts = {"fixes": 0, "regressions": 0, "wrong_to_wrong": 0, "correct_to_correct": 0, "prediction_unchanged": 0, "prediction_changed": 0}
    for qa_id in base:
        base_correct = base[qa_id]["pred"] == gold[qa_id]
        other_correct = other[qa_id]["pred"] == gold[qa_id]
        if not base_correct and other_correct:
            counts["fixes"] += 1
        elif base_correct and not other_correct:
            counts["regressions"] += 1
        elif not base_correct and not other_correct:
            counts["wrong_to_wrong"] += 1
        else:
            counts["correct_to_correct"] += 1
        if base[qa_id]["pred"] == other[qa_id]["pred"]:
            counts["prediction_unchanged"] += 1
        else:
            counts["prediction_changed"] += 1
    return counts


def _accuracy(rows: dict[str, dict[str, Any]], gold: dict[str, str]) -> dict[str, Any]:
    correct = sum(rows[qa_id]["pred"] == answer for qa_id, answer in gold.items())
    return {"correct": correct, "total": len(gold), "accuracy": correct / len(gold) if gold else None}


def _context_audit(path: Path, expected: set[str]) -> dict[str, Any]:
    rows = _jsonl(path)
    if set(rows) != expected:
        raise RuntimeError(f"{path} IDs do not match selected scope")
    pairs = [pair for row in rows.values() for pair in row.get("audit", {}).get("pairs", [])]
    unsupported = sum(row.get("audit", {}).get("status") == "unsupported_subtype" for row in rows.values())
    reliable = sum(pair.get("status") == "reliable_2d" for pair in pairs)
    ambiguous = sum(pair.get("status") == "ambiguous" for pair in pairs)
    insufficient = sum(pair.get("status") == "insufficient" for pair in pairs)
    return {
        "question_count": len(rows),
        "pair_count": len(pairs),
        "reliable_2d_pairs": reliable,
        "ambiguous_pairs": ambiguous,
        "insufficient_pairs": insufficient,
        "unsupported_subtype_records": unsupported,
        "reliable_pair_coverage": reliable / len(pairs) if pairs else 0.0,
        "records_with_reliable_evidence": sum(any(pair.get("status") == "reliable_2d" for pair in row.get("audit", {}).get("pairs", [])) for row in rows.values()),
        "records_with_reliable_evidence_fraction": sum(any(pair.get("status") == "reliable_2d" for pair in row.get("audit", {}).get("pairs", [])) for row in rows.values()) / len(rows) if rows else 0.0,
        "evidence_correctness": "manual qualitative audit required; no gold spatial evidence labels are available",
    }


def analyze(*, scope_dir: Path, selection_path: Path, g0_path: Path | None = None) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text())
    g0 = _jsonl(g0_path or ROOT / "runs/phase3_explicit_geometry/g0_baseline.jsonl")
    results: dict[str, Any] = {"scope": scope_dir.name, "splits": {}, "qualitative_review": {"status": "required_for_changed_cases"}}
    score_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    gold_rows: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        selected = selection["selected_questions"][split]
        expected = {row["qa_id"] for row in selected}
        gold = {row["qa_id"]: row["gold"] for row in selected}
        baseline = {qa_id: g0[qa_id] for qa_id in expected}
        _validate_score(baseline, expected, g0_path or ROOT / "runs/phase3_explicit_geometry/g0_baseline.jsonl")
        g1_path = scope_dir / f"g1_{split}_scores.jsonl"
        g2_path = scope_dir / f"g2_{split}_scores.jsonl"
        g1 = _jsonl(g1_path)
        g2 = _jsonl(g2_path)
        score_rows[split] = {"G0": baseline, "G1": g1, "G2": g2}
        gold_rows[split] = gold
        _validate_score(g1, expected, g1_path)
        _validate_score(g2, expected, g2_path)
        results["splits"][split] = {
            "question_count": len(expected),
            "accuracy": {condition: _accuracy(rows, gold) for condition, rows in (("G0", baseline), ("G1", g1), ("G2", g2))},
            "G1_vs_G0": _transitions(baseline, g1, gold),
            "G2_vs_G1": _transitions(g1, g2, gold),
            "G2_vs_G0": _transitions(baseline, g2, gold),
            "evidence": {
                "G1": _context_audit(scope_dir / f"g1_{split}_contexts.jsonl", expected),
                "G2": _context_audit(scope_dir / f"g2_{split}_contexts.jsonl", expected),
            },
        }
    changed: list[dict[str, Any]] = []
    for split, report in results["splits"].items():
        for relation, pair in (("G1_vs_G0", ("G0", "G1")), ("G2_vs_G1", ("G1", "G2")), ("G2_vs_G0", ("G0", "G2"))):
            for qa_id in {row["qa_id"] for row in selection["selected_questions"][split]}:
                left = score_rows[split][pair[0]][qa_id]
                right = score_rows[split][pair[1]][qa_id]
                if left["pred"] != right["pred"] or (left["pred"] == gold_rows[split][qa_id] and right["pred"] != gold_rows[split][qa_id]):
                    changed.append({"split": split, "qa_id": qa_id, "comparison": relation, "left_prediction": left["pred"], "right_prediction": right["pred"], "manual_review": "required"})
    results["changed_cases"] = changed
    (scope_dir / "phase3_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Phase 3 explicit geometry pilot analysis",
        "",
        "This is an error-enriched pilot summary; it is not split-wide benchmark accuracy.",
        "",
    ]
    for split, report in results["splits"].items():
        lines.extend([f"## {split}", "", "| Condition | Correct | Total | Accuracy |", "|---|---:|---:|---:|"])
        for condition, accuracy in report["accuracy"].items():
            lines.append(f"| {condition} | {accuracy['correct']} | {accuracy['total']} | {accuracy['accuracy']:.3f} |")
        lines.extend(["", "Transition counts:"])
        for relation in ("G1_vs_G0", "G2_vs_G1", "G2_vs_G0"):
            lines.append(f"- {relation}: {report[relation]}")
        lines.extend(["", f"Evidence: {json.dumps(report['evidence'])}", ""])
    lines.extend(["## Qualitative review", "", f"Changed comparison cases requiring manual inspection: {len(changed)}.", "Evidence correctness remains a manual audit because the dataset has no gold box or mask annotations.", ""])
    (scope_dir / "phase3_report.md").write_text("\n".join(lines))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-dir", required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    analyze(scope_dir=ROOT / args.scope_dir, selection_path=ROOT / args.selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
