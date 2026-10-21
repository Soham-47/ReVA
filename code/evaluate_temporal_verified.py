"""Report paired T0/T1/T1R metrics for the verified-window pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from score_temporal_verified import _annotation_digest, _load_records, _question_rows, read_jsonl


def _accuracy(predictions: dict[str, str], gold: dict[str, str], ids: list[str]) -> dict:
    correct = sum(predictions[qa_id] == gold[qa_id] for qa_id in ids)
    return {"correct": correct, "total": len(ids), "accuracy": correct / len(ids) if ids else 0.0}


def _pair(old: dict, new: dict, gold: dict[str, str], ids: list[str], old_arm: str, new_arm: str) -> dict:
    counts = Counter()
    details = []
    for qa_id in ids:
        old_pred, new_pred = old[qa_id]["pred"], new[qa_id]["pred"]
        old_correct, new_correct = old_pred == gold[qa_id], new_pred == gold[qa_id]
        if old_pred == new_pred:
            label = "unchanged_prediction"
        elif not old_correct and new_correct:
            label = "fixed"
        elif old_correct and not new_correct:
            label = "regression"
        else:
            label = "wrong_to_wrong"
        counts[label] += 1
        if label != "unchanged_prediction":
            details.append(
                {
                    "qa_id": qa_id,
                    "old": old_pred,
                    "new": new_pred,
                    "gold": gold[qa_id],
                    "change": label,
                    "probabilities": {old_arm: old[qa_id]["p"], new_arm: new[qa_id]["p"]},
                    "selection_evidence": {
                        old_arm: old[qa_id].get("temporal_selection_by_offset", {}),
                        new_arm: new[qa_id].get("temporal_selection_by_offset", {}),
                    },
                }
            )
    return {
        "n": len(ids),
        "fixed": counts["fixed"],
        "regressions": counts["regression"],
        "wrong_to_wrong": counts["wrong_to_wrong"],
        "unchanged_predictions": counts["unchanged_prediction"],
        "net_correct_change": counts["fixed"] - counts["regression"],
        "changed_answers": details,
    }


def _fallback_counts(rows: dict[str, dict], ids: list[str]) -> dict:
    counts = Counter()
    for qa_id in ids:
        for offset, selection in rows[qa_id].get("temporal_selection_by_offset", {}).items():
            kind = "fallback" if selection.get("fallback") else "reviewed"
            counts[f"{offset}:{kind}"] += 1
            counts[f"{kind}_offsets"] += 1
    return {"offsets": sum(counts[key] for key in ("fallback_offsets", "reviewed_offsets")), **counts}


def _annotation_evidence(records: list[dict]) -> dict:
    statuses = Counter(f"{r['boundary_status']}/{r['identity_status']}" for r in records)
    censoring = Counter(r.get("onset_censoring", "missing") for r in records)
    return {
        "records": len(records),
        "boundary_reviewed": sum(r["boundary_status"] == "reviewed" for r in records),
        "boundary_blank_or_unreviewed": sum(r["boundary_status"] != "reviewed" for r in records),
        "identity_confirmed": sum(r["identity_status"] == "confirmed" for r in records),
        "identity_unconfirmed": sum(r["identity_status"] != "confirmed" for r in records),
        "status_pairs": dict(sorted(statuses.items())),
        "onset_censoring": dict(sorted(censoring.items())),
        "review_text_retained_for_audit_only": True,
    }


def _bracket_coverage(rows: dict[str, dict], ids: list[str]) -> dict:
    counts = Counter()
    details = []
    for qa_id in ids:
        for offset, selection in rows[qa_id].get("temporal_selection_by_offset", {}).items():
            bracket = selection.get("reviewed_bracket_s")
            local_times = selection.get("local_times_s") or []
            if not isinstance(bracket, list) or len(bracket) != 2 or not local_times:
                status = "unavailable"
                before = inside = after = 0
            else:
                start, end = float(bracket[0]), float(bracket[1])
                before = sum(float(value) < start for value in local_times)
                inside = sum(start <= float(value) <= end for value in local_times)
                after = sum(float(value) > end for value in local_times)
                status = "covered" if min(local_times) <= start and max(local_times) >= end else "not_covered"
            counts[status] += 1
            details.append(
                {
                    "qa_id": qa_id,
                    "offset": offset,
                    "status": status,
                    "local_times_s": local_times,
                    "local_samples_before_bracket": before,
                    "local_samples_inside_bracket": inside,
                    "local_samples_after_bracket": after,
                    "window_source": selection.get("window_source"),
                    "fallback": bool(selection.get("fallback")),
                }
            )
    return {
        "offsets": sum(counts.values()),
        "covered": counts["covered"],
        "not_covered": counts["not_covered"],
        "unavailable": counts["unavailable"],
        "details": details,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--subset", choices=("tranche", "full"), default="tranche")
    ap.add_argument("--question-set", type=Path, default=Path("runs/phase2_temporal_coarsefine/question_set.json"))
    ap.add_argument("--baseline-dir", type=Path, default=Path("runs/stageA_q3vl8b_r16/legacy"))
    ap.add_argument("--t1-dir", type=Path, default=Path("runs/phase2_temporal_coarsefine/full"))
    ap.add_argument("--manual-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    _, records = _load_records(args.manifest, args.subset)
    questions = _question_rows(args.question_set)
    by_split = {split: [r for r in records if r["split"] == split] for split in ("valQ", "valT")}
    result = {
        "protocol": "bounded first-appearance pilot; T0 legacy, saved T1 guessed-window, T1R independently reviewed-window",
        "dataset_revision": C.DATASET_REVISION,
        "subset": args.subset,
        "selected_qa_ids": [record["qa_id"] for record in records],
        "annotation_digest": _annotation_digest(records, args.subset),
        "splits": {},
    }
    report = [
        "# Verified temporal-window pilot",
        "",
        "The T1R scorer uses the frozen answer prompt and changes only the selected frame window. Human review prose is retained for audit and is not sent to the answer model.",
        "",
    ]
    for split in ("valQ", "valT"):
        selected = by_split[split]
        ids = [record["qa_id"] for record in selected]
        gold = {qa_id: questions[qa_id]["correct_answer"] for qa_id in ids}
        t0 = read_jsonl(args.baseline_dir / f"scores_{split}.jsonl")
        t1 = read_jsonl(args.t1_dir / f"scores_{split}_T1.jsonl")
        t1r = read_jsonl(args.manual_dir / f"scores_{split}_T1R.jsonl")
        for arm, rows in (("T0", t0), ("T1", t1), ("T1R", t1r)):
            missing = set(ids) - rows.keys()
            if missing:
                raise ValueError(f"{split}/{arm}: missing selected IDs {sorted(missing)}")
        for record in selected:
            provenance = t1r[record["qa_id"]].get("window_manifest_record", {})
            for field in ("review_status", "boundary_status", "identity_status", "onset_censoring"):
                if provenance.get(field) != record[field]:
                    raise ValueError(f"{record['qa_id']}: T1R provenance does not match the review manifest")
        predictions = {
            arm: {qa_id: rows[qa_id]["pred"] for qa_id in ids}
            for arm, rows in (("T0", t0), ("T1", t1), ("T1R", t1r))
        }
        pairs = {
            "T0_vs_T1": _pair(t0, t1, gold, ids, "T0", "T1"),
            "T0_vs_T1R": _pair(t0, t1r, gold, ids, "T0", "T1R"),
            "T1_vs_T1R": _pair(t1, t1r, gold, ids, "T1", "T1R"),
        }
        roles = {}
        for role in ("baseline_error", "correct_control"):
            role_ids = [r["qa_id"] for r in selected if r["review_role"] == role]
            roles[role] = {
                "n": len(role_ids),
                "accuracy": {arm: _accuracy(predictions[arm], gold, role_ids) for arm in predictions},
                "paired": {
                    "T0_vs_T1": _pair(t0, t1, gold, role_ids, "T0", "T1"),
                    "T0_vs_T1R": _pair(t0, t1r, gold, role_ids, "T0", "T1R"),
                    "T1_vs_T1R": _pair(t1, t1r, gold, role_ids, "T1", "T1R"),
                },
            }
        evidence = _annotation_evidence(selected)
        fallback = {"T1": _fallback_counts(t1, ids), "T1R": _fallback_counts(t1r, ids)}
        bracket_coverage = _bracket_coverage(t1r, ids)
        result["splits"][split] = {
            "n": len(ids),
            "accuracy": {arm: _accuracy(predictions[arm], gold, ids) for arm in predictions},
            "paired": pairs,
            "roles": roles,
            "fallback_counts": fallback,
            "boundary_identity_evidence": evidence,
            "bracket_coverage": bracket_coverage,
            "annotation_records": [
                {
                    key: record.get(key)
                    for key in (
                        "qa_id",
                        "review_role",
                        "review_status",
                        "boundary_status",
                        "identity_status",
                        "onset_censoring",
                        "window_start_s",
                        "window_end_s",
                        "boundary_evidence",
                        "identity_evidence",
                        "reviewer",
                        "reviewed_at",
                    )
                }
                for record in selected
            ],
        }
        controls = [r["qa_id"] for r in selected if r["review_role"] == "correct_control"]
        report.extend(
            [
                f"## {split}",
                "",
                f"Accuracy: T0 {_accuracy(predictions['T0'], gold, ids)['correct']}/{len(ids)}; T1 {_accuracy(predictions['T1'], gold, ids)['correct']}/{len(ids)}; T1R {_accuracy(predictions['T1R'], gold, ids)['correct']}/{len(ids)}.",
                "",
                "| Pair | Fixes | Regressions | Wrong→wrong | Unchanged | Net |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, pair in pairs.items():
            report.append(
                f"| {name} | {pair['fixed']} | {pair['regressions']} | {pair['wrong_to_wrong']} | {pair['unchanged_predictions']} | {pair['net_correct_change']:+d} |"
            )
        control_accuracy = {arm: _accuracy(predictions[arm], gold, controls) for arm in predictions}
        report.extend(
            [
                "",
                "Controls: "
                + "; ".join(f"{arm} {value['correct']}/{value['total']}" for arm, value in control_accuracy.items())
                + ".",
                f"Fallback offsets: T1 {fallback['T1'].get('fallback_offsets', 0)}; T1R {fallback['T1R'].get('fallback_offsets', 0)}. Boundary reviewed: {evidence['boundary_reviewed']}/{len(ids)}; identity confirmed: {evidence['identity_confirmed']}/{len(ids)}. T1R bracket coverage: {bracket_coverage['covered']}/{bracket_coverage['offsets']} offsets.",
                "",
            ]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "verified_window_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "verified_window_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
