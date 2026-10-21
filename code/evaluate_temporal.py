"""Paired Phase 2 analysis against the frozen legacy temporal baseline."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import common as C

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "runs/stageA_q3vl8b_r16/legacy"
SPLITS = {name: ROOT / f"data/splits/{name}.json" for name in ("valQ", "valT")}
LETTERS = C.LETTERS
VIDEO_MANIFEST = json.loads((ROOT / "data/manifest.json").read_text())
VIEW_EVIDENCE_FIELDS = ("offset", "perm", "frame_indices", "frame_times_s")
SELECTION_EVIDENCE_FIELDS = (
    "mode",
    "coarse_indices",
    "local_indices",
    "frame_indices",
    "frame_times_s",
    "fallback",
    "confidence",
    "candidate_window_s",
    "coarse_times_s",
    "timestamp_context",
)


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("qa_id") in rows:
            raise ValueError(f"duplicate QA id at {path}:{number}")
        rows[row["qa_id"]] = row
    return rows


def bootstrap_interval(deltas: list[int], seed: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(10000))
    return [means[249], means[9749]]


def temporal_evidence(record: dict, question: dict) -> dict:
    views = []
    info = VIDEO_MANIFEST.get(question["video_path"], {})
    n_frames = int(info.get("n_frames", 0))
    fps = float(info.get("fps", 0.0))
    for view in record.get("views", []):
        evidence = {key: view[key] for key in VIEW_EVIDENCE_FIELDS if key in view}
        if "frame_indices" not in evidence and n_frames and fps > 0:
            indices = C.sample_index_list(n_frames, fps, float(view["offset"]), mode="legacy")
            evidence["frame_indices"] = indices
            evidence["frame_times_s"] = [index / fps for index in indices]
        views.append(evidence)
    result = {"views": views}
    if "temporal_selection_by_offset" in record:
        result["temporal_selection_by_offset"] = {
            offset: {
                key: selection[key]
                for key in SELECTION_EVIDENCE_FIELDS
                if key in selection
            }
            for offset, selection in record["temporal_selection_by_offset"].items()
        }
    return result


def pair_summary(
    gold: dict[str, str],
    old: dict[str, str],
    new: dict[str, str],
    ids: list[str],
    seed: int,
    old_scores: dict[str, dict],
    new_scores: dict[str, dict],
    old_arm: str,
    new_arm: str,
    questions: dict[str, dict],
) -> dict:
    counts = Counter()
    details = []
    deltas = []
    for qa_id in ids:
        was_correct = old[qa_id] == gold[qa_id]
        is_correct = new[qa_id] == gold[qa_id]
        deltas.append(int(is_correct) - int(was_correct))
        if old[qa_id] == new[qa_id]:
            label = "unchanged_prediction"
        elif not was_correct and is_correct:
            label = "fixed"
        elif was_correct and not is_correct:
            label = "regression"
        else:
            label = "wrong_to_wrong"
        counts[label] += 1
        if label != "unchanged_prediction":
            details.append(
                {
                    "qa_id": qa_id,
                    "old": old[qa_id],
                    "new": new[qa_id],
                    "gold": gold[qa_id],
                    "change": label,
                    "probabilities": {
                        old_arm: old_scores[qa_id]["p"],
                        new_arm: new_scores[qa_id]["p"],
                    },
                    "evidence": {
                        old_arm: temporal_evidence(old_scores[qa_id], questions[qa_id]),
                        new_arm: temporal_evidence(new_scores[qa_id], questions[qa_id]),
                    },
                }
            )
    return {
        "n": len(ids),
        "fixed": counts["fixed"],
        "regressions": counts["regression"],
        "wrong_to_wrong": counts["wrong_to_wrong"],
        "unchanged_predictions": counts["unchanged_prediction"],
        "net_correct_change": sum(deltas),
        "paired_accuracy_delta_95ci": bootstrap_interval(deltas, seed),
        "changed_answers": details,
    }


def accuracy(pred: dict[str, str], gold: dict[str, str], ids: list[str]) -> dict:
    correct = sum(pred[qa_id] == gold[qa_id] for qa_id in ids)
    return {"correct": correct, "total": len(ids), "accuracy": correct / len(ids) if ids else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/phase2_temporal_coarsefine/full")
    ap.add_argument("--output-dir", default="runs/phase2_temporal_coarsefine")
    args = ap.parse_args()
    run_dir, output_dir = ROOT / args.run_dir, ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"protocol": "paired inference-only comparison; valQ/valT are reused validation sets, not blind test data", "splits": {}}
    report = ["# Phase 2: question-aware temporal sampling results", "", "Only Temporal Grounding predictions changed. Non-temporal predictions remain the frozen legacy baseline.", ""]
    all_pair_data = {}
    for split_index, (split, split_path) in enumerate(SPLITS.items()):
        questions = json.loads(split_path.read_text())["QA"]
        by_id = {row["qa_id"]: row for row in questions}
        ids = [row["qa_id"] for row in questions]
        gold = {qa_id: by_id[qa_id]["correct_answer"] for qa_id in ids}
        baseline_all = read_jsonl(BASELINE / f"scores_{split}.jsonl")
        target_ids = [qa_id for qa_id in ids if by_id[qa_id]["task"] == "Temporal Grounding"]
        if len(target_ids) != {"valQ": 69, "valT": 133}[split]:
            raise ValueError(f"unexpected temporal question count for {split}")
        variants = {"T0": baseline_all}
        for variant in ("T1", "T2"):
            score = read_jsonl(run_dir / f"scores_{split}_{variant}.jsonl")
            if set(score) != set(target_ids):
                raise ValueError(f"{split}/{variant} incomplete or contains wrong IDs: {len(score)} vs {len(target_ids)}")
            variants[variant] = score
        target_gold = {qa_id: gold[qa_id] for qa_id in target_ids}
        target_pred = {variant: {qa_id: scores[qa_id]["pred"] for qa_id in target_ids} for variant, scores in variants.items()}
        overall = {}
        for variant, scores in variants.items():
            prediction = {qa_id: baseline_all[qa_id]["pred"] for qa_id in ids}
            if variant != "T0":
                prediction.update(target_pred[variant])
            overall[variant] = accuracy(prediction, gold, ids)
        metrics = {
            "questions": len(ids),
            "temporal_questions": len(target_ids),
            "temporal_accuracy": {variant: accuracy(target_pred[variant], target_gold, target_ids) for variant in variants},
            "overall_accuracy_with_non_temporal_baseline": overall,
            "overall_delta_percentage_points": {
                variant: 100 * (overall[variant]["accuracy"] - overall["T0"]["accuracy"])
                for variant in ("T1", "T2")
            },
            "paired": {
                "T0_vs_T1": pair_summary(
                    target_gold,
                    target_pred["T0"],
                    target_pred["T1"],
                    target_ids,
                    20261021 + split_index,
                    variants["T0"],
                    variants["T1"],
                    "T0",
                    "T1",
                    by_id,
                ),
                "T0_vs_T2": pair_summary(
                    target_gold,
                    target_pred["T0"],
                    target_pred["T2"],
                    target_ids,
                    20261031 + split_index,
                    variants["T0"],
                    variants["T2"],
                    "T0",
                    "T2",
                    by_id,
                ),
                "T1_vs_T2": pair_summary(
                    target_gold,
                    target_pred["T1"],
                    target_pred["T2"],
                    target_ids,
                    20261041 + split_index,
                    variants["T1"],
                    variants["T2"],
                    "T1",
                    "T2",
                    by_id,
                ),
            },
        }
        coverage = {}
        for variant in ("T1", "T2"):
            counts = Counter()
            for qa_id in target_ids:
                for selection in variants[variant][qa_id]["temporal_selection_by_offset"].values():
                    counts["fallback" if selection["fallback"] else "localized"] += 1
            coverage[variant] = {"offsets": sum(counts.values()), "localized": counts["localized"], "fallback": counts["fallback"], "localization_rate": counts["localized"] / sum(counts.values())}
        metrics["selector_coverage"] = coverage
        result["splits"][split] = metrics
        all_pair_data[split] = (target_gold, target_pred, target_ids)
        report.extend([
            f"## {split}",
            "",
            f"Temporal Grounding: T0 {metrics['temporal_accuracy']['T0']['correct']}/{len(target_ids)}; T1 {metrics['temporal_accuracy']['T1']['correct']}/{len(target_ids)}; T2 {metrics['temporal_accuracy']['T2']['correct']}/{len(target_ids)}.",
            f"Overall with other tasks frozen: T0 {overall['T0']['correct']}/{overall['T0']['total']}; T1 delta {metrics['overall_delta_percentage_points']['T1']:+.2f} pp; T2 delta {metrics['overall_delta_percentage_points']['T2']:+.2f} pp.",
            "",
            "| Pair | Fixes | Regressions | Wrong→wrong | Unchanged | Net correct | Paired 95% CI (accuracy points) | Localized / offset views |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for pair, label in (("T0_vs_T1", "T0 → T1"), ("T0_vs_T2", "T0 → T2"), ("T1_vs_T2", "T1 → T2")):
            item = metrics["paired"][pair]
            variant = "T1" if pair.endswith("T1") else "T2"
            ci = item["paired_accuracy_delta_95ci"]
            coverage_item = coverage.get(variant, {"localized": "—", "offsets": "—"})
            report.append(f"| {label} | {item['fixed']} | {item['regressions']} | {item['wrong_to_wrong']} | {item['unchanged_predictions']} | {item['net_correct_change']:+d} | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | {coverage_item['localized']}/{coverage_item['offsets']} |")
        report.append("")

    gates = {}
    for variant in ("T1", "T2"):
        q = result["splits"]["valQ"]
        t = result["splits"]["valT"]
        net_q = q["temporal_accuracy"][variant]["correct"] - q["temporal_accuracy"]["T0"]["correct"]
        net_t = t["temporal_accuracy"][variant]["correct"] - t["temporal_accuracy"]["T0"]["correct"]
        overall_drop_ok = all(result["splits"][split]["overall_delta_percentage_points"][variant] >= -1.0 for split in SPLITS)
        gates[variant] = {
            "valT_net_temporal_gain": net_t,
            "valQ_net_temporal_gain": net_q,
            "overall_drop_at_most_1pp_both_splits": overall_drop_ok,
            "candidate_gate_passed": net_t >= 3 and net_q >= 0 and overall_drop_ok,
        }
    result["candidate_gate"] = gates
    report.extend(["## Candidate gate", "", "A variant passes the predeclared pilot gate only with at least +3 net temporal answers on valT, nonnegative net change on valQ, and no more than a 1 pp overall drop on either split.", ""])
    for variant, gate in gates.items():
        report.append(f"- {variant}: valT {gate['valT_net_temporal_gain']:+d}; valQ {gate['valQ_net_temporal_gain']:+d}; overall condition {'pass' if gate['overall_drop_at_most_1pp_both_splits'] else 'fail'}; candidate {'PASS' if gate['candidate_gate_passed'] else 'not passed'}.")
    report.extend(["", "## Changed predictions", "", "Every changed item, including all four aggregated option probabilities per arm, is retained in `phase2_comparison.json`; inspect its saved selected-frame evidence before interpreting qualitative fixes or regressions.", "", "Validation sets are reused for diagnosis and are not blind test sets. The paired bootstrap intervals describe question-level variation only; clips may contribute multiple questions, so they are not cluster-corrected."])

    result_path = output_dir / "phase2_comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path = output_dir / "phase2_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(report_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
