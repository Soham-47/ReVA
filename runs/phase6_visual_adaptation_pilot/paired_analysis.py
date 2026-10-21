#!/usr/bin/env python3
"""Reproducible local paired analysis for the Phase 6 A0/A1/A2 scores.

The script reads only the frozen score artifacts and split metadata.  It does
not invoke the scorer, load a model, or write beside the raw JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "phase6_visual_adaptation_pilot"
SPLITS = ("valQ", "valT")
ARMS = ("A0", "A1", "A2")
LETTERS = "ABCD"
EXPECTED_VIEWS = ((0.0, "ABCD"), (0.0, "BCDA"), (0.5, "ABCD"), (0.5, "BCDA"))
PINNED_MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
PINNED_DATASET_REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261019
GATE = {
    "valT_accuracy_delta": 0.02,
    "valQ_accuracy_delta": 0.01,
    "probability_floor": -0.02,
    "margin_floor": -0.02,
    "slice_floor": -0.01,
    "slice_min_n": 50,
}


def _json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_probabilities(values: Any, label: str) -> list[float]:
    assert isinstance(values, list) and len(values) == 4, f"{label}: expected four probabilities"
    result = [float(value) for value in values]
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result), (
        f"{label}: probability outside [0, 1]"
    )
    assert abs(sum(result) - 1.0) < 1e-4, f"{label}: probabilities do not sum to one"
    return result


def _load_gold(split: str) -> dict[str, dict[str, Any]]:
    payload = _json(ROOT / "data" / "splits" / f"{split}.json")
    rows = payload["QA"]
    result = {row["qa_id"]: row for row in rows}
    assert len(result) == len(rows), f"{split}: duplicate gold IDs"
    return result


def _load_scores(path: Path, gold: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            score = json.loads(line)
            qa_id = score.get("qa_id")
            assert qa_id in gold, f"{path}:{line_number}: unknown qa_id {qa_id!r}"
            assert qa_id not in rows, f"{path}:{line_number}: duplicate qa_id {qa_id!r}"
            pred = score.get("pred")
            assert pred in LETTERS, f"{path}:{line_number}: invalid prediction {pred!r}"
            p = _assert_probabilities(score.get("p"), f"{path}:{line_number}: p")
            views = score.get("views")
            assert isinstance(views, list) and len(views) == len(EXPECTED_VIEWS), (
                f"{path}:{line_number}: expected four views"
            )
            for view_number, (view, expected) in enumerate(zip(views, EXPECTED_VIEWS), 1):
                signature = (float(view.get("offset")), str(view.get("perm")))
                assert math.isclose(signature[0], expected[0], abs_tol=1e-9) and signature[1] == expected[1], (
                    f"{path}:{line_number}: unexpected view {view_number}: {signature!r}"
                )
                _assert_probabilities(view.get("p"), f"{path}:{line_number}: view {view_number} p")
                assert view.get("argmax_shown") in LETTERS, (
                    f"{path}:{line_number}: invalid argmax_shown"
                )
            assert pred == LETTERS[max(range(4), key=p.__getitem__)], (
                f"{path}:{line_number}: pred does not match TTA p"
            )
            question = gold[qa_id]
            gold_letter = question["correct_answer"]
            gold_index = LETTERS.index(gold_letter)
            wrong = p[:gold_index] + p[gold_index + 1 :]
            rows[qa_id] = {
                "qa_id": qa_id,
                "video": question["video_path"],
                "task": question["task"],
                "category": question["category"],
                "source": question["dataset_name"],
                "question": question["question"],
                "gold": gold_letter,
                "pred": pred,
                "single_pred": LETTERS[max(range(4), key=lambda index: float(views[0]["p"][index]))],
                "ok": pred == gold_letter,
                "single_ok": LETTERS[max(range(4), key=lambda index: float(views[0]["p"][index]))]
                == gold_letter,
                "confidence": max(p),
                "p_gold": p[gold_index],
                "margin": p[gold_index] - max(wrong),
            }
    assert set(rows) == set(gold), f"{path}: score ID coverage differs from {path.name} gold split"
    return rows


def _validate_manifest_pair(split: str) -> dict[str, Any]:
    paths = [RUN / "scores" / arm / f"scores_{split}.jsonl.manifest.json" for arm in ("A1", "A2")]
    manifests = [_json(path) for path in paths]
    first, second = manifests
    assert set(first) == set(second), f"{split}: A1/A2 manifest fields differ"
    ignored = {"adapter", "adapter_checksum"}
    assert {key: first[key] for key in first if key not in ignored} == {
        key: second[key] for key in second if key not in ignored
    }, f"{split}: A1/A2 score settings differ"
    expected = {
        "model_revision": PINNED_MODEL_REVISION,
        "dataset_revision": PINNED_DATASET_REVISION,
        "frame_sampling": "legacy",
        "offsets": [0.0, 0.5],
        "rots": 2,
        "schema_version": 1,
        "split": split,
        "nshards": 1,
        "shard": 0,
    }
    for key, value in expected.items():
        assert first.get(key) == value, f"{split}: manifest {key}={first.get(key)!r}, expected {value!r}"
    assert first.get("model_id") == "Qwen/Qwen3-VL-8B-Instruct"
    return {
        "model_id": first["model_id"],
        "model_revision": first["model_revision"],
        "dataset_revision": first["dataset_revision"],
        "frame_sampling": first["frame_sampling"],
        "offsets": first["offsets"],
        "rots": first["rots"],
        "split_checksum": first["split_checksum"],
    }


def _taxonomy(row: dict[str, Any]) -> str:
    task = row["task"]
    category = row["category"]
    if task in {"Temporal Grounding", "Change Detection", "Trend and Pattern"}:
        return "temporal_and_change"
    if category == "Spatial Reasoning":
        return "spatial_and_structural"
    if category == "Causal Reasoning":
        return "reasoning"
    if category == "Factual Perception":
        return "factual_perception"
    return "domain_and_ambiguity"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    assert n
    return {
        "n": n,
        "tta_correct": sum(row["ok"] for row in rows),
        "tta_accuracy": _mean(float(row["ok"]) for row in rows),
        "single_correct": sum(row["single_ok"] for row in rows),
        "single_accuracy": _mean(float(row["single_ok"]) for row in rows),
        "p_gold_mean": _mean(row["p_gold"] for row in rows),
        "p_gold_median": statistics.median(row["p_gold"] for row in rows),
        "p_gold_gt_05": _mean(float(row["p_gold"] > 0.5) for row in rows),
        "margin_mean": _mean(row["margin"] for row in rows),
        "margin_median": statistics.median(row["margin"] for row in rows),
        "margin_positive": _mean(float(row["margin"] > 0.0) for row in rows),
    }


def _transition(candidate: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for qa_id in candidate:
        candidate_ok = candidate[qa_id]["ok"]
        reference_ok = reference[qa_id]["ok"]
        if candidate_ok and not reference_ok:
            counts["fixes"] += 1
        elif not candidate_ok and reference_ok:
            counts["regressions"] += 1
        elif candidate_ok:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    n = len(candidate)
    return {
        "n": n,
        "fixes": counts["fixes"],
        "regressions": counts["regressions"],
        "unchanged_correct": counts["unchanged_correct"],
        "unchanged_wrong": counts["unchanged_wrong"],
        "accuracy_delta": _mean(float(candidate[qa_id]["ok"]) - float(reference[qa_id]["ok"]) for qa_id in candidate),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_delta(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    *,
    seed: int,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for qa_id in candidate:
        row = candidate[qa_id]
        grouped[row["video"]].append(float(row["ok"]) - float(reference[qa_id]["ok"]))
    videos = sorted(grouped)
    assert videos
    rng = random.Random(seed)
    sampled = []
    for _ in range(reps):
        numerator = 0.0
        denominator = 0
        for _ in videos:
            deltas = grouped[videos[rng.randrange(len(videos))]]
            numerator += sum(deltas)
            denominator += len(deltas)
        sampled.append(numerator / denominator)
    sampled.sort()
    point = _mean(float(candidate[qa_id]["ok"]) - float(reference[qa_id]["ok"]) for qa_id in candidate)
    return point, _percentile(sampled, 0.025), _percentile(sampled, 0.975)


def _status(candidate: bool, label: str) -> str:
    return "PASS" if candidate else f"FAIL ({label})"


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def _md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(output)


def _transition_rows(scores: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    fields = [
        "split", "qa_id", "video", "task", "category", "source", "taxonomy", "gold",
        "question", "a0_pred", "a1_pred", "a2_pred", "a0_single_pred", "a1_single_pred",
        "a2_single_pred", "a0_ok", "a1_ok", "a2_ok", "a1_vs_a0", "a2_vs_a0", "a2_vs_a1",
        "a0_p_gold", "a1_p_gold", "a2_p_gold", "a0_margin", "a1_margin", "a2_margin",
        "a0_confidence", "a1_confidence", "a2_confidence",
    ]
    result = []
    for split in SPLITS:
        for qa_id in sorted(scores[split]["A0"]):
            rows = {arm: scores[split][arm][qa_id] for arm in ARMS}
            def state(arm: str, reference: str) -> str:
                if rows[arm]["ok"] and not rows[reference]["ok"]:
                    return "fix"
                if not rows[arm]["ok"] and rows[reference]["ok"]:
                    return "regression"
                return "unchanged_correct" if rows[arm]["ok"] else "unchanged_wrong"
            row = {
                "split": split,
                "qa_id": qa_id,
                "video": rows["A0"]["video"],
                "task": rows["A0"]["task"],
                "category": rows["A0"]["category"],
                "source": rows["A0"]["source"],
                "taxonomy": _taxonomy(rows["A0"]),
                "gold": rows["A0"]["gold"],
                "question": rows["A0"]["question"],
            }
            for arm in ARMS:
                prefix = arm.lower()
                for key in ("pred", "single_pred", "ok", "p_gold", "margin", "confidence"):
                    row[f"{prefix}_{key}"] = rows[arm][key]
            row["a1_vs_a0"] = state("A1", "A0")
            row["a2_vs_a0"] = state("A2", "A0")
            row["a2_vs_a1"] = state("A2", "A1")
            result.append({field: row[field] for field in fields})
    return result


def _write_transitions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _top_changes(
    candidate: dict[str, dict[str, Any]],
    reference: dict[str, dict[str, Any]],
    kind: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows = []
    for qa_id, row in candidate.items():
        if kind == "fix" and not (row["ok"] and not reference[qa_id]["ok"]):
            continue
        if kind == "regression" and not (not row["ok"] and reference[qa_id]["ok"]):
            continue
        rows.append(row | {"reference_pred": reference[qa_id]["pred"]})
    return sorted(rows, key=lambda row: (-row["confidence"], row["qa_id"]))[:limit]


def _build_report(
    scores: dict[str, dict[str, dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
    transition_path: Path,
    reps: int,
) -> str:
    lines = [
        "# Phase 6 paired visual adaptation analysis",
        "",
        "This report uses the frozen A0 score JSONL under `runs/stageA_q3vl8b_r16/legacy/` and the copied A1/A2 score artifacts under this run directory. `valQ` and `valT` are development splits; these results do not establish held-out or test generalization.",
        "",
        f"Validation: A0 copied references match the frozen baseline byte-for-byte; A1/A2 manifests agree on pinned settings, offsets `[0.0, 0.5]`, two rotations, legacy sampling, and four ordered view signatures. Bootstrap uses {reps:,} deterministic video-cluster resamples (seed `{BOOTSTRAP_SEED}`).",
        "",
        "## Compute and provenance",
        "",
        "The supplied Slurm accounting is recorded in [`phase6_provenance.md`](phase6_provenance.md): A1 training job 402807 on H200 took 00:35:07 (train loss 0.3526, peak 27,926 MiB, 85% GPU utilization); A2 training job 402810 on H100 took 00:42:31 (train loss 0.3567, peak 28,070 MiB, 81% utilization); A1/A2 scoring jobs 402864/402865 took 00:24:38/00:26:23 and completed 1,654 rows each. The A2 manifest points to the remote H100 adapter run and its adjacent `RUN_PROVENANCE`; no direct HPC currency charge is exposed.",
        "",
        "## Overall metrics",
        "",
    ]
    for split in SPLITS:
        lines.extend([f"### {split}", ""])
        rows = []
        for arm in ARMS:
            metric = _metrics(scores[split][arm].values())
            rows.append(
                [arm, f"{metric['tta_correct']}/{metric['n']} ({_fmt_pct(metric['tta_accuracy'])})",
                 f"{metric['single_correct']}/{metric['n']} ({_fmt_pct(metric['single_accuracy'])})",
                 f"{metric['p_gold_mean']:.4f}", f"{metric['p_gold_median']:.4f}",
                 _fmt_pct(metric["p_gold_gt_05"]), f"{metric['margin_mean']:.4f}",
                 f"{metric['margin_median']:.4f}", _fmt_pct(metric["margin_positive"])])
        lines.extend([_md_table(
            ["Arm", "TTA accuracy", "Single-view accuracy", "mean p_gold", "median p_gold", "p_gold > 0.5", "mean margin", "median margin", "positive margin"], rows
        ), ""])

    lines.extend(["## A2-minus-A1 confidence deltas", "", _md_table(
        ["Split", "mean p_gold", "median p_gold", "p_gold > 0.5", "mean margin", "median margin", "positive margin"],
        [
            [split] + [
                f"{_metrics(scores[split]["A2"].values())[key] - _metrics(scores[split]["A1"].values())[key]:+.4f}"
                for key in ("p_gold_mean", "p_gold_median", "p_gold_gt_05", "margin_mean", "margin_median", "margin_positive")
            ]
            for split in SPLITS
        ],
    ), ""])

    lines.extend(["## Paired transitions and clustered intervals", ""])
    for split_index, split in enumerate(SPLITS):
        lines.extend([f"### {split}", "", _md_table(
            ["Candidate vs reference", "fixes", "regressions", "unchanged correct", "unchanged wrong", "accuracy delta", "95% video-cluster interval"],
            [
                [f"{candidate} vs {reference}",
                 _transition(scores[split][candidate], scores[split][reference])["fixes"],
                 _transition(scores[split][candidate], scores[split][reference])["regressions"],
                 _transition(scores[split][candidate], scores[split][reference])["unchanged_correct"],
                 _transition(scores[split][candidate], scores[split][reference])["unchanged_wrong"],
                 _fmt_delta(interval[0]), f"[{_fmt_delta(interval[1])}, {_fmt_delta(interval[2])}]" ]
                for pair_index, (candidate, reference) in enumerate((("A1", "A0"), ("A2", "A0"), ("A2", "A1")))
                for interval in [_bootstrap_delta(scores[split][candidate], scores[split][reference], seed=BOOTSTRAP_SEED + split_index * 10 + pair_index, reps=reps)]
            ]
        ), ""])

    lines.extend(["## Task, category, and source slices", ""])
    for split in SPLITS:
        lines.extend([f"### {split}", ""])
        for field, title in (("task", "Task"), ("category", "Top-level category"), ("source", "Source")):
            slice_names = sorted({scores[split]["A0"][qa_id][field] for qa_id in scores[split]["A0"]})
            rows = []
            for value in slice_names:
                arm_metrics = {
                    arm: _metrics(row for row in scores[split][arm].values() if row[field] == value)
                    for arm in ARMS
                }
                rows.append([value, arm_metrics["A0"]["n"], _fmt_pct(arm_metrics["A0"]["tta_accuracy"]),
                             _fmt_pct(arm_metrics["A1"]["tta_accuracy"]), _fmt_pct(arm_metrics["A2"]["tta_accuracy"]),
                             _fmt_delta(arm_metrics["A2"]["tta_accuracy"] - arm_metrics["A1"]["tta_accuracy"])])
            lines.extend([f"#### {title}", "", _md_table([title, "n", "A0 TTA", "A1 TTA", "A2 TTA", "A2-A1"], rows), ""])

    lines.extend(["## Temporal Grounding subset", ""])
    temporal_rows = []
    for split in SPLITS:
        metrics = {arm: _metrics(row for row in scores[split][arm].values() if row["task"] == "Temporal Grounding") for arm in ARMS}
        temporal_rows.append([split, metrics["A0"]["n"], _fmt_pct(metrics["A0"]["tta_accuracy"]), _fmt_pct(metrics["A1"]["tta_accuracy"]), _fmt_pct(metrics["A2"]["tta_accuracy"]), _fmt_delta(metrics["A2"]["tta_accuracy"] - metrics["A1"]["tta_accuracy"])])
    lines.extend([_md_table(["Split", "n", "A0 TTA", "A1 TTA", "A2 TTA", "A2-A1"], temporal_rows), ""])

    lines.extend(["## Predeclared A2-vs-A1 screening gate", "", "The thresholds below are applied exactly as declared in `tasks/phase6_visual_adaptation_pilot.md`; bootstrap intervals are reported for context and do not replace these point thresholds.", ""])
    gate_rows = []
    val_t = {arm: _metrics(scores["valT"][arm].values()) for arm in ARMS}
    val_q = {arm: _metrics(scores["valQ"][arm].values()) for arm in ARMS}
    for split, threshold in (("valT", GATE["valT_accuracy_delta"]), ("valQ", GATE["valQ_accuracy_delta"])):
        metrics = {arm: _metrics(scores[split][arm].values()) for arm in ARMS}
        observed = metrics["A2"]["tta_accuracy"] - metrics["A1"]["tta_accuracy"]
        gate_rows.append([f"{split} TTA accuracy", _fmt_delta(observed), _fmt_delta(threshold), _status(observed >= threshold, "below threshold")])
    for split in SPLITS:
        metrics = {arm: _metrics(scores[split][arm].values()) for arm in ARMS}
        for metric_name, label in (("p_gold_mean", "mean p_gold"), ("margin_mean", "mean gold margin")):
            observed = metrics["A2"][metric_name] - metrics["A1"][metric_name]
            gate_rows.append([f"{split} {label}", f"{observed:+.4f}", f">= {GATE['probability_floor']:+.2f}", _status(observed >= GATE["probability_floor"], "decrease exceeds 0.02")])
    for split in SPLITS:
        for field in ("category", "source"):
            values = sorted({scores[split]["A0"][qa_id][field] for qa_id in scores[split]["A0"]})
            for value in values:
                arm_metrics = {arm: _metrics(row for row in scores[split][arm].values() if row[field] == value) for arm in ARMS}
                if arm_metrics["A0"]["n"] < GATE["slice_min_n"]:
                    continue
                observed = arm_metrics["A2"]["tta_accuracy"] - arm_metrics["A1"]["tta_accuracy"]
                gate_rows.append([f"{split} {field}={value} (n={arm_metrics['A0']['n']})", _fmt_delta(observed), _fmt_delta(GATE["slice_floor"]), _status(observed >= GATE["slice_floor"], "regression exceeds 1.0 pp")])
    lines.extend([_md_table(["Gate", "Observed", "Threshold", "Result"], gate_rows), ""])
    gate_pass = all(row[-1] == "PASS" for row in gate_rows)
    lines.extend([f"**Overall A2-vs-A1 screening result: {'PASS' if gate_pass else 'NO-GO'}.**", ""])
    lines.extend(["## Highest-confidence paired changes", "", "Taxonomy labels are deterministic task/category proxies for review: factual perception, spatial and structural, temporal and change, reasoning, or domain and ambiguity. Input/readout metadata is validated separately; it cannot be inferred from answer probabilities alone.", ""])
    for split in SPLITS:
        for candidate in ("A1", "A2"):
            lines.extend([f"### {split}: {candidate} vs A0", ""])
            for kind, title in (("fix", "Repaired baseline errors"), ("regression", "Regressions")):
                rows = []
                for row in _top_changes(scores[split][candidate], scores[split]["A0"], kind):
                    rows.append([row["qa_id"], row["task"], row["source"], _taxonomy(row), f"{row['reference_pred']}->{row['pred']}", f"{row['confidence']:.3f}", row["question"][:100]])
                lines.extend([f"#### {title}", "", _md_table(["qa_id", "task", "source", "taxonomy", "transition", "candidate confidence", "question"], rows or [["none", "", "", "", "", "", ""]]), ""])
    lines.extend(["## Outputs", "", f"- Per-question paired transitions: `{transition_path.relative_to(ROOT)}`", f"- Source score manifests: `{manifests['valQ']['model_revision']}` model revision and dataset revision `{manifests['valQ']['dataset_revision']}`", ""])
    return "\n".join(lines)


def self_check() -> None:
    base = {
        "video": "v1", "task": "Temporal Grounding", "category": "Temporal Understanding",
        "source": "S", "gold": "A", "question": "q", "single_pred": "A", "single_ok": True,
        "confidence": 0.9, "p_gold": 0.8, "margin": 0.6,
    }
    reference = {"1": base | {"ok": False}, "2": base | {"video": "v2", "ok": True}}
    candidate = {"1": base | {"ok": True}, "2": base | {"video": "v2", "ok": False}}
    assert _metrics(reference.values())["n"] == 2
    assert _transition(candidate, reference)["fixes"] == 1
    assert _transition(candidate, reference)["regressions"] == 1
    first = _bootstrap_delta(candidate, reference, seed=7, reps=40)
    assert first == _bootstrap_delta(candidate, reference, seed=7, reps=40)
    assert _taxonomy(base) == "temporal_and_change"
    assert _assert_probabilities([0.25, 0.25, 0.25, 0.25], "self-check") == [0.25] * 4


def analyze(reps: int) -> tuple[Path, Path]:
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    manifests = {}
    for split in SPLITS:
        gold = _load_gold(split)
        legacy = RUN.parent / "stageA_q3vl8b_r16" / "legacy" / f"scores_{split}.jsonl"
        copied = RUN / "scores" / "A0" / f"scores_{split}.jsonl"
        assert _sha256(legacy) == _sha256(copied), f"{split}: copied A0 differs from frozen legacy baseline"
        paths = {
            "A0": legacy,
            "A1": RUN / "scores" / "A1" / f"scores_{split}.jsonl",
            "A2": RUN / "scores" / "A2" / f"scores_{split}.jsonl",
        }
        scores[split] = {arm: _load_scores(path, gold) for arm, path in paths.items()}
        assert set(scores[split]["A0"]) == set(scores[split]["A1"]) == set(scores[split]["A2"])
        manifests[split] = _validate_manifest_pair(split)
    transitions = _transition_rows(scores)
    transition_path = RUN / "paired_transitions.csv"
    report_path = RUN / "paired_report.md"
    _write_transitions(transition_path, transitions)
    report_path.write_text(_build_report(scores, manifests, transition_path, reps))
    return transition_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run only the assert-based in-memory self-check")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args()
    assert args.bootstrap_reps > 0
    self_check()
    if args.self_check:
        print("self-check: OK")
        return 0
    transitions, report = analyze(args.bootstrap_reps)
    print(f"wrote {transitions}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
