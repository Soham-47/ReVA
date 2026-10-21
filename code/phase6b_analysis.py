#!/usr/bin/env python3
"""Analyze Phase 6B score artifacts without loading a model.

The utility is deliberately independent of the scorer.  It validates the
four-view protocol, joins each row to the frozen split gold labels, and writes
machine-readable summaries plus a compact Markdown report.  Score directories
are passed as ``--arm NAME=DIR``; A0 may point at its frozen legacy directory.
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


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("valQ", "valT")
LETTERS = "ABCD"
EXPECTED_VIEWS = ((0.0, "ABCD"), (0.0, "BCDA"), (0.5, "ABCD"), (0.5, "BCDA"))
DEFAULT_BOOTSTRAP_REPS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261019


def _json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probabilities(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label}: expected four probabilities")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result):
        raise ValueError(f"{label}: probability outside [0, 1]")
    if abs(sum(result) - 1.0) > 1e-4:
        raise ValueError(f"{label}: probabilities do not sum to one")
    return result


def _gold(split: str) -> dict[str, dict[str, Any]]:
    path = ROOT / "data" / "splits" / f"{split}.json"
    rows = _json(path)["QA"]
    result = {row["qa_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{split}: duplicate QA IDs")
    return result


def _load_scores(path: Path, gold: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            score = json.loads(line)
            qa_id = score.get("qa_id")
            if qa_id not in gold:
                raise ValueError(f"{path}:{line_number}: unknown qa_id {qa_id!r}")
            if qa_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate qa_id {qa_id!r}")
            p = _probabilities(score.get("p"), f"{path}:{line_number}: p")
            pred = score.get("pred")
            if pred not in LETTERS or pred != LETTERS[max(range(4), key=p.__getitem__)]:
                raise ValueError(f"{path}:{line_number}: invalid or inconsistent prediction")
            views = score.get("views")
            if not isinstance(views, list) or len(views) != len(EXPECTED_VIEWS):
                raise ValueError(f"{path}:{line_number}: expected four views")
            for view_number, (view, expected) in enumerate(zip(views, EXPECTED_VIEWS), 1):
                signature = (float(view.get("offset")), str(view.get("perm")))
                if not math.isclose(signature[0], expected[0], abs_tol=1e-9) or signature[1] != expected[1]:
                    raise ValueError(f"{path}:{line_number}: unexpected view {view_number}: {signature!r}")
                _probabilities(view.get("p"), f"{path}:{line_number}: view {view_number}")
            question = gold[qa_id]
            gold_letter = question["correct_answer"]
            gold_index = LETTERS.index(gold_letter)
            wrong = p[:gold_index] + p[gold_index + 1 :]
            single_p = _probabilities(views[0].get("p"), f"{path}:{line_number}: single view")
            single_pred = LETTERS[max(range(4), key=single_p.__getitem__)]
            rows[qa_id] = {
                "qa_id": qa_id,
                "video": question["video_path"],
                "task": question["task"],
                "category": question["category"],
                "source": question.get("dataset_name", ""),
                "question": question["question"],
                "gold": gold_letter,
                "pred": pred,
                "single_pred": single_pred,
                "ok": pred == gold_letter,
                "single_ok": single_pred == gold_letter,
                "p_gold": p[gold_index],
                "margin": p[gold_index] - max(wrong),
                "confidence": max(p),
                "views": views,
            }
    if set(rows) != set(gold):
        missing = sorted(set(gold) - set(rows))[:5]
        extra = sorted(set(rows) - set(gold))[:5]
        raise ValueError(f"{path}: score coverage mismatch; missing={missing}, extra={extra}")
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "correct": sum(bool(row["ok"]) for row in rows),
        "accuracy": _mean(float(row["ok"]) for row in rows),
        "single_correct": sum(bool(row["single_ok"]) for row in rows),
        "single_accuracy": _mean(float(row["single_ok"]) for row in rows),
        "p_gold_mean": _mean(row["p_gold"] for row in rows),
        "p_gold_median": statistics.median(row["p_gold"] for row in rows),
        "p_gold_gt_05": _mean(float(row["p_gold"] > 0.5) for row in rows),
        "margin_mean": _mean(row["margin"] for row in rows),
        "margin_median": statistics.median(row["margin"] for row in rows),
        "margin_positive": _mean(float(row["margin"] > 0) for row in rows),
    }


def _transition(candidate: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for qa_id in candidate:
        c, r = bool(candidate[qa_id]["ok"]), bool(reference[qa_id]["ok"])
        if c and not r:
            counts["fixes"] += 1
        elif r and not c:
            counts["regressions"] += 1
        elif c:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    return {
        "n": len(candidate),
        "fixes": counts["fixes"],
        "regressions": counts["regressions"],
        "unchanged_correct": counts["unchanged_correct"],
        "unchanged_wrong": counts["unchanged_wrong"],
        "accuracy_delta": _mean(float(candidate[q]["ok"]) - float(reference[q]["ok"]) for q in candidate),
        "p_gold_delta": _mean(candidate[q]["p_gold"] - reference[q]["p_gold"] for q in candidate),
        "margin_delta": _mean(candidate[q]["margin"] - reference[q]["margin"] for q in candidate),
    }


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def _video_bootstrap(candidate: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]], seed: int, reps: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for qa_id, row in candidate.items():
        grouped[row["video"]].append(float(row["ok"]) - float(reference[qa_id]["ok"]))
    videos = sorted(grouped)
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        values = [delta for _ in videos for delta in grouped[videos[rng.randrange(len(videos))]]]
        samples.append(_mean(values))
    samples.sort()
    observed = _mean(float(candidate[q]["ok"]) - float(reference[q]["ok"]) for q in candidate)
    return {"estimate": observed, "lower_95": _percentile(samples, 0.025), "upper_95": _percentile(samples, 0.975), "videos": len(videos), "reps": reps}


def _manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _json(path)
    payload.setdefault("adapter_merge_mode", "merged")
    return payload


def _resolve_scores(arm_dirs: dict[str, Path], arm: str, split: str) -> Path:
    root = arm_dirs[arm]
    candidate = root / f"scores_{split}.jsonl"
    if candidate.exists():
        return candidate
    if (root / "scores" / arm / f"scores_{split}.jsonl").exists():
        return root / "scores" / arm / f"scores_{split}.jsonl"
    raise FileNotFoundError(f"no {split} score file for {arm}: checked {candidate}")


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be NAME=DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("arm must be NAME=DIR")
    return name, Path(path)


def _csv_rows(scores: dict[str, dict[str, dict[str, Any]]], split: str, arms: list[str]) -> list[dict[str, Any]]:
    result = []
    for qa_id in sorted(scores[split][arms[0]]):
        row = scores[split][arms[0]][qa_id]
        item: dict[str, Any] = {
            "split": split,
            "qa_id": qa_id,
            "video": row["video"],
            "task": row["task"],
            "category": row["category"],
            "source": row["source"],
            "gold": row["gold"],
            "question": row["question"],
        }
        for arm in arms:
            value = scores[split][arm][qa_id]
            prefix = arm.lower()
            item.update({
                f"{prefix}_pred": value["pred"],
                f"{prefix}_ok": int(value["ok"]),
                f"{prefix}_p_gold": value["p_gold"],
                f"{prefix}_margin": value["margin"],
                f"{prefix}_confidence": value["confidence"],
            })
        result.append(item)
    return result


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _fmt_delta(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(item).replace("|", "\\|") for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(summary: dict[str, Any], arms: list[str]) -> str:
    lines = [
        "# Phase 6B paired analysis",
        "",
        "This report is descriptive. It uses the named score arms and the same four-view protocol; paired changes do not by themselves establish causation or held-out generalization.",
        "",
        ("## Cancelled arms\n\n" + ", ".join(summary["cancelled_arms"]) + " was cancelled before complete training/evaluation and has no score artifact.\n\n" if summary.get("cancelled_arms") else ""),
        "## Overall metrics",
        "",
    ]
    for split in SPLITS:
        lines.extend([f"### {split}", ""])
        rows = []
        for arm in arms:
            metric = summary["metrics"][split][arm]
            rows.append([arm, f"{metric['correct']}/{metric['n']} ({_fmt_pct(metric['accuracy'])})", _fmt_pct(metric["single_accuracy"]), f"{metric['p_gold_mean']:.4f}", f"{metric['margin_mean']:.4f}"])
        lines.extend([_table(["Arm", "TTA accuracy", "Single-view accuracy", "mean p_gold", "mean gold margin"], rows), ""])
    lines.extend(["## Paired transitions", ""])
    for split in SPLITS:
        lines.extend([f"### {split}", ""])
        rows = []
        for pair, transition in summary["transitions"][split].items():
            interval = summary["video_bootstrap"][split][pair]
            rows.append([pair, transition["fixes"], transition["regressions"], _fmt_delta(transition["accuracy_delta"]), f"[{_fmt_delta(interval['lower_95'])}, {_fmt_delta(interval['upper_95'])}]"])
        lines.extend([_table(["Candidate vs reference", "Fixes", "Regressions", "Accuracy delta", "95% video-cluster interval"], rows), ""])
    lines.extend(["## Predeclared valT gate", "", "A candidate must exceed A0 on the valT point estimate to pass screening. Strong positive evidence requires the video-cluster 95% interval to lie above zero. The historical category guard requires every valT top-level category with n >= 50 to stay within -1 percentage point.", ""])
    gate_rows = []
    for arm in arms[1:]:
        primary = summary["gate"]["primary_valT_vs_A0"].get(arm, {})
        guard = summary["gate"]["category_guard_vs_A0"].get(arm, {})
        interval = primary.get("video_cluster_interval", {})
        gate_rows.append([
            arm,
            _fmt_delta(primary.get("accuracy_delta", 0.0)),
            "PASS" if primary.get("screening_pass_point_estimate") else "FAIL",
            f"[{_fmt_delta(interval.get('lower_95', 0.0))}, {_fmt_delta(interval.get('upper_95', 0.0))}]",
            "PASS" if primary.get("strong_positive_evidence") else "NO",
            "PASS" if guard.get("passed") else "FAIL",
        ])
    lines.extend([_table(["Arm", "valT delta", "Point screen", "Video 95% interval", "Strong positive", "Category guard"], gate_rows), ""])
    if "A3" in summary["gate"]["primary_valT_vs_A0"] and not summary["gate"]["primary_valT_vs_A0"]["A3"]["screening_pass_point_estimate"]:
        lines.extend(["**A3 does not exceed A0 on the primary valT point estimate; under the predeclared rule this is no evidence to proceed with vision LoRA.**", ""])
    lines.extend(["## Task/category/source slices", ""])
    for split in SPLITS:
        lines.extend([f"### {split}", ""])
        for field in ("task", "category", "source"):
            rows = []
            for value, grouped in summary["slices"][split][field].items():
                rows.append([value, grouped[arms[0]]["n"]] + [_fmt_pct(grouped[arm]["accuracy"]) for arm in arms])
            lines.extend([f"#### {field}", "", _table([field, "n"] + arms, rows), ""])
    lines.extend(["## Provenance", "", "The machine-readable JSON records split checksums, adapter checksums when present, merge mode, view protocol, and score-file SHA256 values. Slurm GPU/memory accounting and the cancelled A4 status are recorded in `phase6b_resources.json`.", ""])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    arm_dirs = dict(args.arm)
    arms = list(arm_dirs)
    if len(arms) < 2 or "A0" not in arm_dirs:
        raise ValueError("provide A0 and at least one candidate via --arm NAME=DIR")
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {"arms": arms, "splits": {}}
    for split_index, split in enumerate(SPLITS):
        gold = _gold(split)
        scores[split] = {}
        provenance["splits"][split] = {"n_gold": len(gold), "arms": {}}
        for arm in arms:
            path = _resolve_scores(arm_dirs, arm, split)
            scores[split][arm] = _load_scores(path, gold)
            manifest = _manifest(Path(str(path) + ".manifest.json"))
            provenance["splits"][split]["arms"][arm] = {
                "path": str(path),
                "sha256": _sha256(path),
                "manifest": manifest,
            }
        manifests = [provenance["splits"][split]["arms"][arm]["manifest"] for arm in arms]
        present = [manifest for manifest in manifests if manifest is not None]
        if present:
            reference = present[0]
            for manifest in present[1:]:
                for key in ("model_id", "model_revision", "dataset_revision", "split", "offsets", "rots", "frame_sampling", "shard", "nshards", "adapter_merge_mode"):
                    if manifest.get(key) != reference.get(key):
                        raise ValueError(f"{split}: manifest {key} differs across arms")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "arms": arms,
        "cancelled_arms": list(args.cancelled_arm),
        "splits": list(SPLITS),
        "view_protocol": [{"offset": offset, "perm": perm} for offset, perm in EXPECTED_VIEWS],
        "bootstrap": {"seed": args.bootstrap_seed, "reps": args.bootstrap_reps, "unit": "video"},
        "provenance": provenance,
        "metrics": {},
        "slices": {},
        "transitions": {},
        "video_bootstrap": {},
        "gate": {"primary_valT_vs_A0": {}, "category_guard_vs_A0": {}},
    }
    for split_index, split in enumerate(SPLITS):
        summary["metrics"][split] = {arm: _metrics(scores[split][arm].values()) for arm in arms}
        summary["slices"][split] = {}
        for field in ("task", "category", "source"):
            values = sorted({row[field] for row in scores[split]["A0"].values()})
            summary["slices"][split][field] = {
                value: {arm: _metrics(row for row in scores[split][arm].values() if row[field] == value) for arm in arms}
                for value in values
            }
        summary["transitions"][split] = {}
        summary["video_bootstrap"][split] = {}
        for candidate in arms[1:]:
            for reference in arms[: arms.index(candidate)]:
                pair = f"{candidate}_vs_{reference}"
                summary["transitions"][split][pair] = _transition(scores[split][candidate], scores[split][reference])
                summary["video_bootstrap"][split][pair] = _video_bootstrap(
                    scores[split][candidate], scores[split][reference],
                    args.bootstrap_seed + split_index * 100 + len(summary["video_bootstrap"][split]),
                    args.bootstrap_reps,
                )
    for candidate in arms[1:]:
        transition = summary["transitions"]["valT"].get(f"{candidate}_vs_A0")
        interval = summary["video_bootstrap"]["valT"].get(f"{candidate}_vs_A0")
        if transition is not None and interval is not None:
            summary["gate"]["primary_valT_vs_A0"][candidate] = {
                "accuracy_delta": transition["accuracy_delta"],
                "screening_pass_point_estimate": transition["accuracy_delta"] > 0,
                "video_cluster_interval": interval,
                "strong_positive_evidence": interval["lower_95"] > 0,
                "strong_any_evidence": interval["lower_95"] > 0 or interval["upper_95"] < 0,
            }
        guard = {}
        for name, grouped in summary["slices"]["valT"]["category"].items():
            if grouped[candidate]["n"] >= 50:
                delta = grouped[candidate]["accuracy"] - grouped["A0"]["accuracy"]
                guard[name] = {
                    "n": grouped[candidate]["n"],
                    "accuracy_delta": delta,
                    "within_minus_1pp_guard": delta >= -0.01,
                }
        summary["gate"]["category_guard_vs_A0"][candidate] = {
            "passed": all(item["within_minus_1pp_guard"] for item in guard.values()),
            "categories": guard,
        }
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_report(summary, arms) + "\n")
    rows = [row for split in SPLITS for row in _csv_rows(scores, split, arms)]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, md_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True, metavar="NAME=DIR")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--cancelled-arm", action="append", default=[], help="arm name to record as cancelled without a score artifact")
    args = parser.parse_args()
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be positive")
    paths = analyze(args)
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
