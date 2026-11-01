"""Compare a fine-tuned Phase 8 arm on the frozen 110-question pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_analysis as A


ARMS = ("stageA", "zero_shot", "fine_tuned")
SLICES = ("task", "category", "source")


def _selection(path: Path, work: Path) -> tuple[dict[str, set[str]], str]:
    payload = json.loads(path.read_text())
    split_ids: dict[str, set[str]] = {}
    for split in A.SPLITS:
        values = payload.get("splits", {}).get(split, {}).get("ids")
        if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
            raise ValueError(f"selection must contain non-empty splits.{split}.ids")
        if len(values) != len(set(values)):
            raise ValueError(f"selection contains duplicate IDs in {split}")
        split_ids[split] = set(values)
    if sum(len(values) for values in split_ids.values()) != 110:
        raise ValueError("frozen pilot selection must contain exactly 110 IDs")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return split_ids, digest


def _load_selected(path: str, split: str, work: Path, ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    score_path = A._resolve_score(path, split, work)
    gold = A._gold(work, split)
    selected_gold = {qa_id: gold[qa_id] for qa_id in ids if qa_id in gold}
    if set(selected_gold) != ids:
        missing = sorted(ids - set(selected_gold))
        raise ValueError(f"{split}: selection IDs absent from gold split: {missing[:5]}")
    return A._load_arm(score_path, selected_gold, split)


def _slice_metrics(rows: dict[str, dict[str, Any]], dimension: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        groups.setdefault(str(row[dimension]), []).append(row)
    return {key: A._metrics(values) for key, values in sorted(groups.items())}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.8 fine-tune frozen pilot comparison",
        "",
        f"Selection SHA-256: `{summary['selection']['sha256']}`; total questions: {summary['selection']['total_questions']}.",
        "",
        "## Accuracy",
        "",
        "| Split | Stage A | Zero-shot Qwen3.8 | Fine-tuned Qwen3.8 | Fine-tuned minus Stage A | Fine-tuned minus zero-shot |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in A.SPLITS:
        metrics = summary["metrics"][split]
        stage = metrics["stageA"]["accuracy"]
        zero = metrics["zero_shot"]["accuracy"]
        fine = metrics["fine_tuned"]["accuracy"]
        lines.append(
            f"| {split} | {metrics['stageA']['correct']}/{metrics['stageA']['n']} ({stage:.3f}) | "
            f"{metrics['zero_shot']['correct']}/{metrics['zero_shot']['n']} ({zero:.3f}) | "
            f"{metrics['fine_tuned']['correct']}/{metrics['fine_tuned']['n']} ({fine:.3f}) | "
            f"{fine - stage:+.3f} | {fine - zero:+.3f} |"
        )
    lines.extend(["", "## Paired changes", ""])
    for split in A.SPLITS:
        lines.append(f"### {split}")
        lines.append("")
        for reference in ("stageA", "zero_shot"):
            pair = summary["paired"][split][reference]
            boot = pair["video_cluster_bootstrap"]
            lines.append(
                f"* Fine-tuned vs {reference}: fixes {pair['fix']}, regressions {pair['regression']}, "
                f"both correct {pair['both_correct']}, both wrong {pair['both_wrong']}; "
                f"delta {pair['accuracy_delta']:+.3f}, clustered 95% interval "
                f"[{boot['lower_95']:+.3f}, {boot['upper_95']:+.3f}]."
            )
        lines.append("")
    lines.extend(["## Slice accuracy", ""])
    for split in A.SPLITS:
        lines.append(f"### {split}")
        lines.append("")
        for dimension in SLICES:
            lines.append(f"#### {dimension}")
            lines.append("")
            lines.append("| Slice | Stage A | Zero-shot | Fine-tuned |")
            lines.append("|---|---:|---:|---:|")
            keys = sorted(summary["slices"][split][dimension])
            for key in keys:
                values = summary["slices"][split][dimension][key]
                lines.append(
                    f"| {key} | {values['stageA']['correct']}/{values['stageA']['n']} ({values['stageA']['accuracy']:.3f}) | "
                    f"{values['zero_shot']['correct']}/{values['zero_shot']['n']} ({values['zero_shot']['accuracy']:.3f}) | "
                    f"{values['fine_tuned']['correct']}/{values['fine_tuned']['n']} ({values['fine_tuned']['accuracy']:.3f}) |"
                )
            lines.append("")
    lines.extend(["## Runtime and provenance", ""])
    lines.append("Runtime telemetry is retained in each score arm's `.summary.json` sidecar.")
    lines.append("")
    lines.append("This 110-question selection is a development pilot and does not support a generalization claim.")
    return "\n".join(lines) + "\n"


def compare(*, work: Path, selection_path: Path, stage_a: str, zero_shot: str, fine_tuned: str, output_prefix: Path, bootstrap_reps: int, bootstrap_seed: int) -> tuple[Path, Path, Path]:
    split_ids, selection_digest = _selection(selection_path, work)
    paths = {"stageA": stage_a, "zero_shot": zero_shot, "fine_tuned": fine_tuned}
    rows: dict[str, dict[str, dict[str, Any]]] = {split: {} for split in A.SPLITS}
    provenance: dict[str, dict[str, Any]] = {split: {} for split in A.SPLITS}
    metrics: dict[str, dict[str, Any]] = {}
    slices: dict[str, dict[str, dict[str, Any]]] = {}
    paired: dict[str, dict[str, Any]] = {}
    paired_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split in A.SPLITS:
        split_rows: dict[str, dict[str, Any]] = {}
        split_provenance: dict[str, Any] = {}
        for arm, path in paths.items():
            arm_rows, arm_provenance = _load_selected(path, split, work, split_ids[split])
            split_rows[arm] = arm_rows
            split_provenance[arm] = arm_provenance
        rows[split] = split_rows
        provenance[split] = split_provenance
        metrics[split] = {arm: A._metrics(split_rows[arm].values()) for arm in ARMS}
        slices[split] = {
            dimension: {
                key: {arm: _slice_metrics(split_rows[arm], dimension)[key] for arm in ARMS}
                for key in sorted(set().union(*(_slice_metrics(split_rows[arm], dimension) for arm in ARMS)))
            }
            for dimension in SLICES
        }
        paired[split] = {}
        paired_records[split] = {}
        for reference in ("stageA", "zero_shot"):
            records = A._paired_records(split_rows["fine_tuned"], split_rows[reference])
            bootstrap = A._video_bootstrap(
                split_rows["fine_tuned"],
                split_rows[reference],
                seed=bootstrap_seed,
                reps=bootstrap_reps,
            )
            paired[split][reference] = A._pair_metrics(records, bootstrap)
            paired_records[split][reference] = records
    summary = {
        "schema_version": 1,
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": selection_digest,
            "ids_per_split": {split: len(ids) for split, ids in split_ids.items()},
            "total_questions": sum(len(ids) for ids in split_ids.values()),
        },
        "arms": list(ARMS),
        "metrics": metrics,
        "slices": slices,
        "paired": paired,
        "paired_records": paired_records,
        "provenance": provenance,
        "bootstrap": {"reps": bootstrap_reps, "seed": bootstrap_seed, "cluster": "video"},
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(summary))
    with csv_path.open("w", newline="") as stream:
        fieldnames = ["split", "reference", "qa_id", "video", "task", "category", "gold", "state", "stageA_pred", "reference_pred", "fine_tuned_pred", "confidence_delta", "margin_delta"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for split in A.SPLITS:
            for reference in ("stageA", "zero_shot"):
                for record in paired_records[split][reference]:
                    writer.writerow({
                        "split": split,
                        "reference": reference,
                        "qa_id": record["qa_id"],
                        "video": record["video"],
                        "task": record["task"],
                        "category": record["category"],
                        "gold": record["gold"],
                        "state": record["state"],
                        "stageA_pred": rows[split]["stageA"][record["qa_id"]]["pred"],
                        "reference_pred": record["a0"]["pred"],
                        "fine_tuned_pred": record["candidate"]["pred"],
                        "confidence_delta": record["confidence_delta"],
                        "margin_delta": record["margin_delta"],
                    })
    return json_path, markdown_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--zero-shot", required=True)
    parser.add_argument("--fine-tuned", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    args = parser.parse_args(argv)
    paths = compare(
        work=args.work.resolve(),
        selection_path=args.selection.resolve(),
        stage_a=args.stage_a,
        zero_shot=args.zero_shot,
        fine_tuned=args.fine_tuned,
        output_prefix=args.output_prefix.resolve(),
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
