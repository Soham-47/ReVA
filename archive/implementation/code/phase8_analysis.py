"""Analyze complete Phase 8 candidate scores against an A0 reference.

The utility is deliberately independent of model loading.  It validates each
score file against its own manifest, requires the frozen four-view protocol and
complete valQ/valT coverage, and writes paired per-question records together
with deterministic video-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import common as C


SPLITS = ("valQ", "valT")
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


def _gold(work: Path, split: str) -> dict[str, dict[str, Any]]:
    path = work / "data" / "splits" / f"{split}.json"
    rows = _json(path)["QA"]
    result = {row["qa_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{split}: duplicate QA IDs")
    return result


def _probabilities(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label}: expected four probabilities")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in result):
        raise ValueError(f"{label}: probability outside [0, 1]")
    if abs(sum(result) - 1.0) > 1e-4:
        raise ValueError(f"{label}: probabilities do not sum to one")
    return result


def _manifest(path: Path, split: str) -> dict[str, Any]:
    sidecar = Path(str(path) + ".manifest.json")
    if not sidecar.exists():
        raise ValueError(f"{path}: missing score manifest {sidecar}")
    payload = _json(sidecar)
    if not isinstance(payload, dict):
        raise ValueError(f"{sidecar}: manifest must be an object")
    checks = {
        "split": split,
        "offsets": [0.0, 0.5],
        "rots": 2,
        # The original Stage A manifest predates this explicit field; its
        # fixed sampler was the same legacy policy used by Phase 8.
        "frame_sampling": "legacy",
    }
    for key, expected in checks.items():
        actual = payload.get(key, expected) if key == "frame_sampling" else payload.get(key)
        if actual != expected:
            raise ValueError(f"{sidecar}: {key} must be {expected!r}")
    if payload.get("view_count", 4) != 4:
        raise ValueError(f"{sidecar}: view_count must be 4")
    if payload.get("frames_per_view", 16) != 16:
        raise ValueError(f"{sidecar}: frames_per_view must be 16")
    labels = payload.get("answer_labels")
    if labels is not None and (labels.get("letters") != list(C.LETTERS) or labels.get("token_count") != 1):
        raise ValueError(f"{sidecar}: answer_labels must describe one-token A-D scoring")
    return payload


def _resolve_score(path_value: str | os.PathLike[str], split: str, work: Path) -> Path:
    raw = Path(path_value).expanduser()
    if not raw.is_absolute():
        raw = work / raw
    if raw.is_file():
        return raw
    if "{split}" in str(raw):
        candidate = Path(str(raw).replace("{split}", split))
        if candidate.is_file():
            return candidate
    if not raw.is_dir():
        raise FileNotFoundError(f"score path does not exist: {raw}")
    direct = raw / f"scores_{split}.jsonl"
    if direct.is_file():
        return direct
    matches = sorted(raw.glob(f"scores_*_{split}.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no {split} score file under {raw}")
    raise ValueError(f"ambiguous {split} score files under {raw}; pass an explicit file or {{split}} path")


def _score_row(score: dict[str, Any], question: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    qa_id = score.get("qa_id")
    p = _probabilities(score.get("p"), f"{path}:{line_number}: p")
    pred = score.get("pred")
    if pred not in C.LETTERS or pred != C.LETTERS[max(range(4), key=p.__getitem__)]:
        raise ValueError(f"{path}:{line_number}: invalid or inconsistent prediction")
    views = score.get("views")
    if not isinstance(views, list) or len(views) != len(EXPECTED_VIEWS):
        raise ValueError(f"{path}:{line_number}: expected complete four-view predictions")
    for view_number, (view, expected) in enumerate(zip(views, EXPECTED_VIEWS), 1):
        signature = (float(view.get("offset")), str(view.get("perm")))
        if not math.isclose(signature[0], expected[0], abs_tol=1e-9) or signature[1] != expected[1]:
            raise ValueError(f"{path}:{line_number}: unexpected view {view_number}: {signature!r}")
        _probabilities(view.get("p"), f"{path}:{line_number}: view {view_number}")
    top = sorted(p, reverse=True)
    gold = question["correct_answer"]
    gold_index = C.LETTERS.index(gold)
    return {
        "qa_id": qa_id,
        "video": question["video_path"],
        "task": question["task"],
        "category": question["category"],
        "source": question.get("dataset_name", ""),
        "question": question["question"],
        "gold": gold,
        "pred": pred,
        "ok": pred == gold,
        "p_gold": p[gold_index],
        "confidence": top[0],
        "margin": top[0] - top[1],
        "views": views,
    }


def _load_scores(path: Path, gold: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
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
            rows[qa_id] = _score_row(score, gold[qa_id], path, line_number)
    if set(rows) != set(gold):
        missing = sorted(set(gold) - set(rows))[:5]
        extra = sorted(set(rows) - set(gold))[:5]
        raise ValueError(f"{path}: scores are incomplete; missing={missing}, extra={extra}")
    return rows, {}


def _load_arm(path: Path, gold: dict[str, dict[str, Any]], split: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = _manifest(path, split)
    rows, _ = _load_scores(path, gold)
    return rows, {
        "path": str(path),
        "sha256": _sha256(path),
        "manifest": manifest,
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "manifest_path": str(Path(str(path) + ".manifest.json")),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "n": len(rows),
        "correct": sum(bool(row["ok"]) for row in rows),
        "accuracy": _mean(float(row["ok"]) for row in rows),
        "confidence_mean": _mean(row["confidence"] for row in rows),
        "margin_mean": _mean(row["margin"] for row in rows),
        "p_gold_mean": _mean(row["p_gold"] for row in rows),
    }


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def _video_bootstrap(
    candidate: Mapping[str, dict[str, Any]],
    reference: Mapping[str, dict[str, Any]],
    *,
    seed: int,
    reps: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for qa_id, row in candidate.items():
        grouped[row["video"]].append(float(row["ok"]) - float(reference[qa_id]["ok"]))
    videos = sorted(grouped)
    if not videos:
        raise ValueError("cannot bootstrap an empty score set")
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        selected = [videos[rng.randrange(len(videos))] for _ in videos]
        samples.append(_mean(delta for video in selected for delta in grouped[video]))
    samples.sort()
    observed = _mean(float(candidate[qa_id]["ok"]) - float(reference[qa_id]["ok"]) for qa_id in candidate)
    return {
        "estimate": observed,
        "lower_95": _percentile(samples, 0.025),
        "upper_95": _percentile(samples, 0.975),
        "videos": len(videos),
        "reps": reps,
        "seed": seed,
    }


def _paired_records(candidate: Mapping[str, dict[str, Any]], reference: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for qa_id in sorted(candidate):
        c, r = candidate[qa_id], reference[qa_id]
        if c["ok"] and r["ok"]:
            state = "both_correct"
        elif c["ok"]:
            state = "fix"
        elif r["ok"]:
            state = "regression"
        else:
            state = "both_wrong"
        result.append(
            {
                "qa_id": qa_id,
                "video": c["video"],
                "task": c["task"],
                "category": c["category"],
                "gold": c["gold"],
                "candidate": {"pred": c["pred"], "ok": c["ok"], "confidence": c["confidence"], "margin": c["margin"]},
                "a0": {"pred": r["pred"], "ok": r["ok"], "confidence": r["confidence"], "margin": r["margin"]},
                "state": state,
                "confidence_delta": c["confidence"] - r["confidence"],
                "margin_delta": c["margin"] - r["margin"],
            }
        )
    return result


def _pair_metrics(records: list[dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    counts = {state: sum(record["state"] == state for record in records) for state in ("fix", "regression", "both_correct", "both_wrong")}
    disagreements = [record for record in records if record["candidate"]["pred"] != record["a0"]["pred"]]
    return {
        **counts,
        "n": len(records),
        "candidate_disagreements": len(disagreements),
        "candidate_disagreement_qa_ids": [record["qa_id"] for record in disagreements],
        "confidence_delta_mean": _mean(record["confidence_delta"] for record in records),
        "margin_delta_mean": _mean(record["margin_delta"] for record in records),
        "accuracy_delta": _mean(float(record["candidate"]["ok"]) - float(record["a0"]["ok"]) for record in records),
        "video_cluster_bootstrap": bootstrap,
    }


def _parse_arm(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=DIR_OR_FILE")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=DIR_OR_FILE")
    return name, path


def _csv_rows(summary: dict[str, Any], arms: list[str]) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        for candidate in arms[1:]:
            for record in summary["paired_records"][split][candidate]:
                rows.append(
                    {
                        "split": split,
                        "candidate": candidate,
                        "qa_id": record["qa_id"],
                        "video": record["video"],
                        "task": record["task"],
                        "category": record["category"],
                        "gold": record["gold"],
                        "a0_pred": record["a0"]["pred"],
                        "candidate_pred": record["candidate"]["pred"],
                        "state": record["state"],
                        "confidence_delta": record["confidence_delta"],
                        "margin_delta": record["margin_delta"],
                        "task_a0_n": summary["slices"][split]["task"][record["task"]]["A0"]["n"],
                        "task_candidate_n": summary["slices"][split]["task"][record["task"]][candidate]["n"],
                        "category_a0_n": summary["slices"][split]["category"][record["category"]]["A0"]["n"],
                        "category_candidate_n": summary["slices"][split]["category"][record["category"]][candidate]["n"],
                    }
                )
    return rows


def _accuracy_delta_svg(summary: dict[str, Any], arms: list[str], path: Path) -> None:
    """Write a dependency-free task accuracy delta chart with denominators."""

    entries: list[dict[str, Any]] = []
    for split in SPLITS:
        for task, grouped in summary["slices"][split]["task"].items():
            for candidate in arms[1:]:
                a0 = grouped["A0"]
                arm = grouped[candidate]
                entries.append({
                    "split": split,
                    "task": task,
                    "candidate": candidate,
                    "delta": float(arm["accuracy"] - a0["accuracy"]),
                    "a0_n": int(a0["n"]),
                    "candidate_n": int(arm["n"]),
                })
    width = 1200
    row_height = 30
    margin_left = 390
    margin_right = 70
    margin_top = 48
    height = margin_top + row_height * (len(entries) + 1) + 24
    plot_width = width - margin_left - margin_right
    zero_x = margin_left + plot_width // 2
    scale = plot_width / 2.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family: sans-serif;font-size:13px} .axis{stroke:#555} .positive{fill:#2a9d55} .negative{fill:#d55e00}</style>',
        '<text x="20" y="25" font-size="18">Phase 8 candidate minus A0 task accuracy</text>',
        f'<line class="axis" x1="{zero_x}" y1="{margin_top - 12}" x2="{zero_x}" y2="{height - 16}"/>',
        f'<text x="{zero_x - 30}" y="{margin_top - 20}">0.00</text>',
    ]
    for index, entry in enumerate(entries):
        y = margin_top + index * row_height
        label = html.escape(f'{entry["split"]} / {entry["candidate"]} / {entry["task"]}')
        delta = entry["delta"]
        bar_width = abs(delta) * scale
        x = zero_x if delta >= 0 else zero_x - bar_width
        class_name = "positive" if delta >= 0 else "negative"
        parts.append(
            f'<text x="10" y="{y + 18}">{label}</text>'
            f'<rect class="{class_name}" x="{x:.2f}" y="{y + 5}" width="{bar_width:.2f}" height="20" '
            f'data-split="{html.escape(entry["split"])}" data-candidate="{html.escape(entry["candidate"])}" '
            f'data-task="{html.escape(entry["task"])}" data-n-a0="{entry["a0_n"]}" '
            f'data-n-candidate="{entry["candidate_n"]}" data-delta="{delta:.8f}"/>'
            f'<text x="{width - margin_right + 8}" y="{y + 18}">{delta:+.3f}</text>'
        )
    parts.append("</svg>\n")
    path.write_text("".join(parts))


def _report(summary: dict[str, Any], arms: list[str]) -> str:
    lines = ["# Phase 8 paired analysis", "", "## Model provenance", ""]
    for split in SPLITS:
        for arm in arms:
            record = summary["provenance"][split][arm]
            manifest = record["manifest"]
            lines.append(f"- {split} / {arm}: `{manifest.get('model_id')}` @ `{manifest.get('model_revision')}`")
    lines.extend(["", "## Overall accuracy", ""])
    for split in SPLITS:
        lines.append(f"### {split}")
        for arm in arms:
            metric = summary["metrics"][split][arm]
            lines.append(f"- {arm}: {metric['correct']}/{metric['n']} ({metric['accuracy']:.4f})")
        lines.append("")
    lines.extend(["## Candidate transitions versus A0", ""])
    for split in SPLITS:
        lines.append(f"### {split}")
        for candidate in arms[1:]:
            pair = summary["paired"][split][candidate]
            interval = pair["video_cluster_bootstrap"]
            lines.append(
                f"- {candidate}: fixes={pair['fix']}, regressions={pair['regression']}, "
                f"both_correct={pair['both_correct']}, both_wrong={pair['both_wrong']}, "
                f"disagreements={pair['candidate_disagreements']}, "
                f"accuracy_delta={pair['accuracy_delta']:+.4f}, "
                f"confidence_delta={pair['confidence_delta_mean']:+.4f}, "
                f"margin_delta={pair['margin_delta_mean']:+.4f}, "
                f"video_cluster_95%=[{interval['lower_95']:+.4f}, {interval['upper_95']:+.4f}]"
            )
        lines.append("")
    lines.extend(["## Task/category slices", ""])
    for split in SPLITS:
        lines.append(f"### {split}")
        for field in ("task", "category"):
            lines.append(f"#### {field}")
            for value, grouped in summary["slices"][split][field].items():
                values = ", ".join(f"{arm}={grouped[arm]['correct']}/{grouped[arm]['n']} ({grouped[arm]['accuracy']:.4f})" for arm in arms)
                lines.append(f"- {value}: {values}")
        lines.append("")
    return "\n".join(lines)


def analyze(
    *,
    a0: str | os.PathLike[str],
    candidates: Mapping[str, str | os.PathLike[str]],
    output_prefix: str | os.PathLike[str],
    work: str | os.PathLike[str],
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[Path, Path, Path]:
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    if not candidates:
        raise ValueError("at least one candidate is required")
    if "A0" in candidates:
        raise ValueError("candidate name A0 is reserved for the reference")
    work_path = Path(work).expanduser().resolve()
    arms = ["A0", *candidates]
    all_scores: dict[str, dict[str, dict[str, Any]]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        gold = _gold(work_path, split)
        all_scores[split] = {}
        provenance[split] = {}
        arm_paths = {"A0": a0, **candidates}
        for arm, root in arm_paths.items():
            path = _resolve_score(root, split, work_path)
            rows, source = _load_arm(path, gold, split)
            all_scores[split][arm] = rows
            provenance[split][arm] = source

    summary: dict[str, Any] = {
        "schema_version": 1,
        "output_kind": "phase8_paired_analysis",
        "splits": list(SPLITS),
        "arms": arms,
        "bootstrap": {"unit": "video", "reps": bootstrap_reps, "seed": bootstrap_seed},
        "provenance": provenance,
        "metrics": {},
        "slices": {},
        "paired": {},
        "paired_records": {},
    }
    for split_index, split in enumerate(SPLITS):
        summary["metrics"][split] = {arm: _metrics(all_scores[split][arm].values()) for arm in arms}
        summary["slices"][split] = {}
        for field in ("task", "category"):
            values = sorted({row[field] for row in all_scores[split]["A0"].values()})
            summary["slices"][split][field] = {
                value: {arm: _metrics(row for row in all_scores[split][arm].values() if row[field] == value) for arm in arms}
                for value in values
            }
        summary["paired"][split] = {}
        summary["paired_records"][split] = {}
        for candidate_index, candidate in enumerate(arms[1:]):
            records = _paired_records(all_scores[split][candidate], all_scores[split]["A0"])
            bootstrap = _video_bootstrap(
                all_scores[split][candidate],
                all_scores[split]["A0"],
                seed=bootstrap_seed + split_index * 100 + candidate_index,
                reps=bootstrap_reps,
            )
            summary["paired_records"][split][candidate] = records
            summary["paired"][split][candidate] = _pair_metrics(records, bootstrap)

    prefix = Path(output_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = work_path / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    csv_path = prefix.with_suffix(".csv")
    svg_path = prefix.with_suffix(".svg")
    summary["visualization"] = {
        "accuracy_delta_svg": str(svg_path),
        "description": "candidate minus A0 task accuracy for valQ and valT",
        "denominators": "SVG data-n-a0/data-n-candidate; CSV task_*_n; JSON slices.*.task",
    }
    _accuracy_delta_svg(summary, arms, svg_path)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_report(summary, arms) + "\n")
    rows = _csv_rows(summary, arms)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, md_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default=os.environ.get("WORK"), required=False)
    parser.add_argument("--a0", required=True, help="A0 score file or run directory")
    parser.add_argument("--candidate", action="append", type=_parse_arm, required=True, metavar="NAME=DIR_OR_FILE")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    if not args.work:
        parser.error("--work or WORK is required")
    try:
        paths = analyze(
            a0=args.a0,
            candidates=dict(args.candidate),
            output_prefix=args.output_prefix,
            work=args.work,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=args.bootstrap_seed,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

