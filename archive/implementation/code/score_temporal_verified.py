"""Score a bounded temporal window review arm against saved T0/T1 evidence.

The default command is validation-only.  ``--run`` is an explicit opt-in for a
future GPU run after human review has filled the manifest.  Human notes never
enter the answer prompt; only a reviewed numeric window changes frame indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import score_temporal as S

SCHEMA_VERSION = 1
REVIEWED_ARM = "T1R"
SUBSET_COUNTS = {
    "tranche": {"valQ": 2, "valT": 2},
    "full": {"valQ": 12, "valT": 12},
}
REVIEW_STATUSES = {"unreviewed", "reviewed", "unknown", "ambiguous"}
BOUNDARY_STATUSES = REVIEW_STATUSES
IDENTITY_STATUSES = {"unreviewed", "confirmed", "unknown", "ambiguous"}
ONSET_CENSORING_STATUSES = {"none", "opening_censored", "unknown", "ambiguous"}
DIGEST_FIELDS = (
    "split",
    "qa_id",
    "video_path",
    "review_role",
    "event_form",
    "review_status",
    "boundary_status",
    "identity_status",
    "onset_censoring",
    "window_start_s",
    "window_end_s",
    "reviewer",
    "reviewed_at",
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or qa_id in rows:
            raise ValueError(f"duplicate or missing qa_id at {path}:{line_number}")
        rows[qa_id] = row
    return rows


def _nullable_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number or null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number or null") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number or null")
    return number


def _is_qualified_record(record: dict) -> bool:
    return (
        record.get("review_status") == "reviewed"
        and record.get("boundary_status") == "reviewed"
        and record.get("identity_status") == "confirmed"
        and record.get("onset_censoring") == "none"
    )


def _load_records(manifest_path: Path, subset: str) -> tuple[dict, list[dict]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema_version at {manifest_path}")
    if manifest.get("dataset_revision") != C.DATASET_REVISION:
        raise ValueError("verified-window manifest does not match the pinned dataset revision")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest records must be a non-empty list")
    subsets = manifest.get("subsets")
    if not isinstance(subsets, dict):
        raise ValueError("manifest must contain a subsets object")
    subset_ids = subsets.get(subset)
    if not isinstance(subset_ids, list) or not subset_ids:
        raise ValueError(f"manifest must contain a non-empty subsets.{subset} list")
    if len(subset_ids) != len(set(subset_ids)):
        raise ValueError(f"subsets.{subset} contains duplicate QA IDs")
    records_by_id = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("qa_id"), str):
            raise ValueError("each manifest record must contain a string qa_id")
        qa_id = record["qa_id"]
        if qa_id in records_by_id:
            raise ValueError(f"duplicate manifest record {qa_id}")
        records_by_id[qa_id] = record
        review_status = record.get("review_status")
        boundary_status = record.get("boundary_status")
        identity_status = record.get("identity_status")
        onset_censoring = record.get("onset_censoring")
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"{qa_id}: invalid review_status")
        if boundary_status not in BOUNDARY_STATUSES:
            raise ValueError(f"{qa_id}: invalid boundary_status")
        if identity_status not in IDENTITY_STATUSES:
            raise ValueError(f"{qa_id}: invalid identity_status")
        if onset_censoring not in ONSET_CENSORING_STATUSES:
            raise ValueError(f"{qa_id}: invalid onset_censoring")
        start = _nullable_number(record.get("window_start_s"), f"{qa_id}.window_start_s")
        end = _nullable_number(record.get("window_end_s"), f"{qa_id}.window_end_s")
        if not _is_qualified_record(record):
            if start is not None or end is not None:
                raise ValueError(
                    f"{qa_id}: non-reviewed or non-confirmed records must leave both window boundaries null"
                )
        elif start is None or end is None or end <= start:
            raise ValueError(f"{qa_id}: reviewed windows require finite start/end with end > start")
        if _is_qualified_record(record):
            for field in ("reviewer", "reviewed_at", "boundary_evidence", "identity_evidence"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"{qa_id}: confirmed review requires non-empty {field}")
    missing = set(subset_ids) - records_by_id.keys()
    if missing:
        raise ValueError(f"subsets.{subset} references missing records: {sorted(missing)}")
    selected = [records_by_id[qa_id] for qa_id in subset_ids]
    expected_counts = SUBSET_COUNTS[subset]
    by_split = {split: [r for r in selected if r.get("split") == split] for split in expected_counts}
    if sum(len(rows) for rows in by_split.values()) != len(selected):
        raise ValueError(f"{subset} contains records outside valQ/valT")
    for split, expected in expected_counts.items():
        if len(by_split[split]) != expected:
            raise ValueError(f"{subset}/{split}: expected {expected} records, found {len(by_split[split])}")
        roles = {role: sum(r.get("review_role") == role for r in by_split[split]) for role in ("baseline_error", "correct_control")}
        if subset == "full" and roles != {"baseline_error": 6, "correct_control": 6}:
            raise ValueError(f"{subset}/{split}: expected six errors and six controls, found {roles}")
        if subset == "tranche" and roles != {"baseline_error": 1, "correct_control": 1}:
            raise ValueError(f"{subset}/{split}: expected one error and one control, found {roles}")
    return manifest, selected


def _question_rows(question_set_path: Path) -> dict[str, dict]:
    payload = read_json(question_set_path)
    if payload.get("dataset_revision") != C.DATASET_REVISION:
        raise ValueError("question set does not match the pinned dataset revision")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"valQ", "valT"}:
        raise ValueError("question set must contain valQ and valT")
    rows = {}
    for split, values in splits.items():
        if not isinstance(values, list):
            raise ValueError(f"question set split {split} must be a list")
        for row in values:
            if row.get("task") != "Temporal Grounding":
                continue
            if row["qa_id"] in rows:
                raise ValueError(f"duplicate question ID {row['qa_id']}")
            rows[row["qa_id"]] = {**row, "split": split}
    return rows


def _annotation_digest(records: list[dict], subset: str) -> str:
    payload = {
        "subset": subset,
        "records": [{field: record.get(field) for field in DIGEST_FIELDS} for record in records],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_sources(
    selected: list[dict],
    question_rows: dict[str, dict],
    baseline_dir: Path,
    t1_dir: Path,
    video_manifest: dict,
) -> dict[str, dict[str, dict]]:
    source_rows = {}
    for split in ("valQ", "valT"):
        baseline = read_jsonl(baseline_dir / f"scores_{split}.jsonl")
        t1 = read_jsonl(t1_dir / f"scores_{split}_T1.jsonl")
        source_rows[split] = {"T0": baseline, "T1": t1}
    for record in selected:
        qa_id = record["qa_id"]
        row = question_rows.get(qa_id)
        if row is None:
            raise ValueError(f"{qa_id}: absent from the temporal question set")
        if row["split"] != record["split"]:
            raise ValueError(f"{qa_id}: manifest split does not match question set")
        if row["video_path"] != record.get("video_path"):
            raise ValueError(f"{qa_id}: manifest video_path does not match question set")
        if record.get("event_form") != "first_appearance":
            raise ValueError(f"{qa_id}: verified-window arm only accepts first_appearance records")
        if row["video_path"] not in video_manifest:
            raise ValueError(f"{qa_id}: video is absent from {record.get('video_manifest', 'video manifest')}")
        split = record["split"]
        baseline = source_rows[split]["T0"].get(qa_id)
        t1 = source_rows[split]["T1"].get(qa_id)
        if baseline is None or t1 is None:
            raise ValueError(f"{qa_id}: saved T0 and T1 rows are both required")
        expected_error = baseline["pred"] != row["correct_answer"]
        expected_role = "baseline_error" if expected_error else "correct_control"
        if record.get("review_role") != expected_role:
            raise ValueError(f"{qa_id}: review_role does not match the saved baseline prediction")
        selections = t1.get("temporal_selection_by_offset")
        if not isinstance(selections, dict) or {str(o) for o in C.DEFAULT_SCORE_OFFSETS} - selections.keys():
            raise ValueError(f"{qa_id}: saved T1 row lacks both offset selections")
        for offset in C.DEFAULT_SCORE_OFFSETS:
            selection = selections[str(offset)]
            if not isinstance(selection, dict) or not isinstance(selection.get("frame_indices"), list) or not selection["frame_indices"]:
                raise ValueError(f"{qa_id}: saved T1 selection is missing frame indices at offset {offset}")
    return source_rows


def _legacy_selection(
    row: dict,
    offset: float,
    info: dict,
    record: dict,
    *,
    fallback_reason: str | None = None,
    reviewed_bracket_s: list[float] | None = None,
    saved_t1_candidate_window_s: list[float] | None = None,
) -> dict:
    indices = C.sample_index_list(int(info["n_frames"]), float(info["fps"]), offset, mode="legacy")
    times = info.get("frame_timestamps")
    if not isinstance(times, list) or len(times) != int(info["n_frames"]):
        times = [index / max(float(info["fps"]), 1e-6) for index in range(int(info["n_frames"]))]
    return {
        "mode": "legacy-fallback-reviewed-window",
        "frame_indices": indices,
        "frame_times_s": [float(times[index]) for index in indices],
        "local_times_s": [],
        "coarse_indices": [],
        "coarse_times_s": [],
        "fallback": True,
        "confidence": "low",
        "candidate_window_s": None,
        "raw_localizer_output": "",
        "window_source": "legacy_t0_fallback",
        "reviewed_bracket_s": reviewed_bracket_s,
        "saved_t1_candidate_window_s": saved_t1_candidate_window_s,
        "derived_window_s": None,
        "review_status": record["review_status"],
        "boundary_status": record["boundary_status"],
        "identity_status": record["identity_status"],
        "onset_censoring": record["onset_censoring"],
        "fallback_reason": fallback_reason
        or "review window is unreviewed, unknown, ambiguous, or identity is unconfirmed",
        "timestamp_context": "",
    }


def _window_bounds(value: object, duration: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (start, end)) or start < 0 or end > duration or end <= start:
        return None
    return [start, end]


def _reviewed_selection(
    row: dict,
    offset: float,
    info: dict,
    record: dict,
    saved_t1_selection: dict,
) -> dict:
    times, duration = S._timelines({row["video_path"]: info}, row["video_path"])
    bracket = _window_bounds([record["window_start_s"], record["window_end_s"]], duration)
    if bracket is None:
        raise ValueError(f"{row['qa_id']}: reviewed window is outside the clip duration")
    saved_window = _window_bounds(saved_t1_selection.get("candidate_window_s"), duration)
    bracket_width = bracket[1] - bracket[0]
    minimum_width = 7 / max(float(info["fps"]), 1e-6)
    if saved_window is None:
        # A reviewed bracket is sufficient to center the shortest useful local
        # window when the saved question-only localizer produced no candidate.
        width = min(duration, max(bracket_width, minimum_width))
        window_source = "reviewed_bracket_minimum_window"
    else:
        width = saved_window[1] - saved_window[0]
        window_source = "independently_reviewed_midpoint_recentered_saved_width"
    if saved_window is not None and width < bracket_width:
        return _legacy_selection(
            row,
            offset,
            info,
            record,
            fallback_reason=(
                "saved T1 candidate window is narrower than the reviewed visibility bracket "
                f"({width:.6f}s < {bracket_width:.6f}s)"
            ),
            reviewed_bracket_s=bracket,
            saved_t1_candidate_window_s=saved_window,
        )
    if saved_window is not None and width < minimum_width:
        return _legacy_selection(
            row,
            offset,
            info,
            record,
            fallback_reason="saved T1 candidate window is too narrow to preserve safely",
            reviewed_bracket_s=bracket,
            saved_t1_candidate_window_s=saved_window,
        )
    center = sum(bracket) / 2
    start = max(0.0, min(duration - width, center - width / 2))
    end = start + width
    selection = S._select_indices(int(info["n_frames"]), float(info["fps"]), offset, times, start, end)
    selection.update(
        {
            "fallback": False,
            "confidence": "human_reviewed",
            "candidate_window_s": [start, end],
            "local_times_s": [times[index] for index in selection.get("local_indices", [])],
            "reviewed_bracket_s": bracket,
            "saved_t1_candidate_window_s": saved_window,
            "derived_window_s": [start, end],
            "window_width_preserved_s": end - start if saved_window is not None else None,
            "window_width_rule": "saved_t1_candidate" if saved_window is not None else "max(reviewed_bracket, seven_frames)",
            "coarse_times_s": [],
            "raw_localizer_output": "",
            "window_source": window_source,
            "review_status": record["review_status"],
            "boundary_status": record["boundary_status"],
            "identity_status": record["identity_status"],
            "onset_censoring": record["onset_censoring"],
            # Keep review prose out of the model input and saved timestamp context.
            "timestamp_context": "",
        }
    )
    return selection


def build_selection(
    row: dict,
    record: dict,
    offset: float,
    video_manifest: dict,
    saved_t1_selection: dict,
) -> dict:
    info = video_manifest[row["video_path"]]
    legacy = C.sample_index_list(int(info["n_frames"]), float(info["fps"]), offset, mode="legacy")
    if _is_qualified_record(record):
        if len(legacy) < C.MAX_FRAMES:
            return _legacy_selection(
                row,
                offset,
                info,
                record,
                fallback_reason=(
                    f"clip has only {len(legacy)} legacy frames; reviewed recentering requires "
                    f"{C.MAX_FRAMES} frames"
                ),
            )
        return _reviewed_selection(row, offset, info, record, saved_t1_selection)
    reason = (
        "opening-censored or otherwise unknown onset; no pre-clip onset was inferred"
        if record["onset_censoring"] != "none"
        else None
    )
    return _legacy_selection(row, offset, info, record, fallback_reason=reason)


def _manifest_for_output(manifest: dict, subset: str, selected: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "arm": REVIEWED_ARM,
        "subset": subset,
        "dataset_revision": C.DATASET_REVISION,
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REV,
        "adapter": "runs/stageA_q3vl8b_r16/final_adapter",
        "offsets": list(C.DEFAULT_SCORE_OFFSETS),
        "option_rotations": ["ABCD", "BCDA"],
        "max_frames_per_offset": C.MAX_FRAMES,
        "window_manifest_schema": manifest["schema_version"],
        "annotation_digest": _annotation_digest(selected, subset),
        "selected_qa_ids": [record["qa_id"] for record in selected],
        "human_review_text_in_prompt": False,
    }


def _preflight_recenterability(
    selected: list[dict],
    questions: dict[str, dict],
    sources: dict[str, dict[str, dict[str, dict]]],
    video_manifest: dict,
) -> tuple[int, list[dict]]:
    """Check qualified records before loading the model or decoding videos."""
    recenterable = 0
    issues = []
    for record in selected:
        if not _is_qualified_record(record):
            continue
        qa_id = record["qa_id"]
        row = questions[qa_id]
        all_offsets_valid = True
        for offset in C.DEFAULT_SCORE_OFFSETS:
            try:
                selection = build_selection(
                    row,
                    record,
                    offset,
                    video_manifest,
                    sources[record["split"]]["T1"][qa_id]["temporal_selection_by_offset"][str(offset)],
                )
            except (KeyError, TypeError, ValueError) as exc:
                selection = {"fallback": True, "fallback_reason": str(exc)}
            if selection.get("fallback"):
                all_offsets_valid = False
                issues.append(
                    {
                        "qa_id": qa_id,
                        "offset": offset,
                        "reason": selection.get("fallback_reason", "reviewed window could not be recentered"),
                    }
                )
        if all_offsets_valid:
            recenterable += 1
    return recenterable, issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--subset", choices=sorted(SUBSET_COUNTS), default="tranche")
    ap.add_argument("--question-set", type=Path, default=Path("runs/phase2_temporal_coarsefine/question_set.json"))
    ap.add_argument("--baseline-dir", type=Path, default=Path("runs/stageA_q3vl8b_r16/legacy"))
    ap.add_argument("--t1-dir", type=Path, default=Path("runs/phase2_temporal_coarsefine/full"))
    ap.add_argument("--video-manifest", type=Path, default=Path("data/manifest.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("runs/phase2_temporal_verified_window"))
    ap.add_argument("--run", action="store_true", help="explicitly load the GPU model and score T1R")
    args = ap.parse_args()

    manifest, selected = _load_records(args.manifest, args.subset)
    question_payload = read_json(args.question_set)
    if C.split_checksum(question_payload["splits"]["valQ"] + question_payload["splits"]["valT"]) != manifest.get("question_set_checksum"):
        raise ValueError("verified-window manifest question_set_checksum does not match the question set")
    questions = _question_rows(args.question_set)
    video_manifest = read_json(args.video_manifest)
    sources = _validate_sources(selected, questions, args.baseline_dir, args.t1_dir, video_manifest)
    by_split = {split: [r for r in selected if r["split"] == split] for split in ("valQ", "valT")}
    recenterable, preflight_issues = _preflight_recenterability(
        selected, questions, sources, video_manifest
    )
    plan = {
        "subset": args.subset,
        "records": len(selected),
        "splits": {split: len(rows) for split, rows in by_split.items()},
        "reviewed_windows": sum(_is_qualified_record(r) for r in selected),
        "recenterable_reviewed_windows": recenterable,
        "reviewed_window_preflight_issues": preflight_issues,
        "legacy_fallback_records": sum(not _is_qualified_record(r) for r in selected),
        "saved_t1_rows_validated": sum(len(source["T1"]) for source in sources.values()),
        "run": args.run,
    }
    if not args.run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("Validation only; no model, video decoder, or external service was invoked.")
        return 0
    required_reviewed = 4 if args.subset == "tranche" else 20
    if recenterable < required_reviewed:
        raise ValueError(
            f"refusing --run: {args.subset} requires at least {required_reviewed} recenterable reviewed windows; "
            f"found {recenterable}; preflight issues: {json.dumps(preflight_issues, sort_keys=True)}"
        )

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {split: args.output_dir / f"scores_{split}_{REVIEWED_ARM}.jsonl" for split in by_split}
    completed = {split: read_jsonl(path) if path.exists() else {} for split, path in output_paths.items()}
    for split, path in output_paths.items():
        manifest_path = C.score_manifest_path(path)
        expected = _manifest_for_output(manifest, args.subset, selected)
        if path.exists() and completed[split] and not manifest_path.exists():
            raise RuntimeError(f"score rows exist without a matching output manifest at {manifest_path}")
        if manifest_path.exists() and read_json(manifest_path) != expected:
            raise RuntimeError(f"output manifest mismatch at {manifest_path}")
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")

    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verified-window scoring")
    C.configure_attention_backend("sdpa", torch)
    model_start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, revision=C.MODEL_REV, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = C.load_peft_for_inference(model, C.work_path("runs/stageA_q3vl8b_r16/final_adapter"))
    device = torch.device("cuda")
    model.to(device).eval()
    timing = {"model_load_seconds": time.perf_counter() - model_start, "scoring_seconds": 0.0, "records": 0}
    with torch.inference_mode():
        for split in ("valQ", "valT"):
            path = output_paths[split]
            with path.open("a") as stream:
                for record in by_split[split]:
                    qa_id = record["qa_id"]
                    if qa_id in completed[split]:
                        continue
                    row = questions[qa_id]
                    selections = {
                        str(offset): build_selection(
                            row,
                            record,
                            offset,
                            video_manifest,
                            sources[split]["T1"][qa_id]["temporal_selection_by_offset"][str(offset)],
                        )
                        for offset in C.DEFAULT_SCORE_OFFSETS
                    }
                    before = time.perf_counter()
                    # T1 keeps the frozen answer prompt; only the selected frames change.
                    score = S._score_variant(processor, model, device, ltr_ids, row, selections, "T1")
                    score["arm"] = REVIEWED_ARM
                    score["window_manifest_record"] = {
                        "review_status": record["review_status"],
                        "boundary_status": record["boundary_status"],
                        "identity_status": record["identity_status"],
                        "onset_censoring": record["onset_censoring"],
                        "window_source": "independently_reviewed"
                        if any(not s["fallback"] for s in selections.values())
                        else "legacy_t0_fallback",
                    }
                    timing["scoring_seconds"] += time.perf_counter() - before
                    stream.write(json.dumps(score, separators=(",", ":")) + "\n")
                    stream.flush()
                    completed[split][qa_id] = score
                    timing["records"] += 1
    (args.output_dir / "verified_window_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
