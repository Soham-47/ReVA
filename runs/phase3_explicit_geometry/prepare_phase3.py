"""Prepare the approved Phase 3 explicit geometry pilot without model calls.

This writes only under ``runs/phase3_explicit_geometry``.  It freezes the
selection, question-only detector/SAM prompt plan, tranche, and cost gate so a
later Modal/Replicate runner cannot silently broaden the experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from geometry_relations import question_entities, relation_subtype


ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
SPLITS = ("valQ", "valT")
SELECTION_SEED = 20261017
ERROR_COUNT = 20
CONTROL_COUNT = 10
TRANCHE_ERROR_COUNT = 1
TRANCHE_CONTROL_COUNT = 1
OBSERVED_SAM3_CALL_USD = 0.018
OFFICIAL_SAM3_CALL_USD = 0.016
COST_CAP_USD = 2.50
MAX_PAID_ATTEMPTS = 138


def _read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _rows(split: str) -> list[dict[str, Any]]:
    return _read_json(ROOT / "data" / "splits" / f"{split}.json")["QA"]


def _baseline(split: str) -> dict[str, dict[str, Any]]:
    path = ROOT / "runs" / "stageA_q3vl8b_r16" / "legacy" / f"scores_{split}.jsonl"
    result = {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            result[record["qa_id"]] = record
    return result


def _hash(seed: int, *values: str) -> int:
    payload = "|".join((str(seed), *values)).encode()
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _stratified(pool: list[dict[str, Any]], count: int, *, split: str, role: str) -> list[dict[str, Any]]:
    """Select a fixed-size pool with a near-even relation-subtype spread."""

    if count > len(pool):
        raise ValueError(f"{split} {role} pool has {len(pool)} rows; need {count}")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        buckets[relation_subtype(row["question"])].append(row)
    names = sorted(buckets)
    selected: list[dict[str, Any]] = []
    # Round-robin from deterministic bucket orders. This keeps outcome role
    # fixed while avoiding a subtype-heavy convenience sample.
    for name in names:
        buckets[name].sort(key=lambda row: _hash(SELECTION_SEED, split, role, name, row["qa_id"]))
    while len(selected) < count:
        progressed = False
        for name in names:
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _phase0_valt() -> dict[str, dict[str, Any]]:
    records = [
        row
        for row in _read_json(ROOT / "results" / "phase0_temporal_geometry_review" / "review_annotations.json")["records"]
        if row.get("task") == "Geometric Relation"
    ]
    roles = {"baseline_error": ERROR_COUNT, "correct_control": CONTROL_COUNT}
    if Counter(row.get("review_role") for row in records) != Counter(roles):
        raise ValueError("Phase 0 geometry annotations are not the approved 20-error/10-control set")
    return {row["qa_id"]: row for row in records}


def _selected(split: str) -> tuple[list[dict[str, Any]], str]:
    rows = _rows(split)
    baseline = _baseline(split)
    geometry = [row for row in rows if row.get("task") == "Geometric Relation"]
    for row in geometry:
        if row["qa_id"] not in baseline:
            raise ValueError(f"baseline is missing geometry question {split}/{row['qa_id']}")
    if split == "valT":
        annotations = _phase0_valt()
        selected = [row for row in geometry if row["qa_id"] in annotations]
        selected.sort(key=lambda row: (0 if annotations[row["qa_id"]]["review_role"] == "baseline_error" else 1, row["qa_id"]))
        for row in selected:
            expected = annotations[row["qa_id"]]
            observed_role = "baseline_error" if baseline[row["qa_id"]]["pred"] != row["correct_answer"] else "correct_control"
            if observed_role != expected["review_role"]:
                raise ValueError(f"Phase 0 role mismatch for {split}/{row['qa_id']}")
        return selected, "results/phase0_temporal_geometry_review/review_annotations.json"
    errors = [row for row in geometry if baseline[row["qa_id"]]["pred"] != row["correct_answer"]]
    controls = [row for row in geometry if baseline[row["qa_id"]]["pred"] == row["correct_answer"]]
    selected_errors = _stratified(errors, ERROR_COUNT, split=split, role="baseline_error")
    selected_controls = _stratified(controls, CONTROL_COUNT, split=split, role="correct_control")
    return selected_errors + selected_controls, "deterministic SHA-256 stratification"


def _record(split: str, row: dict[str, Any], role: str, baseline: dict[str, Any]) -> dict[str, Any]:
    entities = question_entities(row["question"])
    return {
        "qa_id": row["qa_id"],
        "split": split,
        "video_path": row["video_path"],
        "task": row["task"],
        "category": row.get("category"),
        "question": row["question"],
        "relation_subtype": relation_subtype(row["question"]),
        "review_role": role,
        "baseline_correct": baseline["pred"] == row["correct_answer"],
        "baseline_prediction": baseline["pred"],
        "gold": row["correct_answer"],
        "baseline_probability": baseline["p"],
        "prompt_entities_from_question_only": entities,
        "detector_prompt": ". ".join(entities) + ("." if entities else ""),
        "sam3_prompt_entities": entities,
    }


def prepare() -> dict[str, Any]:
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, str] = {}
    for split in SPLITS:
        selected, source = _selected(split)
        baseline = _baseline(split)
        records: list[dict[str, Any]] = []
        for row in selected:
            role = "baseline_error" if baseline[row["qa_id"]]["pred"] != row["correct_answer"] else "correct_control"
            records.append(_record(split, row, role, baseline[row["qa_id"]]))
        if len(records) != ERROR_COUNT + CONTROL_COUNT:
            raise ValueError(f"{split} selected {len(records)} rows")
        if Counter(row["review_role"] for row in records) != Counter({"baseline_error": ERROR_COUNT, "correct_control": CONTROL_COUNT}):
            raise ValueError(f"{split} selection role counts are incorrect")
        selected_by_split[split] = records
        sources[split] = source

    all_records = [row for split in SPLITS for row in selected_by_split[split]]
    by_video: dict[str, dict[str, Any]] = defaultdict(lambda: {"splits": [], "qa_ids": [], "entities": []})
    for row in all_records:
        info = by_video[row["video_path"]]
        if row["split"] not in info["splits"]:
            info["splits"].append(row["split"])
        info["qa_ids"].append(row["qa_id"])
        for entity in row["prompt_entities_from_question_only"]:
            if entity not in info["entities"]:
                info["entities"].append(entity)
    video_plan = []
    for video_path in sorted(by_video):
        info = by_video[video_path]
        # One call per question entity keeps SAM3 labels unambiguous. Every
        # phrase still comes from a selected question and is recorded here.
        video_plan.append(
            {
                "video_path": video_path,
                "splits": info["splits"],
                "qa_ids": sorted(info["qa_ids"]),
                "prompt_entities_from_questions": info["entities"],
                "sam3_call_count": len(info["entities"]),
                "sam3_prompt_policy": "one call per question-entity phrase; no options, gold, or expected locations",
            }
        )

    tranche: dict[str, list[str]] = {}
    for split in SPLITS:
        records = selected_by_split[split]
        errors = [row for row in records if row["review_role"] == "baseline_error"]
        controls = [row for row in records if row["review_role"] == "correct_control"]
        tranche[split] = [
            min(errors, key=lambda row: _hash(SELECTION_SEED, "tranche", split, "error", row["qa_id"]))["qa_id"],
            min(controls, key=lambda row: _hash(SELECTION_SEED, "tranche", split, "control", row["qa_id"]))["qa_id"],
        ]

    all_pairs = {
        (row["video_path"], entity)
        for row in all_records
        for entity in row["sam3_prompt_entities"]
    }
    tranche_ids = {qa_id for ids in tranche.values() for qa_id in ids}
    tranche_pairs = {
        (row["video_path"], entity)
        for row in all_records
        if row["qa_id"] in tranche_ids
        for entity in row["sam3_prompt_entities"]
    }
    planned_calls = len(all_pairs)
    tranche_calls = len(tranche_pairs)
    remaining_calls = len(all_pairs - tranche_pairs)
    conservative_rate = max(OBSERVED_SAM3_CALL_USD, OFFICIAL_SAM3_CALL_USD)
    conservative_full_cost = round(planned_calls * conservative_rate, 6)
    conservative_tranche_cost = round(tranche_calls * conservative_rate, 6)
    cost_gate = {
        "cap_usd": COST_CAP_USD,
        "official_model_page_approx_cost_per_call_usd": OFFICIAL_SAM3_CALL_USD,
        "observed_prior_manifest_approx_cost_per_paid_call_usd": OBSERVED_SAM3_CALL_USD,
        "conservative_rate_usd_per_call": conservative_rate,
        "tranche_call_count": tranche_calls,
        "tranche_conservative_cost_usd": conservative_tranche_cost,
        "full_call_count": planned_calls,
        "full_conservative_cost_usd": conservative_full_cost,
        "remaining_call_count_after_tranche": remaining_calls,
        "remaining_conservative_cost_usd": round(remaining_calls * conservative_rate, 6),
        "full_run_within_cap": conservative_full_cost <= COST_CAP_USD,
        "max_paid_attempts_under_cap": MAX_PAID_ATTEMPTS,
        "max_paid_attempts_conservative_cost_usd": round(MAX_PAID_ATTEMPTS * conservative_rate, 6),
        "retry_headroom_attempts": MAX_PAID_ATTEMPTS - planned_calls,
        "source_files": {
            "official_model_page": "https://internal.replicate.com/lucataco/sam3-video",
            "prior_manifest": "runs/sam3_pilot_stageA_replicate/manifest.json",
        },
        "decision": "eligible_after_tranche" if conservative_full_cost <= COST_CAP_USD else "stop_after_tranche",
    }
    return {
        "schema_version": 1,
        "pilot": "phase3_explicit_geometry",
        "selection_seed": SELECTION_SEED,
        "scope": "error-enriched pilot; not split-wide benchmark accuracy",
        "conditions": {
            "G0": "frozen Stage A baseline, legacy frames, offsets 0/0.5, two rotations",
            "G1": "Stage A plus explicit 2D image-plane relation text from GroundingDINO boxes",
            "G2": "Stage A plus the same relation-text schema from SAM 3 masks",
        },
        "constraints": {
            "only_evidence_context_changes": True,
            "no_metric_distance_or_physical_depth": True,
            "fallback_to_baseline_on_missing_or_ambiguous_evidence": True,
            "prompts_use_question_entities_only": True,
            "options_gold_and_expected_locations_excluded_from_prompts": True,
            "preserve_existing_stage_a_and_prior_runs": True,
        },
        "source": {
            "dataset_revision": _read_json(ROOT / "data" / "dataset_revision.json"),
            "splits": {split: f"data/splits/{split}.json" for split in SPLITS},
            "baseline": {split: f"runs/stageA_q3vl8b_r16/legacy/scores_{split}.jsonl" for split in SPLITS},
            "selection_basis": sources,
        },
        "selected_questions": selected_by_split,
        "tranche": {"questions_per_split": 2, "ids": tranche, "composition": "one baseline error and one correct control per split"},
        "video_prompt_plan": video_plan,
        "cost_gate": cost_gate,
    }


def write_frozen_g0(payload: dict[str, Any]) -> None:
    """Copy only selected baseline records into the new run directory."""

    rows_by_id: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = ROOT / "runs" / "stageA_q3vl8b_r16" / "legacy" / f"scores_{split}.jsonl"
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["qa_id"] in {row["qa_id"] for row in payload["selected_questions"][split]}:
                    rows_by_id[record["qa_id"]] = record
    ordered = [rows_by_id[row["qa_id"]] for split in SPLITS for row in payload["selected_questions"][split]]
    with (RUN / "g0_baseline.jsonl").open("w") as stream:
        for record in ordered:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    (RUN / "g0_baseline.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "condition": "G0",
                "status": "frozen_copy",
                "source": "runs/stageA_q3vl8b_r16/legacy/scores_{valQ,valT}.jsonl",
                "question_count": len(ordered),
                "offsets": [0.0, 0.5],
                "rotations": 2,
                "frame_sampling": "legacy",
                "adapter": "runs/stageA_q3vl8b_r16/final_adapter",
                "evidence_context": None,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> int:
    payload = prepare()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "selection_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (RUN / "cost_gate.json").write_text(json.dumps(payload["cost_gate"], indent=2) + "\n")
    write_frozen_g0(payload)
    print(json.dumps({"selected": {split: len(rows) for split, rows in payload["selected_questions"].items()}, "tranche": payload["tranche"], "cost_gate": payload["cost_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
