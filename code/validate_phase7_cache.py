"""Validate reusable Grounding DINO caches against the Phase 7 frame contract.

This is an offline audit.  It does not decode video, run a detector, score a
question, or treat a cache checksum as a source-file checksum.  The review
packet supplies the fixed cohort; Phase 3 G1 sidecars supply the only trusted
video-path-to-cache mapping available in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "runs" / "phase3_explicit_geometry"))

import common as C
import grounding as G
from geometry_relations import question_entities
from phase7_manifest import TIMESTAMP_TOLERANCE, validate_manifest


SCHEMA_VERSION = 1
KIND = "reva-phase7b-cache-validation"
EXPECTED_THRESHOLDS = {
    "box_threshold": G.DEFAULT_DETECTOR_THRESHOLD,
    "text_threshold": G.DEFAULT_DETECTOR_TEXT_THRESHOLD,
}
EXPECTED_TRACKER = {
    "algorithm": "adjacent-greedy-iou",
    "iou_threshold": G.DEFAULT_TRACK_IOU_THRESHOLD,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_labels(prompt: str) -> set[str]:
    return {part.strip().casefold() for part in str(prompt).split(".") if part.strip()}


def expected_frame_policy(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen 32-frame union policy for one canonical record."""

    samples = record.get("legacy_samples", {})
    indices: list[int] = []
    for offset in (0.0, 0.5):
        sample = samples.get(str(offset), {})
        indices.extend(int(index) for index in sample.get("frame_indices", []))
    return {
        "algorithm": "common.sample_index_list",
        "frame_indices": sorted(set(indices)),
        "offsets": [0.0, 0.5],
        "sampling_mode": "legacy",
    }


def compare_frame_contract(
    manifest_record: Mapping[str, Any], cache: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare cache policy and frame timestamps to the canonical timeline."""

    expected = expected_frame_policy(manifest_record)
    actual = cache.get("frame_policy")
    policy_status = "exact" if actual == expected else "mismatch"
    frame_rows = cache.get("frames", [])
    mismatch_indices: list[int] = []
    out_of_bounds: list[int] = []
    max_error = 0.0
    timestamps = manifest_record.get("frame_timestamps_s", [])
    for frame in frame_rows if isinstance(frame_rows, list) else []:
        try:
            index = int(frame["frame_index"])
            observed = float(frame["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        if index < 0 or index >= len(timestamps):
            out_of_bounds.append(index)
            continue
        error = abs(observed - float(timestamps[index]))
        max_error = max(max_error, error)
        if error > TIMESTAMP_TOLERANCE:
            mismatch_indices.append(index)
    timestamp_status = (
        "exact"
        if frame_rows
        and not out_of_bounds
        and not mismatch_indices
        else "mismatch"
    )
    return {
        "policy_status": policy_status,
        "expected_policy": expected,
        "actual_policy": actual,
        "timestamp_status": timestamp_status,
        "timestamp_max_abs_error_s": max_error,
        "timestamp_mismatch_indices": mismatch_indices,
        "timestamp_out_of_bounds_indices": out_of_bounds,
    }


def cache_identity_status(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Check the frozen detector, preprocessing, thresholds, and tracker IDs."""

    statuses = {
        "detector_revision": (
            "exact" if cache.get("detector_revision") == G.GROUNDING_DETECTOR_REVISION else "mismatch"
        ),
        "preprocessing": (
            "exact" if cache.get("preprocessing") == G.GROUNDING_DETECTOR_PREPROCESSING else "mismatch"
        ),
        "thresholds": "exact" if cache.get("thresholds") == EXPECTED_THRESHOLDS else "mismatch",
        "tracker": "exact" if cache.get("tracker") == EXPECTED_TRACKER else "mismatch",
        "prompt": "present" if _prompt_labels(str(cache.get("prompt", ""))) else "missing",
    }
    return {
        **statuses,
        "all_frozen_identity_exact": all(
            statuses[key] == "exact" for key in ("detector_revision", "preprocessing", "thresholds", "tracker")
        ),
        "prompt_label_count": len(_prompt_labels(str(cache.get("prompt", "")))),
        "prompt_sha256": hashlib.sha256(str(cache.get("prompt", "")).encode()).hexdigest(),
    }


def observation_metrics(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise continuity without interpreting detections as answers."""

    frames = cache.get("frames", [])
    frame_indices = [int(frame["frame_index"]) for frame in frames]
    ordinal = {index: position for position, index in enumerate(frame_indices)}
    tracks = cache.get("tracks", [])
    observation_count = 0
    single_observation_tracks = 0
    uncertain_tracks = 0
    gap_count = 0
    max_gap = 0
    tracks_with_gaps = 0
    label_switch_proxy = 0
    low_iou_proxy = 0
    out_of_policy_observations = 0
    iou_threshold = float(cache.get("tracker", {}).get("iou_threshold", G.DEFAULT_TRACK_IOU_THRESHOLD))
    for track in tracks if isinstance(tracks, list) else []:
        observations = track.get("observations", []) if isinstance(track, Mapping) else []
        observation_count += len(observations)
        if len(observations) == 1:
            single_observation_tracks += 1
        if bool(track.get("uncertain")) if isinstance(track, Mapping) else False:
            uncertain_tracks += 1
        ordered = sorted(observations, key=lambda row: int(row.get("frame_index", -1)))
        positions: list[int] = []
        labels: list[str] = []
        for observation in ordered:
            try:
                frame_index = int(observation["frame_index"])
            except (KeyError, TypeError, ValueError):
                out_of_policy_observations += 1
                continue
            if frame_index not in ordinal:
                out_of_policy_observations += 1
            else:
                positions.append(ordinal[frame_index])
            labels.append(str(observation.get("label", "")).casefold())
            if "association_iou" in observation:
                try:
                    if float(observation["association_iou"]) < iou_threshold:
                        low_iou_proxy += 1
                except (TypeError, ValueError):
                    pass
        if len(set(labels)) > 1:
            label_switch_proxy += sum(a != b for a, b in zip(labels, labels[1:]))
        gaps = [right - left - 1 for left, right in zip(sorted(set(positions)), sorted(set(positions))[1:]) if right - left > 1]
        if gaps:
            tracks_with_gaps += 1
            gap_count += len(gaps)
            max_gap = max(max_gap, max(gaps))
    return {
        "frame_count": len(frames),
        "detection_count": sum(len(frame.get("detections", [])) for frame in frames if isinstance(frame, Mapping)),
        "track_count": len(tracks) if isinstance(tracks, list) else 0,
        "observation_count": observation_count,
        "single_observation_track_count": single_observation_tracks,
        "uncertain_track_count": uncertain_tracks,
        "tracks_with_observation_gaps": tracks_with_gaps,
        "observation_gap_count": gap_count,
        "max_observation_gap_in_sampled_frames": max_gap,
        "label_switch_proxy_count": label_switch_proxy,
        "low_association_iou_proxy_count": low_iou_proxy,
        "out_of_policy_observation_count": out_of_policy_observations,
    }


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _review_rows(packet_path: Path, split_paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    packet = _load_json(packet_path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for split, path in split_paths.items():
        for row in _load_json(path).get("QA", []):
            if isinstance(row, Mapping):
                # Deliberately retain only fields needed for path and prompt
                # coverage; answer fields are never read.
                rows_by_id[str(row["qa_id"])] = {
                    "qa_id": str(row["qa_id"]),
                    "split": split,
                    "video_path": str(row["video_path"]),
                }
    result = []
    for item in packet.get("items", []):
        qa_id = str(item["qa_id"])
        row = dict(rows_by_id[qa_id])
        row["question_entities"] = question_entities(str(item.get("question", "")))
        result.append(row)
    return result


def _sidecar_candidates(selection_path: Path, context_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    selection = _load_json(selection_path)
    selection_rows = {
        str(row["qa_id"]): row
        for rows in selection.get("selected_questions", {}).values()
        for row in rows
    }
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for context_path in context_paths:
        with context_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                selected = selection_rows.get(str(row.get("qa_id")))
                if not selected or not row.get("cache_key"):
                    continue
                candidate = {
                    "cache_key": str(row["cache_key"]),
                    "source_sidecar_qa_ids": [str(row["qa_id"])],
                    "sidecar_content_digests": [str(row.get("cache_content_digest", ""))],
                    "sidecar_prompt_entities": list(row.get("prompt_entities_from_question_only", [])),
                }
                path = str(selected["video_path"])
                existing = next(
                    (item for item in candidates[path] if item["cache_key"] == candidate["cache_key"]),
                    None,
                )
                if existing:
                    existing["source_sidecar_qa_ids"].append(str(row["qa_id"]))
                    existing["sidecar_content_digests"].append(str(row.get("cache_content_digest", "")))
                else:
                    candidates[path].append(candidate)
    return dict(candidates)


def _source_checksum(source_root: Path, video_path: str, cache_checksum: str) -> dict[str, Any]:
    source = source_root / video_path
    if not source.is_file():
        return {"status": "unknown_source_missing", "source_path": str(source), "computed_sha256": None}
    computed = _sha256(source)
    return {
        "status": "match" if computed == cache_checksum else "mismatch",
        "source_path": str(source),
        "computed_sha256": computed,
    }


def _candidate_result(
    path: Path,
    key: str,
    manifest_record: Mapping[str, Any],
    source_root: Path,
    sidecar_digests: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"cache_key": key, "cache_path": str(path)}
    try:
        cache = G.load_detector_cache(path, expected_cache_key=key)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        result.update({"integrity_status": "invalid", "error": str(exc)})
        return result
    result["integrity_status"] = "valid"
    if sidecar_digests:
        result["sidecar_digest_status"] = (
            "match" if all(digest == cache["cache_content_digest"] for digest in sidecar_digests if digest) else "mismatch"
        )
    result["source_checksum"] = _source_checksum(source_root, str(manifest_record["video_path"]), str(cache["video_checksum"]))
    result["frame_contract"] = compare_frame_contract(manifest_record, cache)
    result["identity"] = cache_identity_status(cache)
    result["observation_metrics"] = observation_metrics(cache)
    result["contract_exact"] = bool(
        result["integrity_status"] == "valid"
        and result["frame_contract"]["policy_status"] == "exact"
        and result["frame_contract"]["timestamp_status"] == "exact"
        and result["identity"]["all_frozen_identity_exact"]
        and result["source_checksum"]["status"] != "mismatch"
        and result.get("sidecar_digest_status", "match") == "match"
    )
    return result


def validate(
    *,
    manifest_path: Path,
    review_packet_path: Path,
    selection_path: Path,
    context_paths: list[Path],
    cache_dir: Path,
    source_root: Path,
    control_path: Path,
    split_paths: Mapping[str, Path],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    validate_manifest(manifest)
    review_rows = _review_rows(review_packet_path, split_paths)
    candidates_by_video = _sidecar_candidates(selection_path, context_paths)
    records = manifest["records"]
    video_paths = sorted({row["video_path"] for row in review_rows})
    video_results: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    cases_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        cases_by_video[row["video_path"]].append(row)

    for video_path in video_paths:
        manifest_record = records.get(video_path)
        if manifest_record is None:
            video_results.append({"video_path": video_path, "status": "unaligned_manifest"})
            continue
        candidate_rows = []
        for candidate in candidates_by_video.get(video_path, []):
            cache_path = cache_dir / f"{candidate['cache_key']}.json"
            result = _candidate_result(
                cache_path,
                candidate["cache_key"],
                manifest_record,
                source_root,
                candidate["sidecar_content_digests"],
            )
            result["source_sidecar_qa_ids"] = candidate["source_sidecar_qa_ids"]
            result["sidecar_prompt_entities"] = sorted(set(candidate["sidecar_prompt_entities"]))
            candidate_rows.append(result)
        valid_contracts = [row for row in candidate_rows if row.get("contract_exact")]
        exact_policy = [
            row for row in candidate_rows
            if row.get("integrity_status") == "valid"
            and row.get("frame_contract", {}).get("policy_status") == "exact"
        ]
        if not candidate_rows:
            status = "missing_cache"
        elif valid_contracts:
            status = "exact_cache_reuse_pending_prompt_check"
        elif exact_policy:
            status = "same_video_cache_contract_mismatch"
        elif any(row.get("integrity_status") == "valid" for row in candidate_rows):
            status = "same_video_different_policy"
        else:
            status = "same_video_invalid_cache"
        video_results.append({
            "video_path": video_path,
            "status": status,
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
            "source_timestamp_source": manifest_record.get("timestamp_source"),
        })

        for case in cases_by_video[video_path]:
            target_labels = {str(label).casefold() for label in case["question_entities"]}
            compatible = []
            for candidate in valid_contracts:
                cache = G.load_detector_cache(candidate["cache_path"], expected_cache_key=candidate["cache_key"])
                prompt_labels = _prompt_labels(cache["prompt"])
                if not target_labels:
                    prompt_status = "unknown_question_entities"
                elif target_labels <= prompt_labels:
                    prompt_status = "covered"
                else:
                    prompt_status = "gap"
                compatible.append((candidate, prompt_status))
            if any(status == "covered" for _, status in compatible):
                case_status = "exact_cache_reuse"
                chosen = next(candidate for candidate, status in compatible if status == "covered")
                prompt_status = "covered"
            elif any(status == "unknown_question_entities" for _, status in compatible):
                case_status = "same_video_prompt_unverified"
                chosen, prompt_status = compatible[0]
            elif compatible:
                case_status = "same_video_prompt_gap"
                chosen, prompt_status = compatible[0]
            elif candidate_rows:
                chosen = candidate_rows[0]
                prompt_status = "unavailable_due_to_cache_contract"
                case_status = video_results[-1]["status"]
            else:
                chosen = None
                prompt_status = "unavailable"
                case_status = "missing_cache"
            case_results.append({
                "qa_id": case["qa_id"],
                "split": case["split"],
                "video_path": video_path,
                "question_entity_count": len(target_labels),
                "prompt_status": prompt_status,
                "status": case_status,
                "cache_key": chosen.get("cache_key") if chosen else None,
            })

    cases_by_video_status: dict[str, list[str]] = defaultdict(list)
    for case in case_results:
        cases_by_video_status[case["video_path"]].append(case["status"])
    for video in video_results:
        if video["status"] != "exact_cache_reuse_pending_prompt_check":
            continue
        statuses = cases_by_video_status[video["video_path"]]
        if statuses and all(status == "exact_cache_reuse" for status in statuses):
            video["status"] = "exact_cache_reuse"
        elif "same_video_prompt_gap" in statuses:
            video["status"] = "same_video_prompt_gap"
        else:
            video["status"] = "same_video_prompt_unverified"

    control = _load_json(control_path)
    control_ids = [str(qa_id) for split in control.get("splits", {}).values() for qa_id in split.get("ids", [])]
    deduped_control_ids = list(dict.fromkeys(control_ids))
    review_ids = {row["qa_id"] for row in review_rows}
    mapped_control_ids = sorted(
        qa_id for candidates in candidates_by_video.values() for candidate in candidates for qa_id in candidate["source_sidecar_qa_ids"]
        if qa_id in set(deduped_control_ids) and qa_id not in review_ids
    )
    strata_counts = Counter(
        stratum
        for split in control.get("splits", {}).values()
        for stratum, ids in split.get("strata", {}).items()
        for qa_id in ids
        if qa_id in set(deduped_control_ids)
    )
    video_statuses = Counter(row["status"] for row in video_results)
    case_statuses = Counter(row["status"] for row in case_results)
    mapped_keys = {
        candidate["cache_key"]
        for row in video_results
        for candidate in row.get("candidates", [])
    }
    cache_file_count = len(list(cache_dir.glob("*.json"))) if cache_dir.is_dir() else 0
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "manifest": {"path": str(manifest_path), "content_digest": manifest["content_digest"]},
        "cohort": {
            "source": str(review_packet_path),
            "review_case_count": len(review_rows),
            "review_video_count": len(video_paths),
            "split_counts": dict(Counter(row["split"] for row in review_rows)),
            "answer_fields_read": False,
        },
        "cache_inventory": {
            "cache_dir": str(cache_dir),
            "cache_file_count": cache_file_count,
            "mapped_candidate_key_count": len(mapped_keys),
            "sidecar_contexts": [str(path) for path in context_paths],
            "mapping_basis": "Phase 3 G1 selection video_path plus cache_key sidecars; cache JSON has no video_path",
        },
        "coverage": {
            "video_status_counts": dict(video_statuses),
            "case_status_counts": dict(case_statuses),
            "source_checksum_status_counts": dict(
                Counter(
                    candidate.get("source_checksum", {}).get("status")
                    for row in video_results
                    for candidate in row.get("candidates", [])
                    if candidate.get("source_checksum")
                )
            ),
            "policy_exact_video_count": sum(
                any(candidate.get("frame_contract", {}).get("policy_status") == "exact" for candidate in row.get("candidates", []))
                for row in video_results
            ),
            "timeline_exact_video_count": sum(
                any(candidate.get("frame_contract", {}).get("timestamp_status") == "exact" for candidate in row.get("candidates", []))
                for row in video_results
            ),
        },
        "independent_controls": {
            "source": str(control_path),
            "deduplicated_id_count": len(deduped_control_ids),
            "review_overlap_ids": sorted(set(deduped_control_ids) & review_ids),
            "documented_stratum_counts": dict(strata_counts),
            "cache_aligned_candidate_ids": sorted(set(mapped_control_ids)),
            "used_ids": [],
            "reason_if_empty": "Fixed controls were recorded for audit only; the Phase 7B report stays scoped to the reviewed packet and does not expand workload.",
        },
        "video_results": video_results,
        "case_results": case_results,
        "limitations": [
            "Cache records do not contain video_path; same-video reuse is limited to Phase 3 G1 sidecar mappings.",
            "A missing local source file yields unknown_source_missing; the cache checksum is never treated as a source checksum.",
            "Prompt coverage is checked from question-derived entities only; no answer, gold, or baseline field is read.",
            "Gap and switch values are conservative observation proxies, not identity ground truth or tracking accuracy.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Phase 7B cache validation",
        "",
        "Offline validation of existing Grounding DINO caches against the canonical Phase 7 frame contract and the reviewed Phase 2 packet.",
        "",
        f"- Reviewed cases/videos: {report['cohort']['review_case_count']}/{report['cohort']['review_video_count']}",
        f"- Video statuses: `{json.dumps(coverage['video_status_counts'], sort_keys=True)}`",
        f"- Case statuses: `{json.dumps(coverage['case_status_counts'], sort_keys=True)}`",
        f"- Policy-exact videos: {coverage['policy_exact_video_count']}/{report['cohort']['review_video_count']}",
        f"- Timeline-exact videos: {coverage['timeline_exact_video_count']}/{report['cohort']['review_video_count']}",
        f"- Source checksum statuses: `{json.dumps(coverage['source_checksum_status_counts'], sort_keys=True)}`",
        "",
        "## Gate",
        "",
        "The cache contract is usable for a bounded evidence smoke only where a Phase 3 sidecar maps the same video, policy/timestamps/model identity are exact, and prompt coverage is present. Source checksums remain an explicit unknown for videos absent from `data/ReVA`; obtain that source evidence before treating reuse as fully reproducible.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/phase7A_feasibility/canonical_frame_manifest.json")
    parser.add_argument("--review-packet", type=Path, default=ROOT / "runs/phase2_temporal_verified_window/review_material/review_packet.json")
    parser.add_argument("--selection", type=Path, default=ROOT / "runs/phase3_explicit_geometry/full/execution_selection.json")
    parser.add_argument("--context", type=Path, action="append", default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs/phase3_explicit_geometry/full/grounding_cache")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/ReVA")
    parser.add_argument("--control-list", type=Path, default=ROOT / "tasks/phase3_grounding_pilot_ids.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/phase7B_cache_validation")
    args = parser.parse_args()
    context_paths = args.context or [
        ROOT / "runs/phase3_explicit_geometry/full/g1_valQ_contexts.jsonl",
        ROOT / "runs/phase3_explicit_geometry/full/g1_valT_contexts.jsonl",
    ]
    report = validate(
        manifest_path=args.manifest,
        review_packet_path=args.review_packet,
        selection_path=args.selection,
        context_paths=context_paths,
        cache_dir=args.cache_dir,
        source_root=args.source_root,
        control_path=args.control_list,
        split_paths={"valQ": ROOT / "data/splits/valQ.json", "valT": ROOT / "data/splits/valT.json"},
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
