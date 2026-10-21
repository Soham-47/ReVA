"""Score a split with letter logits and frame/option test-time augmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import grounding as G
import segmentation as S
import sam3_overlay as M

SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}

FEATURE_PREPROCESSING = G.FEATURE_PREPROCESSING


def move_inputs(inputs: dict, device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _read_selection_file(path: str | None, *, label: str, split: str) -> set[str] | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.casefold() == ".json":
        try:
            payload = json.loads(source.read_text())
            section = payload["splits"][split]
            values = section["ids"] if isinstance(section, dict) else section
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} JSON must contain splits.{split}.ids") from exc
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{label} JSON split {split} must contain a string ids list")
    else:
        values = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} file must contain unique non-empty lines")
    return set(values)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="final_adapter directory, or none for zero-shot")
    ap.add_argument(
        "--no-merge-adapters",
        action="store_true",
        help="keep named PEFT adapters active instead of merging them into bfloat16 base weights",
    )
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=','.join(str(value) for value in C.DEFAULT_SCORE_OFFSETS))
    ap.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    ap.add_argument("--frame-sampling", choices=C.FRAME_SAMPLING_MODES, default=C.DEFAULT_FRAME_SAMPLING)
    ap.add_argument(
        "--adaptive-frames",
        action="store_true",
        help="use query-guided frames from an offline feature cache",
    )
    ap.add_argument(
        "--grounding-context",
        "--grounding",
        "--grounding-only",
        dest="grounding_context",
        action="store_true",
        help="add bounded detector tracks from an offline Grounding DINO cache",
    )
    ap.add_argument("--grounding-cache-dir", help="directory containing keyed detector/track cache JSON files")
    ap.add_argument("--grounding-prompt", default=G.DEFAULT_GROUNDING_PROMPT)
    ap.add_argument("--grounding-box-threshold", type=float, default=G.DEFAULT_DETECTOR_THRESHOLD)
    ap.add_argument("--grounding-text-threshold", type=float, default=G.DEFAULT_DETECTOR_TEXT_THRESHOLD)
    ap.add_argument("--grounding-iou-threshold", type=float, default=G.DEFAULT_TRACK_IOU_THRESHOLD)
    ap.add_argument("--grounding-max-tracks", type=int, default=8)
    ap.add_argument("--grounding-context-chars", type=int, default=1200)
    ap.add_argument(
        "--segmentation-context",
        "--segmentation",
        dest="segmentation_context",
        action="store_true",
        help="add bounded SAM 2 mask evidence beside the detector context",
    )
    ap.add_argument("--segmentation-cache-dir", help="directory containing keyed SAM 2 mask caches")
    ap.add_argument(
        "--sam3-overlay-dir",
        help="directory containing direct SAM 3 text-mask overlay caches",
    )
    ap.add_argument("--segmentation-max-masks", type=int, default=8)
    ap.add_argument(
        "--segmentation-context-chars",
        type=int,
        default=1000,
        help="deprecated compatibility option; visual-overlay scoring adds no mask text",
    )
    ap.add_argument("--qa-ids-file", help="newline-delimited IDs or split-aware pilot JSON manifest")
    ap.add_argument("--video-paths-file", help="newline-delimited deterministic pilot video paths")
    ap.add_argument("--feature-cache-dir", help="directory containing keyed offline feature-cache JSON files")
    ap.add_argument("--feature-encoder-revision", default=G.FEATURE_ENCODER_REVISION)
    ap.add_argument("--candidate-rate", type=float, default=G.DEFAULT_CANDIDATE_RATE)
    ap.add_argument("--candidate-cap", type=int, default=G.DEFAULT_CANDIDATE_CAP)
    ap.add_argument("--anchor-count", type=int, default=4)
    ap.add_argument("--min-spacing", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if not 1 <= args.rots <= 4 or not 0 <= args.shard < args.nshards:
        ap.error("rots must be 1..4, and shard must be in [0, nshards)")
    if args.candidate_rate <= 0 or args.candidate_cap <= 0 or not 0 <= args.anchor_count <= 16:
        ap.error("candidate-rate/cap must be positive and anchor-count must be in [0, 16]")
    if args.min_spacing is not None and args.min_spacing < 1:
        ap.error("min-spacing must be positive")
    if args.adaptive_frames and args.frame_sampling != C.DEFAULT_FRAME_SAMPLING:
        ap.error("--adaptive-frames requires --frame-sampling legacy; adaptive offsets select their own anchors")
    if args.adaptive_frames and not 3 <= args.anchor_count <= 16:
        ap.error("adaptive frame selection requires anchor-count in [3, 16]")
    if args.adaptive_frames and not args.feature_cache_dir:
        ap.error("--adaptive-frames requires --feature-cache-dir with offline cached features")
    if args.adaptive_frames and args.grounding_context:
        ap.error("grounding context cannot be combined with adaptive frames before the pilot gate")
    if args.segmentation_context and not args.grounding_context:
        ap.error("--segmentation-context requires --grounding-context so the detector-only control stays identical")
    if args.segmentation_context and not args.segmentation_cache_dir:
        ap.error("--segmentation-context requires --segmentation-cache-dir with offline SAM 2 caches")
    if args.sam3_overlay_dir and not args.grounding_context:
        ap.error("--sam3-overlay-dir requires --grounding-context so the detector-only control stays identical")
    if args.sam3_overlay_dir and args.segmentation_context:
        ap.error("--sam3-overlay-dir cannot be combined with --segmentation-context")
    if args.grounding_context and not args.grounding_cache_dir:
        ap.error("--grounding-context requires --grounding-cache-dir with offline detector caches")
    if (args.qa_ids_file or args.video_paths_file) and not args.grounding_context:
        ap.error("pilot selection files require --grounding-context")
    if args.grounding_context and args.frame_sampling != C.DEFAULT_FRAME_SAMPLING:
        ap.error("grounding context requires baseline legacy frame sampling")
    if not all(
        0 <= value <= 1
        for value in (
            args.grounding_box_threshold,
            args.grounding_text_threshold,
            args.grounding_iou_threshold,
        )
    ):
        ap.error("grounding thresholds must be in [0, 1]")
    if args.grounding_max_tracks < 0 or args.grounding_context_chars <= 0:
        ap.error("grounding max-tracks must be non-negative and context-chars must be positive")
    if args.segmentation_max_masks < 0 or args.segmentation_context_chars <= 0:
        ap.error("segmentation max-masks must be non-negative and context-chars must be positive")
    offsets = [float(value) for value in args.offsets.split(",") if value.strip()]
    if not offsets or any(not 0 <= value < 1 for value in offsets):
        ap.error("offsets must be comma-separated values in [0, 1)")

    split_path = C.work_path(SPLITS[args.split])
    with split_path.open() as f:
        all_rows = json.load(f)["QA"]
    split_digest = C.split_checksum(all_rows)
    rows = all_rows
    rows = rows[args.shard :: args.nshards]
    qa_ids = _read_selection_file(args.qa_ids_file, label="QA IDs", split=args.split)
    video_paths = _read_selection_file(args.video_paths_file, label="video paths", split=args.split)
    if qa_ids is not None or video_paths is not None:
        rows = [
            row
            for row in rows
            if (qa_ids is None or row.get("qa_id") in qa_ids)
            and (video_paths is None or row.get("video_path") in video_paths)
        ]
        if not rows:
            ap.error("pilot selection files selected no questions")
        if qa_ids is not None and qa_ids - {row.get("qa_id") for row in rows}:
            ap.error("QA selection includes IDs absent from the selected split/shard")
        if video_paths is not None and video_paths - {row.get("video_path") for row in rows}:
            ap.error("video selection includes paths absent from the selected split/shard")
    out = C.work_path(args.out)
    if args.sam3_overlay_dir:
        out = M.score_output_path(out)
    elif args.segmentation_context:
        out = S.segmentation_output_path(out)
    elif args.adaptive_frames:
        out = G.evidence_output_path(out)
    elif args.grounding_context:
        out = G.grounding_output_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapter_path = None if args.adapter == "none" else C.work_path(args.adapter)
    expected_manifest = {
        "schema_version": 1,
        "adapter": "none" if adapter_path is None else str(adapter_path.resolve()),
        "adapter_checksum": "none" if adapter_path is None else C.path_checksum(adapter_path),
        "adapter_merge_mode": (
            "none"
            if adapter_path is None
            else "unmerged"
            if args.no_merge_adapters
            else "merged"
        ),
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REV,
        "dataset_revision": C.dataset_revision(),
        "split": args.split,
        "split_checksum": split_digest,
        "offsets": offsets,
        "rots": args.rots,
        "frame_sampling": args.frame_sampling,
        "shard": args.shard,
        "nshards": args.nshards,
    }
    candidate_policy = G.candidate_policy(args.candidate_rate, args.candidate_cap)
    if args.adaptive_frames:
        expected_manifest = G.build_evidence_manifest(
            expected_manifest,
            selector_enabled=True,
            grounding_enabled=False,
            candidate_rate=args.candidate_rate,
            candidate_cap=args.candidate_cap,
            feature_encoder_revision=args.feature_encoder_revision,
            feature_preprocessing=FEATURE_PREPROCESSING,
            selection_params={
                "mode": "uniform-anchors-plus-query-relevance",
                "limit": C.MAX_FRAMES,
                "anchor_count": args.anchor_count,
                "min_spacing": args.min_spacing,
                "offset_specific": True,
                "spacing_fallback": "fill-unique-candidates",
                "short_clip_frame_count": "common.sample_index_list",
                "candidate_policy": candidate_policy,
            },
        )
    elif args.grounding_context:
        expected_manifest = G.build_evidence_manifest(
            expected_manifest,
            selector_enabled=False,
            grounding_enabled=True,
            grounding_params={
                "mode": "baseline-frame-grounding-only",
                "detector_revision": G.GROUNDING_DETECTOR_REVISION,
                "prompt": G.normalize_detector_prompt(args.grounding_prompt),
                "preprocessing": G.GROUNDING_DETECTOR_PREPROCESSING,
                "thresholds": {
                    "box_threshold": args.grounding_box_threshold,
                    "text_threshold": args.grounding_text_threshold,
                },
                "tracker": {
                    "algorithm": "adjacent-greedy-iou",
                    "iou_threshold": args.grounding_iou_threshold,
                },
                "context": {
                    "max_tracks": args.grounding_max_tracks,
                    "max_chars": args.grounding_context_chars,
                },
                "pilot_selection": {
                    "qa_ids_file": args.qa_ids_file,
                    "qa_ids_file_checksum": (
                        C.path_checksum(args.qa_ids_file) if args.qa_ids_file else None
                    ),
                    "video_paths_file": args.video_paths_file,
                    "video_paths_file_checksum": (
                        C.path_checksum(args.video_paths_file) if args.video_paths_file else None
                    ),
                    "selected_questions": len(rows),
                    "selected_videos": len({row["video_path"] for row in rows}),
                },
            },
            output_kind=(
                "sam3_segmentation_score"
                if args.sam3_overlay_dir
                else "segmentation_score"
                if args.segmentation_context
                else "grounding_score"
            ),
        )
        expected_manifest["detector_cache_key_fields"] = list(G.GROUNDING_CACHE_KEY_FIELDS)
        if args.segmentation_context:
            expected_manifest["segmentation_cache_key_fields"] = list(S.SEGMENTATION_CACHE_KEY_FIELDS)
            expected_manifest["segmentation"] = {
                "enabled": True,
                "version": S.SEGMENTATION_VERSION,
                "evidence": "visual-overlay-only",
                "text_context": False,
                "model_revision": S.SEGMENTATION_MODEL_REVISION_ID,
                "mask_policy": S.segmentation_mask_policy(S.DEFAULT_MAX_MASKS_PER_FRAME),
            }
        if args.sam3_overlay_dir:
            expected_manifest["sam3_segmentation"] = {
                "enabled": True,
                "version": M.SAM3_OVERLAY_VERSION,
                "evidence": "direct-text-mask-overlay-only",
                "text_context": False,
                "model": "lucataco/sam3-video",
                "direct_prompt_calls": True,
                "cache_dir": str(C.work_path(args.sam3_overlay_dir).resolve()),
                "selected_questions": len(rows),
            }
    manifest_path = C.score_manifest_path(out)
    if manifest_path.exists():
        with manifest_path.open() as f:
            actual_manifest = json.load(f)
        if args.adaptive_frames and actual_manifest.get("output_kind") != "evidence_score":
            raise RuntimeError("adaptive output cannot reuse a baseline score path; choose a new output")
        if args.segmentation_context and actual_manifest.get("output_kind") != "segmentation_score":
            raise RuntimeError("segmentation output cannot reuse a baseline score path; choose a new output")
        if args.sam3_overlay_dir and actual_manifest.get("output_kind") != "sam3_segmentation_score":
            raise RuntimeError("SAM 3 output cannot reuse a baseline score path; choose a new output")
        if (
            args.grounding_context
            and not args.segmentation_context
            and not args.sam3_overlay_dir
            and actual_manifest.get("output_kind") != "grounding_score"
        ):
            raise RuntimeError("grounding output cannot reuse a baseline score path; choose a new output")
        actual_manifest.setdefault("frame_sampling", C.DEFAULT_FRAME_SAMPLING)
        # Existing manifests predate this field and were produced by the
        # historical merged loader.  Fill it in memory so baseline reruns stay
        # compatible while unmerged runs cannot reuse merged scores.
        actual_manifest.setdefault(
            "adapter_merge_mode",
            "none" if adapter_path is None else "merged",
        )
        if actual_manifest != expected_manifest:
            raise RuntimeError(f"score manifest mismatch: {manifest_path}; choose a new output or remove stale scores")
    elif (args.adaptive_frames or args.grounding_context) and out.exists():
        raise RuntimeError("opt-in output path already exists without an evidence manifest; choose a new output")
    elif out.exists() and out.stat().st_size:
        raise RuntimeError(f"existing score file has no manifest: {out}; choose a new output or remove stale scores")
    else:
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with temporary.open("w") as f:
            json.dump(expected_manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        temporary.replace(manifest_path)
    done = C.read_completed_jsonl(out)
    stop_file = os.environ.get("REVA_STOP_FILE")

    adaptive_rows: dict[str, dict] = {}
    if args.adaptive_frames:
        cache_dir = C.work_path(args.feature_cache_dir)
        cache_by_video: dict[str, dict] = {}
        video_checksums: dict[str, str] = {}
        video_manifest = C.load_manifest()
        for row in rows:
            if row["qa_id"] in done:
                continue
            relative_path = row["video_path"]
            try:
                info = video_manifest[relative_path]
            except KeyError as exc:
                raise KeyError(f"{relative_path!r} is absent from {C.manifest_path()}") from exc
            video_checksum = video_checksums.get(relative_path)
            if video_checksum is None:
                video_checksum = C.path_checksum(C.dataset_dir() / relative_path)
                video_checksums[relative_path] = video_checksum
            cache_key = G.feature_cache_key(
                video_checksum,
                args.feature_encoder_revision,
                candidate_policy,
                FEATURE_PREPROCESSING,
            )
            expected_grid = G.candidate_index_list(
                int(info["n_frames"]),
                float(info["fps"]),
                candidate_policy["rate"],
                candidate_policy["cap"],
            )
            cache = cache_by_video.get(cache_key)
            if cache is None:
                cache_path = G.feature_cache_path(
                    cache_dir,
                    video_checksum,
                    args.feature_encoder_revision,
                    candidate_policy,
                    FEATURE_PREPROCESSING,
                )
                cache = G.load_feature_cache(cache_path, expected_cache_key=cache_key)
                cache_by_video[cache_key] = cache
            if not G.candidate_grid_matches(cache["candidates"], expected_grid):
                raise RuntimeError(
                    f"offline feature cache {cache_key} has a candidate-grid mismatch for {relative_path!r}"
                )
            try:
                query_feature = cache["query_features"][row["qa_id"]]
            except KeyError as exc:
                raise RuntimeError(
                    f"offline feature cache {cache_key} has no query feature for {row['qa_id']!r}"
                ) from exc
            actual_fingerprint = G.query_input_fingerprint(row)
            if cache["query_fingerprints"].get(row["qa_id"]) != actual_fingerprint:
                raise RuntimeError(
                    f"offline feature cache {cache_key} has stale query features for {row['qa_id']!r}"
                )
            selected_by_offset = {}
            for offset in offsets:
                target_count = len(
                    C.sample_index_list(
                        int(info["n_frames"]),
                        float(info["fps"]),
                        offset,
                        mode=C.DEFAULT_FRAME_SAMPLING,
                    )
                )
                selected = G.select_adaptive_frames(
                    cache["candidates"],
                    query_feature,
                    limit=min(C.MAX_FRAMES, target_count),
                    anchor_count=args.anchor_count,
                    min_spacing=args.min_spacing,
                    temporal_offset=offset,
                )
                selected = G.pad_frame_selection(selected, target_count)
                selected_frames = []
                for candidate in selected:
                    index = int(candidate["frame_index"])
                    if not 0 <= index < int(info["n_frames"]):
                        raise ValueError(f"feature cache selected invalid frame {index} for {relative_path!r}")
                    timestamp = index / max(float(info["fps"]), 1e-6)
                    selected_frames.append(
                        {
                            "frame_index": index,
                            "timestamp": float(timestamp),
                            "relevance": float(candidate.get("relevance", 0.0)),
                        }
                    )
                if not selected_frames:
                    raise RuntimeError(f"offline feature cache selected no frames for {row['qa_id']!r}")
                selected_by_offset[str(offset)] = selected_frames
            adaptive_rows[row["qa_id"]] = {
                "cache_key": cache_key,
                "cache_content_digest": cache["cache_content_digest"],
                "views": selected_by_offset,
            }

    grounding_rows: dict[str, dict] = {}
    segmentation_rows: dict[str, dict] = {}
    sam3_rows: dict[str, dict] = {}
    if args.sam3_overlay_dir:
        for row in rows:
            if row["qa_id"] in done:
                continue
            sam3_rows[row["qa_id"]] = M.load_overlay_cache(
                C.work_path(args.sam3_overlay_dir), row["qa_id"]
            )
    if args.grounding_context:
        cache_dir = C.work_path(args.grounding_cache_dir)
        cache_by_video: dict[str, dict] = {}
        video_checksums: dict[str, str] = {}
        video_manifest = C.load_manifest()
        prompt = G.normalize_detector_prompt(args.grounding_prompt)
        thresholds = {
            "box_threshold": args.grounding_box_threshold,
            "text_threshold": args.grounding_text_threshold,
        }
        tracker = {
            "algorithm": "adjacent-greedy-iou",
            "iou_threshold": args.grounding_iou_threshold,
        }
        for row in rows:
            if row["qa_id"] in done:
                continue
            relative_path = row["video_path"]
            try:
                info = video_manifest[relative_path]
            except KeyError as exc:
                raise KeyError(f"{relative_path!r} is absent from {C.manifest_path()}") from exc
            video_checksum = video_checksums.get(relative_path)
            if video_checksum is None:
                video_checksum = C.path_checksum(C.dataset_dir() / relative_path)
                video_checksums[relative_path] = video_checksum
            frame_indices = []
            for offset in offsets:
                frame_indices.extend(
                    C.sample_index_list(
                        int(info["n_frames"]),
                        float(info["fps"]),
                        offset,
                        mode=C.DEFAULT_FRAME_SAMPLING,
                    )
                )
            frame_policy = G.detector_frame_policy(frame_indices, offsets, C.DEFAULT_FRAME_SAMPLING)
            cache_key = G.detector_cache_key(
                video_checksum,
                G.GROUNDING_DETECTOR_REVISION,
                prompt,
                G.GROUNDING_DETECTOR_PREPROCESSING,
                thresholds,
                frame_policy,
                tracker,
            )
            cache = cache_by_video.get(cache_key)
            if cache is None:
                cache_path = G.detector_cache_path(
                    cache_dir,
                    video_checksum,
                    G.GROUNDING_DETECTOR_REVISION,
                    prompt,
                    G.GROUNDING_DETECTOR_PREPROCESSING,
                    thresholds,
                    frame_policy,
                    tracker,
                )
                cache = G.load_detector_cache(cache_path, expected_cache_key=cache_key)
                cache_by_video[cache_key] = cache
            if cache["frame_policy"] != frame_policy:
                raise RuntimeError(
                    f"offline detector cache {cache_key} has a frame-policy mismatch for {relative_path!r}"
                )
            entities = G.question_track_entities(row, cache["tracks"])
            relevant = G.filter_question_tracks(
                cache["tracks"],
                entities,
                min_confidence=args.grounding_box_threshold,
                max_tracks=args.grounding_max_tracks,
            )
            context = G.format_grounding_context(
                relevant,
                max_chars=args.grounding_context_chars,
                max_tracks=args.grounding_max_tracks,
            )
            grounding_rows[row["qa_id"]] = {
                "cache_key": cache_key,
                "cache_content_digest": cache["cache_content_digest"],
                "entities": entities,
                "context": context,
            }
            if args.segmentation_context:
                mask_policy = {
                    **S.segmentation_mask_policy(S.DEFAULT_MAX_MASKS_PER_FRAME),
                }
                segmentation_key = S.segmentation_cache_key(
                    video_checksum,
                    S.SEGMENTATION_MODEL_REVISION_ID,
                    cache_key,
                    frame_policy,
                    mask_policy,
                )
                segmentation_path = S.segmentation_cache_path(
                    C.work_path(args.segmentation_cache_dir),
                    video_checksum,
                    S.SEGMENTATION_MODEL_REVISION_ID,
                    cache_key,
                    frame_policy,
                    mask_policy,
                )
                segmentation_cache = S.load_segmentation_cache(
                    segmentation_path, expected_cache_key=segmentation_key
                )
                masks = S.flatten_masks(segmentation_cache)
                mask_entities = G.question_track_entities(row, masks)
                relevant_masks = S.filter_question_masks(
                    masks,
                    mask_entities,
                    min_confidence=args.grounding_box_threshold,
                    max_masks=args.segmentation_max_masks,
                )
                segmentation_rows[row["qa_id"]] = {
                    "cache_key": segmentation_key,
                    "cache_content_digest": segmentation_cache["cache_content_digest"],
                    "entities": mask_entities,
                    "masks": relevant_masks,
                    "contact_sheet": segmentation_cache.get("contact_sheet"),
                }

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video scoring")
    attn_implementation = "sdpa"
    C.configure_attention_backend(attn_implementation)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    if adapter_path is not None:
        model = C.load_peft_for_inference(model, adapter_path, merge=not args.no_merge_adapters)
    device = torch.device("cuda")
    model.to(device).eval()
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(args.rots)]
    with out.open("a") as stream, torch.inference_mode():
        for row in rows:
            if row["qa_id"] in done:
                continue
            if stop_file and Path(stop_file).exists():
                print(f"stop requested; resume from {out}")
                break
            views = []
            selected_provenance = adaptive_rows.get(row["qa_id"]) if args.adaptive_frames else None
            grounding_provenance = grounding_rows.get(row["qa_id"]) if args.grounding_context else None
            segmentation_provenance = (
                segmentation_rows.get(row["qa_id"]) if args.segmentation_context else None
            )
            sam3_provenance = (
                sam3_rows.get(row["qa_id"]) if args.sam3_overlay_dir else None
            )
            for offset in offsets:
                view_selection = None
                if args.adaptive_frames:
                    view_selection = selected_provenance["views"][str(offset)]
                    frames, meta = C.get_video_by_indices(
                        row["video_path"],
                        [item["frame_index"] for item in view_selection],
                    )
                else:
                    frames, meta = C.get_video(row["video_path"], offset, mode=args.frame_sampling)
                # Keep no-match questions identical to detector-only input:
                # do not duplicate frames or add an empty segmentation prompt.
                overlay_applied = bool(
                    segmentation_provenance is not None
                    and segmentation_provenance["masks"]
                )
                sampled_indices = {int(index) for index in meta.frames_indices}
                sam3_view_masks = (
                    M.masks_for_frame_indices(sam3_provenance["masks"], sampled_indices)
                    if sam3_provenance is not None
                    else []
                )
                sam3_overlay_applied = bool(sam3_view_masks)
                if overlay_applied:
                    frames, meta = S.augment_video_with_masks(
                        frames,
                        meta,
                        segmentation_provenance["masks"],
                        C.work_path(args.segmentation_cache_dir),
                    )
                if sam3_overlay_applied:
                    frames, meta = S.augment_video_with_masks(
                        frames,
                        meta,
                        sam3_view_masks,
                        C.work_path(args.sam3_overlay_dir),
                    )
                for permutation in rotations:
                    encoded, actual_perm = C.encode(
                        processor,
                        row,
                        frames,
                        meta,
                        permutation,
                        grounding_context=(
                            grounding_provenance["context"] if grounding_provenance is not None else None
                        ),
                    )
                    logits = model(**move_inputs(encoded, device)).logits[0, -1, ltr_ids].float()
                    shown_prob = torch.softmax(logits, dim=-1).cpu().numpy()
                    original_prob = [0.0] * 4
                    for shown_index, original_letter in enumerate(actual_perm):
                        original_prob[C.LETTERS.index(original_letter)] = float(shown_prob[shown_index])
                    view = {
                        "offset": offset,
                        "perm": "".join(actual_perm),
                        "p": original_prob,
                        "argmax_shown": C.LETTERS[int(np.argmax(shown_prob))],
                    }
                    if view_selection is not None:
                        view["feature_cache_key"] = selected_provenance["cache_key"]
                        view["feature_cache_digest"] = selected_provenance["cache_content_digest"]
                        view["selected_frames"] = view_selection
                    if grounding_provenance is not None:
                        view["grounding_cache_key"] = grounding_provenance["cache_key"]
                        view["grounding_cache_digest"] = grounding_provenance["cache_content_digest"]
                    if segmentation_provenance is not None:
                        view["segmentation_cache_key"] = segmentation_provenance["cache_key"]
                        view["segmentation_cache_digest"] = segmentation_provenance["cache_content_digest"]
                        view["segmentation_overlay_applied"] = overlay_applied
                    if sam3_provenance is not None:
                        view["sam3_overlay_cache_key"] = sam3_provenance["cache_key"]
                        view["sam3_overlay_cache_digest"] = sam3_provenance["cache_content_digest"]
                        view["sam3_overlay_applied"] = sam3_overlay_applied
                    views.append(view)
            mean = np.mean([view["p"] for view in views], axis=0).tolist()
            result = {
                "qa_id": row["qa_id"],
                "p": mean,
                "pred": C.LETTERS[int(np.argmax(mean))],
                "views": views,
            }
            if selected_provenance is not None:
                result["feature_cache_key"] = selected_provenance["cache_key"]
                result["feature_cache_digest"] = selected_provenance["cache_content_digest"]
                result["selected_frames_by_offset"] = selected_provenance["views"]
            if grounding_provenance is not None:
                result["grounding_cache_key"] = grounding_provenance["cache_key"]
                result["grounding_cache_digest"] = grounding_provenance["cache_content_digest"]
                result["grounding_entities"] = grounding_provenance["entities"]
                result["grounding_context"] = grounding_provenance["context"]
            if segmentation_provenance is not None:
                result["segmentation_cache_key"] = segmentation_provenance["cache_key"]
                result["segmentation_cache_digest"] = segmentation_provenance["cache_content_digest"]
                result["segmentation_entities"] = segmentation_provenance["entities"]
                result["segmentation_masks"] = segmentation_provenance["masks"]
                result["segmentation_contact_sheet"] = segmentation_provenance["contact_sheet"]
                result["segmentation_overlay_applied"] = bool(segmentation_provenance["masks"])
            if sam3_provenance is not None:
                result["sam3_overlay_cache_key"] = sam3_provenance["cache_key"]
                result["sam3_overlay_cache_digest"] = sam3_provenance["cache_content_digest"]
                result["sam3_prompts"] = sam3_provenance["prompts"]
                result["sam3_mask_count"] = sam3_provenance["mask_count"]
                result["sam3_overlay_applied"] = bool(sam3_provenance["masks"])
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
    print(f"scored {args.split} shard {args.shard}/{args.nshards}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
