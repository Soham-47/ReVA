"""Build the identical 2D relation-text schema from DINO or SAM3 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C
import grounding as G
import sam3_overlay as M
from geometry_relations import question_entities, relation_context, relation_subtype


OFFSETS = (0.0, 0.5)
TRACKER = {"algorithm": "adjacent-greedy-iou", "iou_threshold": G.DEFAULT_TRACK_IOU_THRESHOLD}
THRESHOLDS = {"box_threshold": G.DEFAULT_DETECTOR_THRESHOLD, "text_threshold": G.DEFAULT_DETECTOR_TEXT_THRESHOLD}


def _legacy_frame_indices(row: dict[str, Any], manifest: dict[str, Any]) -> set[int]:
    info = manifest[row["video_path"]]
    indices: set[int] = set()
    for offset in OFFSETS:
        indices.update(
            C.sample_index_list(int(info["n_frames"]), float(info["fps"]), offset, mode="legacy")
        )
    return indices


def _selected_rows(split: str, selection_path: Path) -> list[dict[str, Any]]:
    split_rows = json.loads((ROOT / "data" / "splits" / f"{split}.json").read_text())["QA"]
    selected = json.loads(selection_path.read_text())["selected_questions"][split]
    by_id = {row["qa_id"]: row for row in split_rows}
    try:
        return [by_id[row["qa_id"]] for row in selected]
    except KeyError as exc:
        raise RuntimeError(f"selected question is absent from {split}: {exc}") from exc


def _detector_records(cache: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for track in cache.get("tracks", []):
        label = next(iter(G._track_labels(track)), "unknown")
        confidence = float(track.get("confidence", track.get("score", 0.0)))
        for observation in G._observations(track):
            record = dict(observation)
            record["label"] = label
            record["confidence"] = confidence
            records.append(record)
    return records


def _detector_cache(
    row: dict[str, Any],
    *,
    cache_dir: Path,
    prompt: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    info = manifest[row["video_path"]]
    checksum = C.path_checksum(C.dataset_dir() / row["video_path"])
    frame_indices = []
    for offset in OFFSETS:
        frame_indices.extend(C.sample_index_list(int(info["n_frames"]), float(info["fps"]), offset, mode="legacy"))
    frame_policy = G.detector_frame_policy(frame_indices, OFFSETS, "legacy")
    key = G.detector_cache_key(
        checksum,
        G.GROUNDING_DETECTOR_REVISION,
        prompt,
        G.GROUNDING_DETECTOR_PREPROCESSING,
        THRESHOLDS,
        frame_policy,
        TRACKER,
    )
    path = G.detector_cache_path(
        cache_dir,
        checksum,
        G.GROUNDING_DETECTOR_REVISION,
        prompt,
        G.GROUNDING_DETECTOR_PREPROCESSING,
        THRESHOLDS,
        frame_policy,
        TRACKER,
    )
    cache = G.load_detector_cache(path, expected_cache_key=key)
    if cache["frame_policy"] != frame_policy:
        raise RuntimeError(f"detector cache frame policy mismatch for {row['qa_id']}")
    return cache


def build(
    *,
    split: str,
    condition: str,
    selection_path: Path,
    out: Path,
    grounding_cache_dir: Path | None = None,
    sam3_cache_dir: Path | None = None,
    detector_prompt: str | None = None,
) -> Path:
    if condition not in {"G1", "G2"}:
        raise ValueError("condition must be G1 or G2")
    rows = _selected_rows(split, selection_path)
    manifest = C.load_manifest()
    if condition == "G1":
        if grounding_cache_dir is None or not detector_prompt:
            raise ValueError("G1 requires grounding cache dir and question-only detector prompt")
    elif sam3_cache_dir is None:
        raise ValueError("G2 requires SAM3 overlay cache dir")
    output: list[dict[str, Any]] = []
    for row in rows:
        entities = question_entities(row["question"])
        if condition == "G1":
            cache = _detector_cache(row, cache_dir=grounding_cache_dir, prompt=detector_prompt, manifest=manifest)
            records = _detector_records(cache)
            cache_key = cache["cache_key"]
            cache_digest = cache["cache_content_digest"]
        else:
            cache = M.load_overlay_cache(sam3_cache_dir, row["qa_id"])
            frame_indices = _legacy_frame_indices(row, manifest)
            records = [record for record in cache["masks"] if int(record["frame_index"]) in frame_indices]
            cache_key = cache["cache_key"]
            cache_digest = cache["cache_content_digest"]
        subtype = relation_subtype(row["question"])
        if subtype not in {"image_plane_direction", "overlap_containment"}:
            context = (
                "Explicit geometry context: the question subtype is unsupported by the strict "
                "2D image-plane schema; fall back to baseline."
            )
            audit = {
                "entities": entities,
                "pairs": [],
                "status": "unsupported_subtype",
                "relation_subtype": subtype,
                "context_chars": len(context),
            }
        else:
            context, audit = relation_context(records, entities)
            audit["relation_subtype"] = subtype
        audit["frame_policy"] = {
            "mode": "legacy",
            "offsets": list(OFFSETS),
            "evidence_frame_count": len({int(record["frame_index"]) for record in records}),
        }
        output.append(
            {
                "qa_id": row["qa_id"],
                "split": split,
                "condition": condition,
                "relation_schema": "explicit-2d-image-plane-v1",
                "prompt_entities_from_question_only": entities,
                "context": context,
                "audit": audit,
                "cache_key": cache_key,
                "cache_content_digest": cache_digest,
                "fallback_policy": "use G0 baseline semantics when evidence is missing, ambiguous, or unsupported",
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        for row in output:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("valQ", "valT"), required=True)
    parser.add_argument("--condition", choices=("G1", "G2"), required=True)
    parser.add_argument("--selection", default="runs/phase3_explicit_geometry/selection_manifest.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--grounding-cache-dir")
    parser.add_argument("--sam3-cache-dir")
    parser.add_argument("--detector-prompt")
    args = parser.parse_args()
    build(
        split=args.split,
        condition=args.condition,
        selection_path=ROOT / args.selection,
        out=ROOT / args.out,
        grounding_cache_dir=ROOT / args.grounding_cache_dir if args.grounding_cache_dir else None,
        sam3_cache_dir=ROOT / args.sam3_cache_dir if args.sam3_cache_dir else None,
        detector_prompt=args.detector_prompt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
