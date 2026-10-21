"""Stage and freeze source-frame material for the approved Phase 4 QCSRR run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase4_qcsrr"
PLAN = ROOT / "tasks/qcsrr_final_benchmark_selection.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(frame_count: int, fps: float) -> list[int]:
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    import common

    return common.sample_index_list(frame_count, fps, 0.0, mode="legacy")


def _video_metadata(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0)
        width, height = int(stream.width), int(stream.height)
        frame_count = sum(1 for _ in container.decode(stream))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid video stream metadata: {path}")
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}


def _extract_samples(path: Path, indices: list[int], out_dir: Path) -> None:
    import av

    wanted = set(indices)
    seen: set[int] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            destination = out_dir / f"frame-{index:06d}.jpg"
            # shortcut: lossy review cache only; source videos remain authoritative,
            # use them for any boundary judgment that JPEG detail cannot resolve.
            frame.to_image().save(destination, format="JPEG", quality=85, optimize=True)
            seen.add(index)
    if seen != wanted:
        raise RuntimeError(f"decoded frame indices differ for {path.name}: missing={sorted(wanted-seen)}")


def _contact_sheet(frame_paths: list[Path], metadata: dict[str, Any], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    columns, rows, tile_w, tile_h, label_h = 4, 4, 320, 180, 22
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "#f3f3f3")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for slot, frame_path in enumerate(frame_paths):
        index = int(frame_path.stem.split("-")[1])
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
            x = (slot % columns) * tile_w
            y = (slot // columns) * (tile_h + label_h)
            sheet.paste(image, (x, y))
            timestamp = index / float(metadata["fps"])
            draw.text((x + 4, y + tile_h + 5), f"source frame {index} | t={timestamp:.3f}s", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="JPEG", quality=75, optimize=True)


def prepare() -> dict[str, Any]:
    from PIL import Image

    plan = json.loads(PLAN.read_text())
    if plan["source"]["dataset_revision"] != "06224fa1b42e5e35677a8b7f2145e06307a9a4ee":
        raise ValueError("selection dataset revision changed")
    source_root = RUN / "source_videos"
    frames_root = RUN / "source_frames"
    sheets_root = RUN / "annotations/source_contact_sheets"
    records: dict[str, Any] = {}
    for question in plan["questions"]:
        qa_id, relative = str(question["qa_id"]), str(question["video_path"])
        source = source_root / relative
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"staged source video missing or empty: {relative}")
        metadata = _video_metadata(source)
        indices = _sample_indices(metadata["frame_count"], metadata["fps"])
        if len(indices) != 16:
            raise ValueError(f"legacy Stage A sampler returned {len(indices)} frames for {qa_id}")
        frame_dir = frames_root / qa_id
        _extract_samples(source, indices, frame_dir)
        frame_paths = [frame_dir / f"frame-{index:06d}.jpg" for index in indices]
        timestamps = [round(index / metadata["fps"], 6) for index in indices]
        records[qa_id] = {
            "video_path": relative,
            "file_size_bytes": source.stat().st_size,
            "video_sha256": _sha256(source),
            **metadata,
            "sample_mode": "legacy",
            "sample_offset": 0.0,
            "source_frame_indices": indices,
            "source_timestamps_s": timestamps,
            "unique_sample_count": len(set(indices)),
            "source_frames_dir": str(frame_dir.relative_to(RUN)),
            "contact_sheet": str((sheets_root / f"{qa_id}.jpg").relative_to(RUN)),
        }
        _contact_sheet(frame_paths, metadata, RUN / records[qa_id]["contact_sheet"])
        for image_path in frame_paths:
            with Image.open(image_path) as image:
                if image.size != (metadata["width"], metadata["height"]):
                    raise ValueError(f"sampled source frame dimensions changed: {image_path}")

    if len(records) != 12 or len({item["video_path"] for item in records.values()}) != 12:
        raise AssertionError("Phase 4 source set must contain 12 distinct videos")
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "source_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    annotation_template = {
        "schema_version": 1,
        "annotation_status": "source_only_pending_manual_review",
        "blinding": "no options, gold answers, model masks, predictions, or rankings included",
        "questions": {
            q["qa_id"]: {
                "question": q["question"],
                "decomposition": q["question_only_decomposition"],
                "assessability": None,
                "identity_scope": None,
                "adjudication": None,
                "frames": [
                    {
                        "packed_index": i,
                        "source_frame_index": records[q["qa_id"]]["source_frame_indices"][i],
                        "timestamp_s": records[q["qa_id"]]["source_timestamps_s"][i],
                        "entities": {
                            phrase: {
                                "visible": None,
                                "box_xyxy_source_pixels": None,
                                "instance_id": None,
                                "ambiguity": None,
                                "confidence": None,
                            }
                            for phrase in q["entity_prompt_phrases"]
                        },
                        "observed_relation": None,
                        "notes": None,
                    }
                    for i in range(16)
                ],
            }
            for q in plan["questions"]
        },
    }
    annotations_dir = RUN / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / "source_annotations.json").write_text(json.dumps(annotation_template, indent=2) + "\n")
    (annotations_dir / "annotation_protocol.md").write_text(
        "# Source-video annotation protocol\n\n"
        "Review the exact question and original sampled frames only. Do not inspect answer options, gold answers, predictions, masks, rankings, or model output.\n\n"
        "For each entity and sampled frame, mark visibility, source-pixel XYXY box when defensible, whether identity is unique, ambiguity, confidence, and relation evidence. Use `ambiguous`/`unassessable` when the question does not identify one instance or the frame cannot resolve it. Annotations are evaluation-only and must never enter prompts, filtering, or ranking. Record timestamps from original source frame index divided by measured source FPS.\n\n"
        "Every contact sheet preserves all 16 legacy offset-0 samples. JPEG frame images are a compact review cache at source dimensions; original videos are authoritative for fine boundary judgments.\n"
    )
    return {"video_count": len(records), "frame_count": sum(len(x["source_frame_indices"]) for x in records.values()), "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("pass --prepare after the approved 12 video files are staged")
    print(json.dumps(prepare(), indent=2))
