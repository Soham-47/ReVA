"""Render the frozen blind temporal audit sample into CPU-only review sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from textwrap import wrap


DEFAULT_WORK_ROOT = "/scratch/ambpdc/soham/ReVA"
SAMPLE_REL = "tasks/temporal_training_audit_blind_review.json"
OUTPUT_REL = "runs/temporal_training_data_audit/review_material"
OFFSETS = (("0.0", 0.0), ("0.5", 0.5))
SHEET_COLUMNS = 4
CELL_WIDTH = 300
IMAGE_HEIGHT = 170
LABEL_HEIGHT = 24
SECTION_HEIGHT = 28
HEADER_HEIGHT = 96


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe relative video path: {value!r}")
    return path


def _legacy_indices(n_frames: int, fps: float, offset: float) -> list[int]:
    n = max(int(n_frames), 1)
    rate = max(float(fps), 1e-6)
    k = max(16, min(16, round(n / rate * 4)))
    k = min(k, max(n, 2))
    k -= k % 2
    k = max(k, 2)
    return [min(n - 1, max(0, math.floor((i + offset) * n / k))) for i in range(k)]


def _check_sample(payload: dict) -> list[dict]:
    assert payload.get("schema_version") == 1
    assert payload.get("blind_to_options_and_gold") is True
    records = payload.get("records")
    assert isinstance(records, list) and len(records) == 100, "expected 100 frozen audit records"
    assert len({record.get("case_id") for record in records}) == 100, "case IDs are not unique"
    assert len({record.get("video_path") for record in records}) == 100, "video paths are not unique"

    for record in records:
        case_id = record.get("case_id")
        assert isinstance(case_id, str) and case_id.replace("_", "").replace("-", "").isalnum()
        assert "options" not in record and "gold" not in record and "correct_answer" not in record
        video = record.get("video")
        assert isinstance(video, dict)
        n_frames = int(video["n_frames"])
        fps = float(video["fps"])
        assert n_frames > 0 and math.isfinite(fps) and fps > 0
        path = _safe_relative_path(record["video_path"])
        assert path.parts and path.suffix.lower() == ".mp4"
        samples = record.get("legacy_frame_samples")
        assert isinstance(samples, dict)
        for offset_key, offset in OFFSETS:
            sample = samples.get(offset_key)
            assert isinstance(sample, dict), f"{case_id}: missing legacy offset {offset_key}"
            indices = sample.get("frame_indices")
            timestamps = sample.get("timestamps_s")
            assert isinstance(indices, list) and isinstance(timestamps, list)
            assert len(indices) == len(timestamps) == len(_legacy_indices(n_frames, fps, offset))
            assert indices == _legacy_indices(n_frames, fps, offset), (
                f"{case_id}: frame indices drifted for offset {offset_key}"
            )
            assert timestamps == [round(index / fps, 4) for index in indices], (
                f"{case_id}: timestamps drifted for offset {offset_key}"
            )
            assert all(isinstance(index, int) and 0 <= index < n_frames for index in indices), (
                f"{case_id}: frame index out of bounds"
            )
    return records


def _load_frames(record: dict, clip: Path) -> dict[int, object]:
    import av

    case_id = record["case_id"]
    expected_count = int(record["video"]["n_frames"])
    requested = {
        index
        for offset_key, _ in OFFSETS
        for index in record["legacy_frame_samples"][offset_key]["frame_indices"]
    }
    frames: dict[int, object] = {}
    try:
        with av.open(str(clip)) as container:
            streams = container.streams.video
            if not streams:
                raise RuntimeError("clip has no video stream")
            for index, frame in enumerate(container.decode(streams[0])):
                if index in requested:
                    frames[index] = frame.to_image().convert("RGB")
            actual_count = index + 1 if "index" in locals() else 0
    except Exception as exc:
        raise RuntimeError(f"{case_id}: could not decode {clip}: {exc}") from exc
    assert actual_count == expected_count, (
        f"{case_id}: decoded {actual_count} frames, expected {expected_count}"
    )
    assert requested.issubset(frames), f"{case_id}: one or more recorded frames did not render"
    assert all(0 <= index < actual_count for index in requested), (
        f"{case_id}: recorded frame index out of decoded bounds"
    )
    return frames


def _draw_frame(draw, sheet, image, x: int, y: int, label: str) -> None:
    image.thumbnail((CELL_WIDTH - 8, IMAGE_HEIGHT - 8))
    sheet.paste(
        image,
        (x + (CELL_WIDTH - image.width) // 2, y + (IMAGE_HEIGHT - image.height) // 2),
    )
    draw.rectangle((x, y, x + CELL_WIDTH - 1, y + IMAGE_HEIGHT - 1), outline="#bbbbbb")
    draw.text((x + 4, y + IMAGE_HEIGHT + 3), label, fill="black")


def _render_sheet(record: dict, frames: dict[int, object], destination: Path) -> None:
    from PIL import Image, ImageDraw

    question = str(record["question"])
    header_lines = [
        f"{record['case_id']} | {record['qa_id']} | {record['video_path']}",
        *wrap(question, width=105)[:2],
    ]
    section_rows = [
        (len(record["legacy_frame_samples"][offset_key]["frame_indices"]) + SHEET_COLUMNS - 1)
        // SHEET_COLUMNS
        for offset_key, _ in OFFSETS
    ]
    sheet_height = HEADER_HEIGHT + sum(
        SECTION_HEIGHT + rows * (IMAGE_HEIGHT + LABEL_HEIGHT) for rows in section_rows
    )
    sheet = Image.new("RGB", (SHEET_COLUMNS * CELL_WIDTH, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    for line_index, line in enumerate(header_lines):
        draw.text((8, 8 + line_index * 18), line, fill="black")

    y = HEADER_HEIGHT
    for (offset_key, _), rows in zip(OFFSETS, section_rows):
        draw.text((8, y + 5), f"offset {offset_key}", fill="black")
        y += SECTION_HEIGHT
        sample = record["legacy_frame_samples"][offset_key]
        for sample_index, (frame_index, timestamp) in enumerate(
            zip(sample["frame_indices"], sample["timestamps_s"])
        ):
            x = (sample_index % SHEET_COLUMNS) * CELL_WIDTH
            cell_y = y + (sample_index // SHEET_COLUMNS) * (IMAGE_HEIGHT + LABEL_HEIGHT)
            _draw_frame(
                draw,
                sheet,
                frames[frame_index].copy(),
                x,
                cell_y,
                f"frame {frame_index} | t={timestamp:.4f}s",
            )
        y += rows * (IMAGE_HEIGHT + LABEL_HEIGHT)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}.", suffix=".jpg", delete=False
        ) as handle:
            temporary = Path(handle.name)
        sheet.save(temporary, format="JPEG", quality=90, optimize=True)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest(source_bytes: bytes, records: list[dict], files: list[str]) -> dict:
    return {
        "schema_version": 1,
        "purpose": "blind temporal training-data feasibility audit review material",
        "source_sample": SAMPLE_REL,
        "source_sample_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "record_count": len(records),
        "unique_case_count": len({record["case_id"] for record in records}),
        "unique_video_count": len({record["video_path"] for record in records}),
        "offsets": [offset_key for offset_key, _ in OFFSETS],
        "files": files,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [
            {
                "case_id": record["case_id"],
                "qa_id": record["qa_id"],
                "video_path": record["video_path"],
                "question": record["question"],
                "video": record["video"],
                "legacy_frame_samples": record["legacy_frame_samples"],
                "contact_sheet": f"{record['case_id']}.jpg",
            }
            for record in records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", default=DEFAULT_WORK_ROOT, type=Path)
    args = parser.parse_args()
    work_root = args.work_root.expanduser().resolve()
    sample_path = work_root / SAMPLE_REL
    dataset_root = work_root / "data" / "ReVA"
    output_dir = work_root / OUTPUT_REL
    if not sample_path.is_file():
        raise FileNotFoundError(f"frozen audit sample is missing: {sample_path}")
    source_bytes = sample_path.read_bytes()
    records = _check_sample(json.loads(source_bytes))
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for record in records:
        relative = _safe_relative_path(record["video_path"])
        clip = (dataset_root / Path(*relative.parts)).resolve()
        if dataset_root.resolve() not in clip.parents or not clip.is_file():
            raise FileNotFoundError(f"{record['case_id']}: selected clip is missing: {clip}")
        frames = _load_frames(record, clip)
        destination = output_dir / f"{record['case_id']}.jpg"
        _render_sheet(record, frames, destination)
        files.append(destination.name)

    files.sort()
    manifest_path = output_dir / "review_material_manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(source_bytes, records, files + [manifest_path.name]), indent=2) + "\n"
    )
    print(f"wrote {len(records)} contact sheets and {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
