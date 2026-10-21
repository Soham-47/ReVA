"""Fetch the frozen Phase 0 sample and create visual review sheets on Modal CPU."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import modal


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/root/ReVA")
SELECTION = "results/phase0_temporal_geometry_review/selection.json"
REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"
DATASET = "ReVA-Benchmark/ReVA"
OUTPUT = ROOT / "results/phase0_temporal_geometry_review"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("av==19.0.1", "pillow", "huggingface-hub==2.2.0")
    .add_local_file(str(ROOT / SELECTION), f"{WORK}/{SELECTION}", copy=True)
)
app = modal.App("reva-phase0-review", image=image)


@app.function(cpu=4, memory=32768, timeout=60 * 60)
def make_contact_sheets() -> tuple[bytes, bytes]:
    import av
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image, ImageDraw

    selection = json.loads((WORK / SELECTION).read_text())
    records = selection["records"]
    if len(records) != 60:
        raise RuntimeError(f"expected 60 review items, got {len(records)}")

    by_video: dict[str, list[dict]] = {}
    for row in records:
        by_video.setdefault(row["video_path"], []).append(row)
    data_root = Path("/tmp/reva_phase0_data")
    data_root.mkdir(parents=True, exist_ok=True)
    train_json = hf_hub_download(
        repo_id=DATASET,
        repo_type="dataset",
        filename="train.json",
        revision=REVISION,
        local_dir=str(data_root),
    )
    train_rows = json.loads(Path(train_json).read_text())["QA"]
    train_by_id = {row["qa_id"]: row for row in train_rows}
    for row in records:
        source = train_by_id.get(row["qa_id"])
        if source is None or source["video_path"] != row["video_path"]:
            raise RuntimeError(f"dataset revision mismatch for {row['qa_id']}")

    video_paths = sorted(by_video)
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(data_root),
        allow_patterns=video_paths,
        max_workers=8,
    )

    cell_width, image_height, label_height = 200, 112, 20
    columns, sample_count = 6, 24
    metadata: dict[str, dict] = {}
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for video_path in video_paths:
            clip = data_root / PurePosixPath(video_path)
            if not clip.is_file():
                raise RuntimeError(f"selected clip missing: {video_path}")
            container = av.open(str(clip))
            stream = container.streams.video[0]
            try:
                fps = float(stream.average_rate)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 30.0
            frames: list[tuple[float, Image.Image]] = []
            for index, frame in enumerate(container.decode(stream)):
                timestamp = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else index / fps
                )
                thumb = frame.to_image().convert("RGB")
                thumb.thumbnail((cell_width - 8, image_height - 8))
                frames.append((timestamp, thumb.copy()))
            if not frames:
                container.close()
                raise RuntimeError(f"no decodable frames: {video_path}")

            duration = frames[-1][0]
            if stream.duration is not None and stream.time_base is not None:
                duration = max(duration, float(stream.duration * stream.time_base))
            container.close()
            target_times = [duration * i / (sample_count - 1) for i in range(sample_count)]
            selected_frames = [min(frames, key=lambda item: abs(item[0] - t)) for t in target_times]

            rows = by_video[video_path]
            title = f"{video_path} | " + ", ".join(
                f"{row['qa_id']}:{row['task'][:1]}-{row['review_role']}" for row in rows
            )
            rows_count = (sample_count + columns - 1) // columns
            sheet = Image.new("RGB", (columns * cell_width, 36 + rows_count * (image_height + label_height)), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((6, 6), title[:180], fill="black")
            sample_meta = []
            for i, (target, (timestamp, thumb)) in enumerate(zip(target_times, selected_frames)):
                x = (i % columns) * cell_width
                y = 36 + (i // columns) * (image_height + label_height)
                sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y + (image_height - thumb.height) // 2))
                draw.text((x + 4, y + image_height), f"{timestamp:.3f}s", fill="black")
                sample_meta.append({"target_time_s": target, "decoded_time_s": timestamp})

            digest = hashlib.sha256(video_path.encode()).hexdigest()[:8]
            qa_slug = rows[0]["qa_id"]
            name = f"{qa_slug}_{digest}.png"
            png = io.BytesIO()
            sheet.save(png, format="PNG", optimize=True)
            info = tarfile.TarInfo(name=f"contact_sheets/{name}")
            info.size = len(png.getvalue())
            archive.addfile(info, io.BytesIO(png.getvalue()))
            metadata[video_path] = {
                "contact_sheet": f"contact_sheets/{name}",
                "qa_items": [
                    {key: row[key] for key in ("qa_id", "task", "review_role", "gold", "baseline_prediction", "question", "options")}
                    for row in rows
                ],
                "fps": fps,
                "decoded_frame_count": len(frames),
                "duration_s": duration,
                "sampled_frames": sample_meta,
            }
        meta_bytes = json.dumps(metadata, indent=2).encode() + b"\n"
        info = tarfile.TarInfo(name="metadata.json")
        info.size = len(meta_bytes)
        archive.addfile(info, io.BytesIO(meta_bytes))

    return archive_buffer.getvalue(), meta_bytes


@app.local_entrypoint()
def main() -> None:
    import tarfile

    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive_bytes, metadata_bytes = make_contact_sheets.remote()
    archive_path = OUTPUT / "contact_sheets.tar.gz"
    archive_path.write_bytes(archive_bytes)
    (OUTPUT / "video_metadata.json").write_bytes(metadata_bytes)
    sheets_dir = OUTPUT / "contact_sheets"
    sheets_dir.mkdir(exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith("contact_sheets/") or ".." in PurePosixPath(member.name).parts:
                continue
            archive.extract(member, OUTPUT)
    print(f"saved {len(list(sheets_dir.glob('*.png')))} sheets to {sheets_dir}")
