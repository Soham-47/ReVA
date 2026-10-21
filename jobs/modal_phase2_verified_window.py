"""Modal orchestration for the bounded verified-window temporal pilot.

The default local entrypoint is the CPU-only review material job.  The GPU
scoring job is available only through ``--mode score`` and delegates the
review gate, selection provenance, and resumability checks to the local
verified scorer.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal


ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN = "runs/phase2_temporal_verified_window"
REVIEW_RUN = f"{RUN}/review_material_fullres"
SCORE_RUN = f"{RUN}/modal_full"
MANIFEST_REL = "tasks/phase2_temporal_verified_window_manifest.json"
QUESTION_SET_REL = "runs/phase2_temporal_coarsefine/question_set.json"
BASELINE_REL = "runs/stageA_q3vl8b_r16/legacy"
T1_REL = "runs/phase2_temporal_coarsefine/full"
ADAPTER_REL = "runs/stageA_q3vl8b_r16/final_adapter"
DATASET = "ReVA-Benchmark/ReVA"
REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"
SPLITS = ("valQ", "valT")
FULL_COUNT = 24
CONTACT_SAMPLE_COUNT = 48
CONTACT_COLUMNS = 6
CONTACT_CELL_WIDTH = 480
CONTACT_IMAGE_HEIGHT = 270
CONTACT_LABEL_HEIGHT = 22
MAX_REVIEW_RESULT_BYTES = 80 * 1024 * 1024

REVIEW_DEST = ROOT / REVIEW_RUN
SCORE_DEST = ROOT / SCORE_RUN


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return path


def _read_questions(path: Path, selected: list[dict], expected_checksum: str | None = None) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    if payload.get("dataset_revision") != REVISION:
        raise RuntimeError("question set has the wrong dataset revision")
    split_rows = payload.get("splits", {})
    checksum = hashlib.sha256(
        json.dumps(
            split_rows.get("valQ", []) + split_rows.get("valT", []),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if expected_checksum is not None and checksum != expected_checksum:
        raise RuntimeError("question set checksum does not match the verified-window manifest")
    rows: dict[str, dict] = {}
    for split, values in split_rows.items():
        for row in values:
            rows[row["qa_id"]] = {**row, "split": split}
    selected_rows: dict[str, dict] = {}
    for record in selected:
        qa_id = record["qa_id"]
        row = rows.get(qa_id)
        if row is None or row.get("split") != record["split"] or row.get("video_path") != record["video_path"]:
            raise RuntimeError(f"question-set mismatch for {qa_id}")
        selected_rows[qa_id] = row
    return selected_rows


def _selected_paths(selected: list[dict]) -> list[str]:
    paths = sorted({record["video_path"] for record in selected})
    if not paths or len(paths) > FULL_COUNT:
        raise RuntimeError(f"unexpected selected video count: {len(paths)}")
    return paths


def _download_selected(data_root: Path, selected_paths: list[str]) -> None:
    from huggingface_hub import snapshot_download

    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(data_root),
        allow_patterns=selected_paths,
        max_workers=8,
    )
    missing = [relative for relative in selected_paths if not (data_root / _safe_relative_path(relative)).is_file()]
    if missing:
        raise RuntimeError(f"selected dataset videos are missing: {', '.join(missing[:5])}")


def _timestamped_video_info(path: Path) -> dict:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        try:
            fps = float(stream.average_rate or stream.base_rate or 30.0)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = 30.0
        raw_times: list[float] = []
        first_time: float | None = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                raw = float(frame.pts * frame.time_base)
            else:
                raw = index / max(fps, 1e-6)
            if first_time is None:
                first_time = raw
            raw_times.append(raw - first_time)
        if not raw_times:
            raise RuntimeError(f"no decodable frames: {path}")
        if any(a > b for a, b in zip(raw_times, raw_times[1:])):
            times = [index / max(fps, 1e-6) for index in range(len(raw_times))]
            timestamp_source = "index_over_average_fps"
        else:
            times = raw_times
            timestamp_source = "presentation_timestamp"
        duration = times[-1]
        if stream.duration is not None and stream.time_base is not None and first_time is not None:
            duration = max(duration, float(stream.duration * stream.time_base) - first_time)
        return {
            "n_frames": len(times),
            "fps": fps,
            "width": int(stream.width or stream.codec_context.width),
            "height": int(stream.height or stream.codec_context.height),
            "duration": max(float(duration), 0.0),
            "frame_timestamps": times,
            "timestamp_source": timestamp_source,
        }


def _review_video(path: Path, qa_ids: list[str]) -> tuple[bytes, dict]:
    import av
    from PIL import Image, ImageDraw

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        try:
            fps = float(stream.average_rate or stream.base_rate or 30.0)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = 30.0
        decoded: list[tuple[float, Image.Image]] = []
        first_time: float | None = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                raw = float(frame.pts * frame.time_base)
            else:
                raw = index / max(fps, 1e-6)
            if first_time is None:
                first_time = raw
            timestamp = raw - first_time
            thumb = frame.to_image().convert("RGB")
            thumb.thumbnail((CONTACT_CELL_WIDTH - 8, CONTACT_IMAGE_HEIGHT - 8))
            decoded.append((timestamp, thumb.copy()))
        if not decoded:
            raise RuntimeError(f"no decodable frames: {path}")
        if any(a > b for (a, _), (b, _) in zip(decoded, decoded[1:])):
            decoded = [(index / max(fps, 1e-6), image) for index, (_, image) in enumerate(decoded)]
            timestamp_source = "index_over_average_fps"
        else:
            timestamp_source = "presentation_timestamp"
        duration = decoded[-1][0]
        if stream.duration is not None and stream.time_base is not None and first_time is not None:
            duration = max(duration, float(stream.duration * stream.time_base) - first_time)

    targets = [duration * index / (CONTACT_SAMPLE_COUNT - 1) for index in range(CONTACT_SAMPLE_COUNT)]
    chosen = [min(decoded, key=lambda item: abs(item[0] - target)) for target in targets]
    rows = (CONTACT_SAMPLE_COUNT + CONTACT_COLUMNS - 1) // CONTACT_COLUMNS
    header = f"Temporal review | QA IDs: {', '.join(qa_ids)}"
    sheet = Image.new(
        "RGB",
        (CONTACT_COLUMNS * CONTACT_CELL_WIDTH, 42 + rows * (CONTACT_IMAGE_HEIGHT + CONTACT_LABEL_HEIGHT)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), header[:220], fill="black")
    sampled_times: list[float] = []
    for index, (target, (timestamp, image)) in enumerate(zip(targets, chosen)):
        x = (index % CONTACT_COLUMNS) * CONTACT_CELL_WIDTH
        y = 42 + (index // CONTACT_COLUMNS) * (CONTACT_IMAGE_HEIGHT + CONTACT_LABEL_HEIGHT)
        sheet.paste(image, (x + (CONTACT_CELL_WIDTH - image.width) // 2, y + (CONTACT_IMAGE_HEIGHT - image.height) // 2))
        draw.text((x + 4, y + CONTACT_IMAGE_HEIGHT), f"{timestamp:.3f}s", fill="black")
        sampled_times.append(float(timestamp))
    png = io.BytesIO()
    # JPEG keeps the returned review bundle comfortably below Modal result
    # limits while retaining enough detail for small aerial targets.
    sheet.save(png, format="JPEG", quality=85, optimize=True)
    return png.getvalue(), {
        "qa_ids": qa_ids,
        "fps": fps,
        "n_frames": len(decoded),
        "duration_s": float(max(duration, 0.0)),
        "timestamp_source": timestamp_source,
        "sampled_timestamps_s": sampled_times,
    }


def _artifact_manifest(selection: dict, selected: list[dict], *, mode: str, files: list[str]) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "mode": mode,
                "dataset_repo": DATASET,
                "dataset_revision": REVISION,
                "question_set_checksum": selection["question_set_checksum"],
                "selected_count": len(selected),
                "unique_video_count": len(_selected_paths(selected)),
                "selected_qa_ids": [record["qa_id"] for record in selected],
                "files": files,
            },
            indent=2,
        ).encode()
        + b"\n"
    )


def _review_artifacts() -> dict[str, bytes]:
    selection, selected = _read_selection_remote()
    questions = _read_questions(
        Path(WORK) / QUESTION_SET_REL, selected, selection["question_set_checksum"]
    )
    selected_paths = _selected_paths(selected)
    data_root = Path("/tmp/reva_phase2_verified_window_review")
    _download_selected(data_root, selected_paths)
    by_video: dict[str, list[dict]] = {}
    for record in selected:
        by_video.setdefault(record["video_path"], []).append(record)

    artifacts: dict[str, bytes] = {}
    timing_by_video: dict[str, dict] = {}
    for index, video_path in enumerate(selected_paths, 1):
        qa_ids = [record["qa_id"] for record in by_video[video_path]]
        image_bytes, timing = _review_video(data_root / _safe_relative_path(video_path), qa_ids)
        digest = hashlib.sha256(video_path.encode()).hexdigest()[:10]
        sheet_name = f"contact_sheets/clip_{index:02d}_{digest}.jpg"
        timing["contact_sheet"] = sheet_name
        artifacts[sheet_name] = image_bytes
        timing_by_video[video_path] = timing

    packet_items = []
    for record in selected:
        timing = timing_by_video[record["video_path"]]
        packet_items.append(
            {
                "qa_id": record["qa_id"],
                "split": record["split"],
                "question": questions[record["qa_id"]]["question"],
                "contact_sheet": timing["contact_sheet"],
                "video_timing": {
                    "fps": timing["fps"],
                    "n_frames": timing["n_frames"],
                    "duration_s": timing["duration_s"],
                    "timestamp_source": timing["timestamp_source"],
                    "sampled_timestamps_s": timing["sampled_timestamps_s"],
                },
            }
        )
    packet = {
        "schema_version": 1,
        "blind": True,
        "dataset_revision": REVISION,
        "selected_count": len(packet_items),
        "items": packet_items,
    }
    artifacts["review_packet.json"] = (json.dumps(packet, indent=2, ensure_ascii=False) + "\n").encode()
    artifacts["video_timing.json"] = (
        json.dumps(
            {
                "clips": sorted(timing_by_video.values(), key=lambda item: item["contact_sheet"]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode()
    artifacts["review_material_manifest.json"] = _artifact_manifest(
        selection,
        selected,
        mode="review",
        files=sorted(artifacts) + ["review_material_manifest.json"],
    )
    total_bytes = sum(len(contents) for contents in artifacts.values())
    if total_bytes > MAX_REVIEW_RESULT_BYTES:
        raise RuntimeError(
            f"review artifact bundle is {total_bytes:,} bytes; refusing to exceed "
            f"the {MAX_REVIEW_RESULT_BYTES:,}-byte result budget"
        )
    return artifacts


def _safe_log(text: str) -> str:
    text = re.sub(r"hf_[A-Za-z0-9_-]{10,}", "[REDACTED_HF_TOKEN]", text)
    return re.sub(r"(?i)(bearer\s+|(?:token|secret|authorization)[=:]\s*)\S+", r"\1[REDACTED]", text)


def _runtime_env() -> dict[str, str]:
    # Do not pass the process environment wholesale: tokens and unrelated
    # credentials must not be copied into the scoring subprocess environment.
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/root",
        "HF_HOME": "/root/hf_cache",
        "WORK": WORK,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }


def _score_artifacts() -> dict[str, bytes]:
    selection, selected = _read_selection_remote()
    questions_path = Path(WORK) / QUESTION_SET_REL
    selected_questions = _read_questions(questions_path, selected, selection["question_set_checksum"])
    del selected_questions
    selected_paths = _selected_paths(selected)
    work = Path(WORK)
    output_dir = work / SCORE_RUN
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "modal_error.txt").unlink(missing_ok=True)
    logs: list[str] = []
    try:
        data_root = work / "data" / "ReVA"
        _download_selected(data_root, selected_paths)
        source_manifest = {
            relative: _timestamped_video_info(data_root / _safe_relative_path(relative))
            for relative in selected_paths
        }
        (work / "data" / "manifest.json").write_text(
            json.dumps(source_manifest, separators=(",", ":")) + "\n"
        )
        (output_dir / "source_manifest.json").write_text(
            json.dumps(source_manifest, separators=(",", ":")) + "\n"
        )
        (output_dir / "source_manifest_meta.json").write_text(
            json.dumps(
                {
                    "dataset_repo": DATASET,
                    "dataset_revision": REVISION,
                    "selected_qa_ids": [record["qa_id"] for record in selected],
                    "selected_item_count": len(selected),
                    "selected_video_count": len(selected_paths),
                    "question_set_checksum": selection["question_set_checksum"],
                },
                indent=2,
            )
            + "\n"
        )
        command = [
            sys.executable,
            "code/score_temporal_verified.py",
            "--manifest",
            MANIFEST_REL,
            "--subset",
            "full",
            "--run",
            "--question-set",
            QUESTION_SET_REL,
            "--baseline-dir",
            BASELINE_REL,
            "--t1-dir",
            T1_REL,
            "--video-manifest",
            f"{SCORE_RUN}/source_manifest.json",
            "--output-dir",
            SCORE_RUN,
        ]
        result = subprocess.run(
            command,
            cwd=WORK,
            env=_runtime_env(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = _safe_log(result.stdout)
        logs.append("$ " + " ".join(command) + "\n" + output)
        print(output, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"verified scorer exited with status {result.returncode}")
        report_command = [
            sys.executable,
            "code/evaluate_temporal_verified.py",
            "--manifest",
            MANIFEST_REL,
            "--subset",
            "full",
            "--question-set",
            QUESTION_SET_REL,
            "--baseline-dir",
            BASELINE_REL,
            "--t1-dir",
            T1_REL,
            "--manual-dir",
            SCORE_RUN,
            "--output-dir",
            f"{SCORE_RUN}/report",
        ]
        report_result = subprocess.run(
            report_command,
            cwd=WORK,
            env=_runtime_env(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        report_output = _safe_log(report_result.stdout)
        logs.append("$ " + " ".join(report_command) + "\n" + report_output)
        print(report_output, end="", flush=True)
        if report_result.returncode:
            raise RuntimeError(f"verified-window evaluator exited with status {report_result.returncode}")
    except Exception as exc:
        logs.append(f"{type(exc).__name__}: {exc}\n")
        (output_dir / "modal_error.txt").write_text("".join(logs))
        print(logs[-1], end="", flush=True)
    (output_dir / "modal_score.log").write_text("\n".join(logs))
    return {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }


def _read_selection_remote() -> tuple[dict, list[dict]]:
    path = Path(WORK) / MANIFEST_REL
    payload = json.loads(path.read_text())
    if payload.get("dataset_revision") != REVISION:
        raise RuntimeError("remote verified-window manifest has the wrong dataset revision")
    ids = payload.get("subsets", {}).get("full")
    records = payload.get("records")
    if not isinstance(ids, list) or len(ids) != FULL_COUNT or len(ids) != len(set(ids)):
        raise RuntimeError("remote verified-window full subset must contain 24 unique IDs")
    by_id = {record.get("qa_id"): record for record in records or []}
    if set(ids) - by_id.keys():
        raise RuntimeError("remote verified-window subset references missing records")
    selected = [by_id[qa_id] for qa_id in ids]
    for record in selected:
        if record.get("split") not in SPLITS or record.get("event_form") != "first_appearance":
            raise RuntimeError(f"invalid remote selected record: {record.get('qa_id')}")
        _safe_relative_path(record["video_path"])
    return payload, selected


review_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("av==19.0.1", "pillow", "huggingface-hub==2.2.0")
    .add_local_file(str(ROOT / MANIFEST_REL), f"{WORK}/{MANIFEST_REL}", copy=True)
    .add_local_file(str(ROOT / QUESTION_SET_REL), f"{WORK}/{QUESTION_SET_REL}", copy=True)
    .env({"WORK": WORK, "HF_HOME": "/root/hf_cache"})
)

score_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "transformers==5.19.0",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "huggingface-hub==2.2.0",
        "av==19.0.1",
        "numpy",
        "pyyaml",
        "tqdm",
    )
    .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
    .add_local_file(str(ROOT / MANIFEST_REL), f"{WORK}/{MANIFEST_REL}", copy=True)
    .add_local_file(str(ROOT / QUESTION_SET_REL), f"{WORK}/{QUESTION_SET_REL}", copy=True)
    .add_local_dir(str(ROOT / ADAPTER_REL), f"{WORK}/{ADAPTER_REL}", copy=True)
    .add_local_file(
        str(ROOT / BASELINE_REL / "scores_valQ.jsonl"),
        f"{WORK}/{BASELINE_REL}/scores_valQ.jsonl",
        copy=True,
    )
    .add_local_file(
        str(ROOT / BASELINE_REL / "scores_valT.jsonl"),
        f"{WORK}/{BASELINE_REL}/scores_valT.jsonl",
        copy=True,
    )
    .add_local_file(
        str(ROOT / T1_REL / "scores_valQ_T1.jsonl"),
        f"{WORK}/{T1_REL}/scores_valQ_T1.jsonl",
        copy=True,
    )
    .add_local_file(
        str(ROOT / T1_REL / "scores_valT_T1.jsonl"),
        f"{WORK}/{T1_REL}/scores_valT_T1.jsonl",
        copy=True,
    )
    .env({"WORK": WORK, "HF_HOME": "/root/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
)
if SCORE_DEST.is_dir():
    score_image = score_image.add_local_dir(str(SCORE_DEST), f"{WORK}/{SCORE_RUN}", copy=True)

app = modal.App("reva-phase2-verified-window")


@app.function(image=review_image, cpu=8, memory=32768, timeout=60 * 60)
def review_material() -> dict[str, bytes]:
    """Download only selected review clips and return blind contact-sheet artifacts."""

    return _review_artifacts()


@app.function(image=score_image, gpu="A100", cpu=8, memory=65536, timeout=15 * 60)
def score_full() -> dict[str, bytes]:
    """Download selected clips and run the gated full verified scorer."""

    return _score_artifacts()


def _materialize(result: dict[str, bytes], destination: Path, *, refuse_existing: bool) -> None:
    if refuse_existing and destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing review material: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in result.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe Modal result path: {name}")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        print(f"saved {path} ({len(contents):,} bytes)")


@app.local_entrypoint()
def main(mode: str = "review") -> None:
    if mode not in {"review", "score"}:
        raise ValueError("mode must be review or score")
    if mode == "review":
        result = review_material.remote()
        _materialize(result, REVIEW_DEST, refuse_existing=True)
    else:
        result = score_full.remote()
        _materialize(result, SCORE_DEST, refuse_existing=False)
    if "modal_error.txt" in result:
        raise RuntimeError(f"Modal {mode} job failed; partial artifacts were saved")
