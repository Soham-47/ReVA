"""Charades-STA interval SFT/GRPO pilot.

This path is deliberately separate from ``code/train.py``: ReVA's trainer
optimizes one multiple-choice token, while this pilot needs a strict temporal
interval and a verifiable IoU reward.  The module keeps its pure data and
reward helpers importable on a login node; ML/video dependencies are imported
only by the training commands.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import signal
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import common as C


DATASET_REPO = "yeliudev/VideoMind-Dataset"
DATASET_REVISION = "3518d8e8c2c7c1ceec4b242afc07162bbafa33d8"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
EXPECTED_TRAIN_ROWS = 12_408
EXPECTED_TEST_ROWS = 3_720
INTERVAL_FORMAT = "start=<seconds> end=<seconds>"
INTERVAL_RE = re.compile(
    r"^\s*start\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"end\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)
_STOP_REQUESTED = False


def _request_stop(_signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _install_stop_handler() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGUSR1, _request_stop)


def parse_interval(text: str, duration: float) -> tuple[float, float] | None:
    """Parse exactly ``start=<number> end=<number>`` without clipping."""

    duration_num = _number(duration)
    if not isinstance(text, str) or duration_num is None or not math.isfinite(duration_num) or duration_num <= 0:
        return None
    match = INTERVAL_RE.fullmatch(text)
    if match is None:
        return None
    start, end = (float(value) for value in match.groups())
    if not all(math.isfinite(value) for value in (start, end)):
        return None
    if not (0.0 <= start < end <= duration_num):
        return None
    return start, end


def format_interval(start: float, end: float, duration: float | None = None) -> str:
    """Format a gold interval after applying the same strict validation."""

    if duration is None:
        duration = max(float(end), 1e-9)
    if parse_interval(f"start={start:.6f} end={end:.6f}", duration) is None:
        raise ValueError(f"invalid interval [{start}, {end}] for duration {duration}")
    return f"start={float(start):.6f} end={float(end):.6f}"


def interval_iou(pred: tuple[float, float] | None, gold: tuple[float, float]) -> float:
    """Continuous temporal IoU; invalid predictions receive zero."""

    if pred is None:
        return 0.0
    ps, pe = pred
    gs, ge = gold
    overlap = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return overlap / union if union > 0 else 0.0


def interval_reward(text: str, gold: tuple[float, float], duration: float) -> float:
    return interval_iou(parse_interval(text, duration), gold)


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _duration_map(payload: Any) -> dict[str, float]:
    values = payload.get("durations", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, dict):
        raise ValueError("durations.json must contain an object mapping video IDs to seconds")
    result: dict[str, float] = {}
    for key, value in values.items():
        number = _number(value.get("duration") if isinstance(value, dict) else value)
        if number is not None and math.isfinite(number) and number > 0:
            result[str(key)] = number
    return result


def _lookup_duration(durations: dict[str, float], video_id: str) -> float | None:
    candidates = (video_id, Path(video_id).stem, Path(video_id).name)
    for candidate in candidates:
        if candidate in durations:
            return durations[candidate]
    return None


def canonical_row(
    row: dict[str, Any],
    index: int,
    durations: dict[str, float],
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert common Charades-STA/HF field variants and reject bad labels."""

    video_path = _first(row, "video_path", "path", "file_name")
    video_id = _first(row, "video_id", "video", "vid", "video_name") or video_path
    query = _first(row, "query", "sentence", "text", "description", "caption")
    start = _first(row, "start", "start_time", "start_sec", "start_seconds")
    end = _first(row, "end", "end_time", "end_sec", "end_seconds")
    segment = _first(row, "segment", "timestamp", "timestamps", "interval", "answer", "target")
    if (start is None or end is None) and isinstance(segment, (list, tuple)) and len(segment) == 2:
        start, end = segment
    if not isinstance(video_id, str) or not video_id.strip():
        return None, "missing video_id"
    if not isinstance(query, str) or not query.strip():
        return None, "missing query"
    video_id = video_id.strip()
    duration = _number(_first(row, "duration", "video_duration", "duration_sec"))
    duration = duration if duration is not None else _lookup_duration(durations, video_id)
    start_num, end_num = _number(start), _number(end)
    if duration is None:
        return None, "missing duration"
    if start_num is None or end_num is None:
        return None, "non-numeric interval"
    if not (math.isfinite(duration) and duration > 0):
        return None, "invalid duration"
    if not (math.isfinite(start_num) and math.isfinite(end_num) and 0 <= start_num < end_num <= duration):
        return None, "interval outside duration"
    qa_id = str(_first(row, "id", "query_id", "annotation_id", "uid") or f"charades_{index:06d}")
    return {
        "qa_id": qa_id,
        "video_id": video_id,
        "video_path": str(video_path or video_id),
        "query": query.strip(),
        "start": start_num,
        "end": end_num,
        "duration": duration,
        "source": row,
    }, None


def load_charades_sta_rows(
    annotations_path: Path,
    durations: dict[str, float],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the released ``video start end##query`` annotation format."""

    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, line in enumerate(annotations_path.read_text().splitlines()):
        if not line.strip():
            continue
        info, separator, query = line.partition("##")
        parts = info.split()
        if not separator or len(parts) != 3:
            rejected.append({"index": index, "reason": "malformed Charades-STA line", "line": line})
            continue
        video_id, start, end = parts
        declared_duration = _lookup_duration(durations, video_id)
        end_num = _number(end)
        raw = {
            "id": f"charades_{split}_{index:06d}",
            "video_id": video_id,
            "video_path": f"videos_3fps_480_noaudio/{video_id}.mp4",
            "query": query.strip(),
            "start": start,
            "end": end,
            # The release durations end at the last frame timestamp; annotations
            # can round to the next frame boundary. PyAV validates selected rows.
            "duration": max(declared_duration, end_num) if declared_duration is not None and end_num is not None else declared_duration,
        }
        normalized, reason = canonical_row(raw, index, durations)
        if normalized is None:
            rejected.append({"index": index, "reason": reason, "line": line})
        else:
            normalized["duration_metadata"] = declared_duration
            accepted.append(normalized)
    ids = [row["qa_id"] for row in accepted]
    if len(ids) != len(set(ids)):
        raise ValueError(f"canonical {split} rows contain duplicate qa_id values")
    return accepted, rejected


def split_by_video(
    rows: list[dict[str, Any]],
    sample_rows: int = 1024,
    dev_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sample train rows while keeping the dev videos disjoint from train."""

    if sample_rows <= 0 or not 0 < dev_fraction < 1:
        raise ValueError("sample_rows must be positive and dev_fraction must be in (0, 1)")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    videos = list(grouped)
    if len(videos) < 2:
        raise ValueError("at least two videos are required for a video-held-out dev split")
    rng = random.Random(seed)
    rng.shuffle(videos)
    target_dev = max(1, round(len(rows) * dev_fraction))
    dev_videos: list[str] = []
    dev_count = 0
    for video in videos:
        if dev_count >= target_dev and dev_videos:
            break
        dev_videos.append(video)
        dev_count += len(grouped[video])
    dev_set = set(dev_videos)
    dev_rows = [row for row in rows if row["video_id"] in dev_set]
    train_pool = [row for row in rows if row["video_id"] not in dev_set]
    rng.shuffle(train_pool)
    train_rows = train_pool[: min(sample_rows, len(train_pool))]
    if not train_rows or not dev_rows or {r["video_id"] for r in train_rows} & {r["video_id"] for r in dev_rows}:
        raise AssertionError("video split is empty or leaked")
    return train_rows, dev_rows


def _scheduled_rows(rows: list[dict[str, Any]], steps: int, seed: int) -> list[dict[str, Any]]:
    """Use the same deterministic prompt sequence for SFT and GRPO."""

    if not rows or steps <= 0:
        raise ValueError("training rows and optimizer steps must be non-empty and positive")
    rng = random.Random(seed)
    order: list[int] = []
    while len(order) < steps:
        epoch = list(range(len(rows)))
        rng.shuffle(epoch)
        order.extend(epoch)
    return [rows[index] for index in order[:steps]]


def _safe_output(path: Path, *, resume: bool = False) -> Path:
    root = Path(__file__).resolve().parents[1]
    path = path.expanduser().resolve()
    allowed_runtime = root / "runs" / "charades_grounding"
    if root == path or (root in path.parents and allowed_runtime not in path.parents and path != allowed_runtime):
        raise ValueError(f"runtime artifacts must be outside source checkout: {path}")
    existing = list(path.iterdir()) if path.exists() else []
    if existing and (not resume or any(not item.name.startswith("checkpoint-") for item in existing)):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).expanduser().resolve()
    train_annotations = root / args.train_annotations
    durations_json = root / args.durations_json
    test_annotations = root / args.test_annotations
    for path in (train_annotations, durations_json, test_annotations):
        if not path.is_file():
            raise FileNotFoundError(path)
    with durations_json.open() as handle:
        durations = _duration_map(json.load(handle))
    rows, rejected = load_charades_sta_rows(train_annotations, durations, "train")
    source_train_rows = sum(bool(line.strip()) for line in train_annotations.read_text().splitlines())
    source_test_rows = sum(bool(line.strip()) for line in test_annotations.read_text().splitlines())
    if source_train_rows != EXPECTED_TRAIN_ROWS or source_test_rows != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"pinned release count mismatch: train={source_train_rows} (expected {EXPECTED_TRAIN_ROWS}), "
            f"test={source_test_rows} (expected {EXPECTED_TEST_ROWS})"
        )
    train_rows, dev_rows = split_by_video(rows, args.sample_rows, args.dev_fraction, args.seed)
    selected_rows, media_rejected, media_summary = _validate_media_rows(train_rows + dev_rows, root)
    rejected.extend(media_rejected)
    valid_ids = {row["qa_id"] for row in selected_rows}
    train_rows = [row for row in train_rows if row["qa_id"] in valid_ids]
    dev_rows = [row for row in dev_rows if row["qa_id"] in valid_ids]
    if not train_rows or not dev_rows:
        raise ValueError("decoded media validation emptied the train or dev split")
    out = _safe_output(Path(args.output_dir))
    (out / "train.json").write_text(json.dumps(train_rows, indent=2) + "\n")
    (out / "dev.json").write_text(json.dumps(dev_rows, indent=2) + "\n")
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "source_train_rows": source_train_rows,
        "source_test_rows": source_test_rows,
        "accepted_rows": len(rows),
        "rejected_rows": len(rejected),
        "sampled_train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_videos": len({r["video_id"] for r in train_rows}),
        "dev_videos": len({r["video_id"] for r in dev_rows}),
        "seed": args.seed,
        "media_validation": media_summary,
        "official_test_used_for_selection": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def preflight(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).expanduser().resolve()
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    required_free = 24 * 1024**3
    status: dict[str, Any] = {
        "dataset_root": str(root),
        "free_bytes": free,
        "free_gib": round(free / 1024**3, 3),
        "media_archive_required_gib": 11.6,
        "requirements": {
            "dataset_root": root.is_dir(),
            "train_annotations": (root / args.train_annotations).is_file(),
            "test_annotations": (root / args.test_annotations).is_file(),
            "durations_json": (root / args.durations_json).is_file(),
            "enough_free_space_for_download_and_extraction": free >= required_free,
            "media_files_present": any(root.rglob("*.mp4")) if root.is_dir() else False,
        },
    }
    try:
        import importlib.util

        status["python_packages"] = {
            name: bool(importlib.util.find_spec(name))
            for name in ("torch", "transformers", "peft", "accelerate", "av")
        }
        import torch

        status["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "count": int(torch.cuda.device_count()),
            "bf16": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
        }
    except Exception as exc:
        status["runtime_error"] = f"{type(exc).__name__}: {exc}"
    status["ready_for_training"] = all(status["requirements"].values()) and bool(
        status.get("cuda", {}).get("available")
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ready_for_training"] else 2


def _resolve_media(dataset_root: Path, row: dict[str, Any]) -> Path | None:
    """Resolve extracted archive layouts without downloading or guessing URLs."""

    relative = str(row.get("video_path") or row["video_id"])
    candidates = [
        dataset_root / relative,
        dataset_root / "videos_3fps_480_noaudio" / relative,
        dataset_root / "videos_3fps_480_noaudio" / f"{row['video_id']}.mp4",
        dataset_root / f"{row['video_id']}.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    basename = Path(relative).name
    matches = list(dataset_root.rglob(basename)) if basename else []
    return matches[0] if len(matches) == 1 and matches[0].is_file() else None


def _decoded_duration(path: Path) -> tuple[float, float, int]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0.0)
        if fps <= 0:
            raise ValueError(f"video has no usable frame rate: {path}")
        frames = 0
        first_timestamp = last_timestamp = None
        last_frame_duration = 0.0
        for frame in container.decode(stream):
            frames += 1
            time_base = float(frame.time_base or stream.time_base)
            if frame.pts is not None:
                timestamp = frame.pts * time_base
                first_timestamp = timestamp if first_timestamp is None else first_timestamp
                last_timestamp = timestamp
            if frame.duration:
                last_frame_duration = frame.duration * time_base
        if frames < 1:
            raise ValueError(f"video has no decodable frames: {path}")
        if first_timestamp is not None and last_timestamp is not None:
            actual = last_timestamp - first_timestamp + (last_frame_duration or 1.0 / fps)
        else:
            actual = frames / fps
    if not math.isfinite(actual) or actual <= 0:
        raise ValueError(f"decoded duration is invalid: {path}")
    return actual, fps, frames


def _validate_media_rows(rows: list[dict[str, Any]], dataset_root: Path):
    """Validate annotation bounds against decoded frames; never clip labels."""

    try:
        import av  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("media validation requires PyAV; install requirements before prepare") from exc
    cache: dict[str, tuple[float, float, int]] = {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        path = _resolve_media(dataset_root, row)
        if path is None:
            raise FileNotFoundError(f"no extracted media for {row['video_id']} under {dataset_root}")
        key = str(path.resolve())
        if key not in cache:
            cache[key] = _decoded_duration(path)
        actual, fps, frames = cache[key]
        if row["end"] > actual + 1e-6 or row["start"] < -1e-6:
            rejected.append({
                "qa_id": row["qa_id"],
                "reason": "interval outside decoded duration",
                "decoded_duration": actual,
                "decoded_fps": fps,
                "decoded_frames": frames,
                "row": row,
            })
            continue
        row = dict(row)
        row["media_path"] = str(path)
        row["decoded_duration"] = actual
        row["decoded_fps"] = fps
        row["decoded_frames"] = frames
        row["duration"] = actual
        accepted.append(row)
    if not accepted:
        raise ValueError("media validation rejected every training row")
    return accepted, rejected, {
        "videos_decoded": len(cache),
        "method": "PyAV presentation timestamps with frame-count/FPS fallback",
    }


def _load_video(path: Path, frame_count: int = 16):
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 3.0)
        declared = int(stream.frames or 0)
        frames: dict[int, Any] = {}
        all_frames: list[Any] | None = None
        if declared < 2:
            all_frames = []
        indices = None if all_frames is not None else [
            min(declared - 1, round(i * (declared - 1) / (frame_count - 1)))
            for i in range(frame_count)
        ]
        wanted = set(indices or [])
        for number, frame in enumerate(container.decode(stream)):
            if all_frames is not None:
                all_frames.append(frame.to_ndarray(format="rgb24"))
            elif number in wanted:
                frames[number] = frame.to_ndarray(format="rgb24")
        if all_frames is not None:
            if not all_frames:
                raise ValueError(f"video has no decodable frames: {path}")
            declared = len(all_frames)
            indices = [min(declared - 1, round(i * (declared - 1) / (frame_count - 1))) for i in range(frame_count)]
            sampled = [all_frames[index] for index in indices]
        else:
            if wanted - frames.keys():
                raise ValueError(f"container frame count does not match decoded stream: {path}")
            sampled = [frames[index] for index in indices]
        if any(frame is None for frame in sampled):
            raise ValueError(f"could not decode requested frames: {path}")
    metadata = C._video_metadata(
        total_num_frames=declared,
        fps=fps,
        frames_indices=indices,
        width=int(sampled[0].shape[1]),
        height=int(sampled[0].shape[0]),
        duration=declared / max(fps, 1e-6),
    )
    return np.stack(sampled), metadata


def _processor_call(processor, text: str, frames: Any, metadata: Any):
    base = dict(
        text=[text], videos=[frames], video_metadata=[metadata],
        do_sample_frames=False, size=C.VIDEO_SIZE,
        cap_pixels_per_frame=True, return_tensors="pt",
    )
    last: Exception | None = None
    optional_keys = ("cap_pixels_per_frame", "do_sample_frames", "size")
    for remove_count in range(len(optional_keys) + 1):
        kwargs = dict(base)
        for key in optional_keys[:remove_count]:
            kwargs.pop(key, None)
        try:
            return processor(**kwargs)
        except TypeError as exc:
            last = exc
            if not any(key in str(exc) for key in (*optional_keys, "video_metadata")):
                raise
            if "video_metadata" in str(exc):
                break
    raise last or RuntimeError("processor call failed")


def _prompt(processor, row: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "Locate the described event in the video. Return exactly one finite interval "
            "using the format start=<seconds> end=<seconds>. Do not add any other text."
        )}]},
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": f"The clip duration is {row['duration']:.3f} seconds. When is this visible: {row['query']}?"},
        ]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _model_inputs(processor, row: dict[str, Any], dataset_root: Path):
    media = Path(row.get("media_path", "")) if row.get("media_path") else _resolve_media(dataset_root, row)
    if media is None:
        raise FileNotFoundError(f"no media for {row['video_id']} under {dataset_root}")
    frames, metadata = _load_video(media)
    raw = _processor_call(processor, _prompt(processor, row), frames, metadata)
    allowed = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}
    result = {key: value for key, value in raw.items() if key in allowed}
    required = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw"}
    missing = required - result.keys()
    if missing:
        raise KeyError(f"processor missing model inputs: {sorted(missing)}")
    return result


def _append_tokens(inputs: dict[str, Any], target_ids: list[int], im_end_id: int):
    import torch

    ids = inputs["input_ids"]
    suffix = torch.tensor(target_ids + [im_end_id], dtype=ids.dtype, device=ids.device).unsqueeze(0)
    inputs = dict(inputs)
    inputs["input_ids"] = torch.cat((ids, suffix), dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    types = torch.zeros((1, suffix.shape[1]), dtype=inputs["mm_token_type_ids"].dtype, device=ids.device)
    inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], types), dim=1)
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -len(target_ids) - 1 : -1] = suffix[:, :-1]
    inputs["labels"] = labels
    return inputs


class IntervalDataset:
    def __init__(self, rows: list[dict[str, Any]], processor, dataset_root: Path, tokenizer, im_end_id: int):
        self.rows, self.processor, self.dataset_root = rows, processor, dataset_root
        self.tokenizer, self.im_end_id = tokenizer, im_end_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        inputs = _model_inputs(self.processor, row, self.dataset_root)
        target = format_interval(row["start"], row["end"], row["duration"])
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        return _append_tokens(inputs, target_ids, self.im_end_id)


def _collate(batch: list[dict[str, Any]]):
    if len(batch) != 1:
        raise ValueError("video pilot uses batch size 1")
    return batch[0]


def _load_components(adapter: Path, trainable: bool):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter not found: {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    return processor, model


def run_sft(args: argparse.Namespace) -> int:
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if args.arm not in {"s0", "s1"}:
        raise ValueError("SFT arms are s0 (shared initialization) or s1 (SFT continuation)")
    if args.max_steps <= 0 or args.learning_rate <= 0 or args.save_steps <= 0:
        raise ValueError("max_steps, learning_rate, and save_steps must be positive")
    _install_stop_handler()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires a CUDA GPU with bf16 support; run preflight first")
    if args.grad_accum != 1:
        raise ValueError("the paired pilot fixes gradient accumulation at 1 to match prompt exposures")
    output = _safe_output(Path(args.output_dir), resume=True)
    rows = _scheduled_rows(json.loads(Path(args.train_json).read_text()), args.max_steps, args.seed)
    processor, model = _load_components(Path(args.init_adapter), True)
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise RuntimeError("tokenizer has no <|im_end|> token")
    model.config.use_cache = False
    model.enable_input_require_grads()
    dataset = IntervalDataset(rows, processor, Path(args.dataset_root), tokenizer, im_end_id)
    training = TrainingArguments(
        output_dir=str(output), num_train_epochs=1, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate, warmup_steps=math.ceil(args.max_steps * 0.03),
        lr_scheduler_type="cosine",
        bf16=True, gradient_checkpointing=True, remove_unused_columns=False,
        logging_steps=1, save_strategy="steps", save_steps=max(1, args.save_steps),
        save_total_limit=2, report_to="none", dataloader_num_workers=0,
        seed=args.seed, data_seed=args.seed,
    )

    class OrderedTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(train_dataset or self.train_dataset)

    class StopAtCheckpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if _STOP_REQUESTED:
                control.should_save = True
                control.should_training_stop = True
            return control

    started = time.monotonic()
    checkpoints = sorted(
        output.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[1])
    )
    resume_from = str(checkpoints[-1]) if checkpoints else None
    trainer = OrderedTrainer(
        model=model, args=training, train_dataset=dataset, data_collator=_collate,
        callbacks=[StopAtCheckpoint],
    )
    trainer.train(
        resume_from_checkpoint=resume_from
    )
    if _STOP_REQUESTED:
        return 75
    model.save_pretrained(output)
    processor.save_pretrained(output)
    target_tokens = sum(
        len(tokenizer(format_interval(row["start"], row["end"], row["duration"]), add_special_tokens=False)["input_ids"])
        for row in rows
    )
    elapsed = time.monotonic() - started
    (output / "arm.json").write_text(json.dumps({
        "arm": args.arm,
        "init_adapter": str(Path(args.init_adapter).resolve()),
        "generated_tokens": 0,
        "optimizer_updates": args.max_steps,
        "prompt_schedule_seed": args.seed,
        "training_target_tokens": target_tokens,
        "elapsed_s": elapsed,
        "gpu_hours": elapsed / 3600,
        "gpu_cost_usd": None,
    }, indent=2) + "\n")
    return 0


def _move_inputs(inputs: dict[str, Any], device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _completion_logprob(
    model, inputs: dict[str, Any], completion_ids: Any, grad: bool, temperature: float = 1.0
):
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = inputs["input_ids"]
    completion_ids = completion_ids.to(ids.device).unsqueeze(0)
    joined = torch.cat((ids, completion_ids), dim=1)
    joined_inputs = dict(inputs)
    joined_inputs["input_ids"] = joined
    joined_inputs["attention_mask"] = torch.ones_like(joined)
    types = torch.zeros((1, completion_ids.shape[1]), dtype=inputs["mm_token_type_ids"].dtype, device=ids.device)
    joined_inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], types), dim=1)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(**joined_inputs, use_cache=False).logits[:, :-1]
        first = ids.shape[1] - 1
        scores = torch.log_softmax(logits[:, first:] / temperature, dim=-1)
        return scores.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)


def run_grpo(args: argparse.Namespace) -> int:
    import torch

    if args.arm != "r1":
        raise ValueError("GRPO arm must be r1 and must initialize from the shared S0 adapter")
    if (
        args.max_steps <= 0 or args.learning_rate <= 0 or args.save_steps <= 0
        or args.group_size < 2 or args.temperature <= 0 or args.max_new_tokens <= 0
        or args.kl_beta < 0 or not 0 < args.clip_range < 1
    ):
        raise ValueError("invalid GRPO steps, optimizer, group, sampling, KL, or clipping setting")
    _install_stop_handler()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GRPO requires a CUDA GPU with bf16 support; run preflight first")
    output = _safe_output(Path(args.output_dir), resume=True)
    rows = _scheduled_rows(json.loads(Path(args.train_json).read_text()), args.max_steps, args.seed)
    checkpoints = sorted(
        (path for path in output.glob("checkpoint-*")
         if (path / "training_state.pt").is_file() and (path / "adapter_model.safetensors").is_file()),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    start_step = int(checkpoints[-1].name.rsplit("-", 1)[1]) if checkpoints else 0
    adapter = checkpoints[-1] if checkpoints else Path(args.init_adapter)
    processor, model = _load_components(adapter, True)
    _, reference = _load_components(Path(args.reference_adapter or args.init_adapter), False)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    device = torch.device("cuda")
    model.to(device).eval()
    reference.to(device).eval()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    generated_tokens = 0
    if checkpoints:
        state = torch.load(checkpoints[-1] / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        generated_tokens = int(state["generated_tokens"])
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    started = time.monotonic()
    for update, row in enumerate(rows[start_step:], start=start_step):
        base = _move_inputs(_model_inputs(processor, row, Path(args.dataset_root)), device)
        completions: list[Any] = []
        rewards: list[float] = []
        old_logps: list[Any] = []
        for _ in range(args.group_size):
            with torch.no_grad():
                output_ids = model.generate(**base, do_sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            completion = output_ids[0, base["input_ids"].shape[1] :]
            text = processor.tokenizer.decode(completion, skip_special_tokens=True)
            completions.append(completion)
            generated_tokens += int(completion.numel())
            rewards.append(interval_reward(text, (row["start"], row["end"]), row["duration"]))
            old_logps.append(
                _completion_logprob(model, base, completion, grad=False, temperature=args.temperature).detach()
            )
        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        std = reward_tensor.std(unbiased=False)
        advantages = (reward_tensor - reward_tensor.mean()) / (std + 1e-6) if float(std) else torch.zeros_like(reward_tensor)
        optimizer.zero_grad(set_to_none=True)
        for completion, old_logp, advantage in zip(completions, old_logps, advantages):
            new_logp = _completion_logprob(
                model, base, completion, grad=True, temperature=args.temperature
            )
            ref_logp = _completion_logprob(
                reference, base, completion, grad=False, temperature=args.temperature
            )
            ratio = torch.exp((new_logp - old_logp).clamp(-20, 20))
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            policy = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            (policy + args.kl_beta * (new_logp - ref_logp).mean()).div(len(completions)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (update + 1) % args.log_steps == 0 or update == 0:
            print(json.dumps({"update": update + 1, "reward_mean": sum(rewards) / len(rewards), "reward_std": float(std), "generated_tokens": generated_tokens, "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        if (update + 1) % args.save_steps == 0 or _STOP_REQUESTED:
            checkpoint = output / f"checkpoint-{update + 1}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "generated_tokens": generated_tokens,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            }, checkpoint / "training_state.pt")
        if _STOP_REQUESTED:
            return 75
    model.save_pretrained(output)
    processor.save_pretrained(output)
    elapsed = time.monotonic() - started
    (output / "arm.json").write_text(json.dumps({
        "arm": args.arm,
        "updates": args.max_steps,
        "group_size": args.group_size,
        "prompt_schedule_seed": args.seed,
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed,
        "gpu_hours": elapsed / 3600,
        "gpu_cost_usd": None,
    }, indent=2) + "\n")
    return 0


def smoke(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("model/video smoke requires a CUDA GPU with bf16 support")
    rows = json.loads(Path(args.rows).read_text())
    if not 0 <= args.row_index < len(rows) or args.generate_tokens <= 0:
        raise ValueError("row-index must select a row and generate-tokens must be positive")
    row = rows[args.row_index]
    processor, model = _load_components(Path(args.init_adapter), True)
    model.to(torch.device("cuda")).train()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise RuntimeError("tokenizer has no <|im_end|> token")
    target = format_interval(row["start"], row["end"], row["duration"])
    encoded = _append_tokens(
        _model_inputs(processor, row, Path(args.dataset_root)),
        tokenizer(target, add_special_tokens=False)["input_ids"],
        im_end_id,
    )
    encoded = _move_inputs(encoded, torch.device("cuda"))
    loss = model(**encoded).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("training forward pass returned a missing or non-finite loss")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("training backward pass produced no trainable adapter gradients")
    model.eval()
    prompt = _move_inputs(_model_inputs(processor, row, Path(args.dataset_root)), torch.device("cuda"))
    with torch.inference_mode():
        generated = model.generate(**prompt, do_sample=False, max_new_tokens=args.generate_tokens)
    completion = generated[0, prompt["input_ids"].shape[1] :]
    print(json.dumps({
        "video_id": row["video_id"], "video_frames": 16,
        "input_tokens": int(prompt["input_ids"].shape[1]),
        "loss": float(loss.detach()),
        "generated_text": tokenizer.decode(completion, skip_special_tokens=True),
        "cuda_device": torch.cuda.get_device_name(),
        "bf16": bool(torch.cuda.is_bf16_supported()),
    }, indent=2))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Charades dev evaluation requires a CUDA GPU")
    rows = json.loads(Path(args.rows).read_text())
    if not rows:
        raise ValueError("evaluation rows are empty")
    output = _safe_output(Path(args.output_dir))
    processor, model = _load_components(Path(args.adapter), False)
    model.to(torch.device("cuda")).eval()
    tokenizer = processor.tokenizer
    scores: list[dict[str, Any]] = []
    ious: list[float] = []
    valid = 0
    started = time.monotonic()
    for index, row in enumerate(rows):
        inputs = _move_inputs(_model_inputs(processor, row, Path(args.dataset_root)), torch.device("cuda"))
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens
            )
        completion = output_ids[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(completion, skip_special_tokens=True).strip()
        prediction = parse_interval(text, row["duration"])
        iou = interval_iou(prediction, (row["start"], row["end"]))
        valid += prediction is not None
        ious.append(iou)
        scores.append({
            "qa_id": row["qa_id"], "video_id": row["video_id"], "query": row["query"],
            "prediction_text": text, "prediction": prediction,
            "gold": [row["start"], row["end"]], "temporal_iou": iou,
        })
        if (index + 1) % args.log_steps == 0:
            print(json.dumps({"evaluated": index + 1, "total": len(rows)}), flush=True)
    summary = {
        "dataset_repo": DATASET_REPO, "dataset_revision": DATASET_REVISION,
        "adapter": str(Path(args.adapter).resolve()), "rows": len(rows),
        "valid_format_rate": valid / len(rows),
        "mean_temporal_iou": sum(ious) / len(ious),
        "recall_at_iou": {str(threshold): sum(iou >= threshold for iou in ious) / len(ious)
                           for threshold in (0.3, 0.5, 0.7)},
        "elapsed_s": time.monotonic() - started,
    }
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(score, ensure_ascii=False) + "\n" for score in scores)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight",):
        command = sub.add_parser(name)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--train-annotations", default="charades_sta_train.txt")
        command.add_argument("--test-annotations", default="charades_sta_test.txt")
        command.add_argument("--durations-json", default="durations.json")
    command = sub.add_parser("smoke")
    command.add_argument("--dataset-root", required=True)
    command.add_argument("--rows", required=True)
    command.add_argument("--init-adapter", required=True)
    command.add_argument("--row-index", type=int, default=0)
    command.add_argument("--generate-tokens", type=int, default=8)
    command = sub.add_parser("evaluate")
    command.add_argument("--dataset-root", required=True)
    command.add_argument("--rows", required=True)
    command.add_argument("--adapter", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--max-new-tokens", type=int, default=32)
    command.add_argument("--log-steps", type=int, default=20)
    command = sub.add_parser("prepare")
    command.add_argument("--dataset-root", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--train-annotations", default="charades_sta_train.txt")
    command.add_argument("--test-annotations", default="charades_sta_test.txt")
    command.add_argument("--durations-json", default="durations.json")
    command.add_argument("--sample-rows", type=int, default=1024)
    command.add_argument("--dev-fraction", type=float, default=0.1)
    command.add_argument("--seed", type=int, default=0)
    for name in ("sft", "grpo"):
        command = sub.add_parser(name)
        command.add_argument("--arm", choices=("s0", "s1", "r1"), required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--train-json", required=True)
        command.add_argument("--init-adapter", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--max-steps", type=int, default=100)
        command.add_argument("--learning-rate", type=float, default=1e-5 if name == "sft" else 1e-6)
        command.add_argument("--grad-accum", type=int, default=1)
        command.add_argument("--save-steps", type=int, default=25)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--log-steps", type=int, default=1)
        if name == "grpo":
            command.add_argument("--reference-adapter")
            command.add_argument("--group-size", type=int, default=4)
            command.add_argument("--temperature", type=float, default=0.7)
            command.add_argument("--max-new-tokens", type=int, default=32)
            command.add_argument("--kl-beta", type=float, default=0.02)
            command.add_argument("--clip-range", type=float, default=0.2)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return preflight(args)
    if args.command == "prepare":
        return prepare(args)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "sft":
        return run_sft(args)
    if args.command == "grpo":
        return run_grpo(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
