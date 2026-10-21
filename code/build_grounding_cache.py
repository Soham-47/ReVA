"""Build offline Grounding DINO detections and deterministic 2D tracks.

The builder only loads a caller-supplied local snapshot. It never resolves a
model name or downloads weights, and it writes detector caches separately from
the adaptive frame-feature caches.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import grounding as G


SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}

# Immutable metadata from the official HF tree API for the pinned revision.
# Regular files use their Git blob SHA-1; the safetensors LFS file uses its
# published SHA-256 content digest. The revision marker is provenance only.
# Source: https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71
SNAPSHOT_FILE_INTEGRITY = {
    "config.json": {"algorithm": "git-blob-sha1", "digest": "521abd71ae4ca275a31ea98bc68bd3e41450188c", "size": 1644},
    "model.safetensors": {"algorithm": "sha256", "digest": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3", "size": 689359096},
    "preprocessor_config.json": {"algorithm": "git-blob-sha1", "digest": "5cb45d963917ed130ce46a93204b349cbec21131", "size": 457},
    "added_tokens.json": {"algorithm": "git-blob-sha1", "digest": "f84095a3e2962f44bdd2f865e4333c35ae95d73f", "size": 82},
    "special_tokens_map.json": {"algorithm": "git-blob-sha1", "digest": "a8b3208c2884c4efb86e49300fdd3dc877220cdf", "size": 125},
    "tokenizer.json": {"algorithm": "git-blob-sha1", "digest": "688882a79f44442ddc1f60d70334a7ff5df0fb47", "size": 711396},
    "tokenizer_config.json": {"algorithm": "git-blob-sha1", "digest": "ed97a84add5f9b2091e756765ad3ba087a345e17", "size": 1237},
    "vocab.txt": {"algorithm": "git-blob-sha1", "digest": "fb140275c155a9c7c5a3b3e0e77a9e839594a938", "size": 231508},
}


def detector_revision() -> str:
    return G.GROUNDING_DETECTOR_REVISION


def _file_digest(path: Path, *, algorithm: str, size: int) -> str:
    """Hash one staged snapshot file without materializing it in memory."""

    if algorithm == "git-blob-sha1":
        hasher = hashlib.sha1()
        hasher.update(f"blob {size}\0".encode("ascii"))
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:  # pragma: no cover - the source-level manifest is fixed
        raise ValueError(f"unsupported snapshot integrity algorithm: {algorithm}")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_local_model_dir(model_dir: str | Path) -> Path:
    """Validate the staged detector snapshot before importing Transformers."""

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"local Grounding DINO model directory does not exist: {root}; "
            "stage the pinned snapshot first"
        )
    marker = root / G.GROUNDING_CACHE_MARKER
    if not marker.is_file():
        raise RuntimeError(
            f"local Grounding DINO model directory {root} is missing {G.GROUNDING_CACHE_MARKER}; "
            f"write the exact pinned SHA {G.GROUNDING_DETECTOR_MODEL_REVISION} after staging"
        )
    if marker.read_text().strip() != G.GROUNDING_DETECTOR_MODEL_REVISION:
        raise RuntimeError(
            f"{marker} must contain the exact pinned model revision "
            f"{G.GROUNDING_DETECTOR_MODEL_REVISION}"
        )
    missing = [name for name in G.GROUNDING_DETECTOR_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"local Grounding DINO model directory {root} is missing: {', '.join(missing)}; "
            "no network download is attempted"
        )
    mismatched = []
    for name, expected in SNAPSHOT_FILE_INTEGRITY.items():
        path = root / name
        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            mismatched.append(f"{name} size {actual_size} (expected {expected['size']})")
            continue
        actual_digest = _file_digest(
            path, algorithm=expected["algorithm"], size=expected["size"]
        )
        if actual_digest != expected["digest"]:
            mismatched.append(
                f"{name} digest {actual_digest} (expected {expected['digest']})"
            )
    if mismatched:
        raise RuntimeError(
            f"local Grounding DINO files do not match immutable HF snapshot "
            f"{G.GROUNDING_DETECTOR_MODEL_REVISION}: {'; '.join(mismatched)}"
        )
    return root


def _read_id_file(path: str | Path, *, label: str, split: str) -> set[str]:
    source = Path(path)
    if source.suffix.casefold() == ".json":
        try:
            payload = json.loads(source.read_text())
            section = payload["splits"][split]
            values = section["ids"] if isinstance(section, Mapping) else section
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{label} JSON must contain splits.{split}.ids"
            ) from exc
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{label} JSON split {split} must contain a string ids list")
    else:
        values = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} file contains duplicate IDs")
    if not values:
        raise ValueError(f"{label} file is empty")
    return set(values)


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    qa_ids_file: str | None,
    video_paths_file: str | None,
    split: str,
) -> list[dict[str, Any]]:
    qa_ids = _read_id_file(qa_ids_file, label="QA IDs", split=split) if qa_ids_file else None
    video_paths = _read_id_file(video_paths_file, label="video paths", split=split) if video_paths_file else None
    selected = [
        dict(row)
        for row in rows
        if (qa_ids is None or row.get("qa_id") in qa_ids)
        and (video_paths is None or row.get("video_path") in video_paths)
    ]
    if not selected:
        raise ValueError("filters selected no questions")
    if qa_ids is not None:
        found = {row.get("qa_id") for row in selected}
        missing = sorted(qa_ids - found)
        if missing:
            raise ValueError(f"QA IDs are absent from the selected split: {', '.join(missing[:5])}")
    if video_paths is not None:
        found = {row.get("video_path") for row in selected}
        missing = sorted(video_paths - found)
        if missing:
            raise ValueError(f"video paths are absent from the selected split: {', '.join(missing[:5])}")
    return selected


def _offsets(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(not 0 <= item < 1 for item in parsed):
        raise ValueError("offsets must be comma-separated values in [0, 1)")
    return parsed


def frame_policy(info: Mapping[str, Any], offsets: Sequence[float], sampling_mode: str) -> dict[str, Any]:
    indices: list[int] = []
    for offset in offsets:
        indices.extend(
            C.sample_index_list(int(info["n_frames"]), float(info["fps"]), offset, mode=sampling_mode)
        )
    return G.detector_frame_policy(indices, offsets, sampling_mode)


def _load_local_detector(model_dir: Path, device: str):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "building Grounding DINO caches requires the installed torch and Transformers "
            "dependencies; no dependency or model download is attempted"
        ) from exc
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_dir), revision=G.GROUNDING_DETECTOR_MODEL_REVISION, local_files_only=True
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(model_dir), revision=G.GROUNDING_DETECTOR_MODEL_REVISION, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load staged Grounding DINO files from {model_dir} with local_files_only=True"
        ) from exc
    model.to(device).eval()
    return torch, processor, model


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def detect_frames(
    processor: Any,
    model: Any,
    frames: Iterable[Any],
    frame_indices: Sequence[int],
    *,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    torch: Any,
    device: str,
) -> list[dict[str, Any]]:
    """Run the documented Grounding DINO processor/model flow per image."""

    from PIL import Image

    labels = [part.strip() for part in prompt.split(".") if part.strip()]
    output: list[dict[str, Any]] = []
    with torch.inference_mode():
        for frame_index, frame in zip(frame_indices, frames):
            height, width = int(frame.shape[0]), int(frame.shape[1])
            image = Image.fromarray(frame)
            inputs = processor(images=image, text=[labels], return_tensors="pt")
            inputs = _move_inputs(inputs, device)
            results = model(**inputs)
            processed = processor.post_process_grounded_object_detection(
                results,
                inputs["input_ids"] if isinstance(inputs, Mapping) else inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(height, width)],
            )[0]
            boxes = _as_list(processed.get("boxes", []))
            scores = _as_list(processed.get("scores", []))
            # Recent Transformers releases expose human-readable labels under
            # ``text_labels`` while ``labels`` may be integer token indices.
            result_labels = processed.get("text_labels")
            if result_labels is None:
                result_labels = processed.get("labels", [])
            result_labels = _as_list(result_labels) if not isinstance(result_labels, list) else result_labels
            detections = []
            for order, (box, score, label) in enumerate(zip(boxes, scores, result_labels)):
                detections.append(
                    {
                        "detection_index": order,
                        "label": str(label),
                        "confidence": float(score),
                        "box": [float(value) for value in box],
                    }
                )
            output.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp": float(frame_index),
                    "width": width,
                    "height": height,
                    "detections": detections,
                }
            )
    return output


@contextmanager
def cache_writer_lock(cache_dir: str | Path):
    """Allow one grounding-cache builder process to write a directory at a time."""

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".build_grounding_cache.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"grounding-cache directory is busy: {root}") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield root
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def build_caches(args: argparse.Namespace) -> list[Path]:
    cache_dir = C.work_path(args.grounding_cache_dir)
    prompt = G.normalize_detector_prompt(args.prompt)
    offsets = _offsets(args.offsets)
    thresholds = {
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
    }
    tracker = {
        "algorithm": "adjacent-greedy-iou",
        "iou_threshold": float(args.iou_threshold),
    }
    with cache_writer_lock(cache_dir):
        split_path = C.work_path(SPLITS[args.split])
        with split_path.open() as stream:
            rows = json.load(stream)["QA"]
        rows = _selected_rows(
            rows,
            qa_ids_file=args.qa_ids_file,
            video_paths_file=args.video_paths_file,
            split=args.split,
        )
        manifest = C.load_manifest()
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_video[row["video_path"]].append(row)
        pending: list[tuple[str, str, dict[str, Any], dict[str, Any], Path]] = []
        output_paths: list[Path] = []
        for relative_path in by_video:
            try:
                info = manifest[relative_path]
            except KeyError as exc:
                raise KeyError(f"{relative_path!r} is absent from {C.manifest_path()}") from exc
            checksum = C.path_checksum(C.dataset_dir() / relative_path)
            policy = frame_policy(info, offsets, args.frame_sampling)
            path = G.detector_cache_path(
                cache_dir,
                checksum,
                detector_revision(),
                prompt,
                G.GROUNDING_DETECTOR_PREPROCESSING,
                thresholds,
                policy,
                tracker,
            )
            try:
                cache = G.load_detector_cache(path, expected_cache_key=path.stem)
            except (FileNotFoundError, RuntimeError, ValueError):
                cache = None
            if cache is not None:
                output_paths.append(path)
                continue
            pending.append((relative_path, checksum, info, policy, path))
        if not pending:
            return output_paths
        model_dir = validate_local_model_dir(args.model_dir)
        torch, processor, model = _load_local_detector(model_dir, args.device)
        for relative_path, checksum, info, policy, path in pending:
            frame_indices = policy["frame_indices"]
            frames = []
            for batch in C.iter_read_frames(relative_path, frame_indices, args.batch_size):
                frames.extend(batch)
            frames_out = detect_frames(
                processor,
                model,
                frames,
                frame_indices,
                prompt=prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                torch=torch,
                device=args.device,
            )
            fps = max(float(info["fps"]), 1e-6)
            for frame in frames_out:
                frame["timestamp"] = frame["frame_index"] / fps
            tracks = G.associate_detections(frames_out, iou_threshold=args.iou_threshold)
            written = G.write_detector_cache(
                cache_dir,
                video_checksum=checksum,
                detector_revision=detector_revision(),
                prompt=prompt,
                preprocessing=G.GROUNDING_DETECTOR_PREPROCESSING,
                thresholds=thresholds,
                frame_policy=policy,
                tracker=tracker,
                frames=frames_out,
                tracks=tracks,
            )
            output_paths.append(written)
            print(f"wrote {written} for {relative_path}")
        return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--model-dir", required=True, help="local pinned Grounding DINO snapshot directory")
    parser.add_argument("--grounding-cache-dir", required=True)
    parser.add_argument("--prompt", default=G.DEFAULT_GROUNDING_PROMPT)
    parser.add_argument("--offsets", default="0,0.5")
    parser.add_argument("--frame-sampling", choices=C.FRAME_SAMPLING_MODES, default=C.DEFAULT_FRAME_SAMPLING)
    parser.add_argument("--box-threshold", type=float, default=G.DEFAULT_DETECTOR_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=G.DEFAULT_DETECTOR_TEXT_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=G.DEFAULT_TRACK_IOU_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qa-ids-file", help="newline-delimited IDs or split-aware pilot JSON manifest")
    parser.add_argument("--video-paths-file")
    args = parser.parse_args()
    if not 0 <= args.box_threshold <= 1 or not 0 <= args.text_threshold <= 1 or not 0 <= args.iou_threshold <= 1:
        parser.error("box/text/IoU thresholds must be in [0, 1]")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    try:
        build_caches(args)
    except (KeyError, RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
