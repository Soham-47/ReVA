"""Build offline SAM 2 masks from the fixed Grounding DINO frame boxes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import grounding as G
import segmentation as S


SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def validate_local_model_dir(model_dir: str | Path) -> Path:
    """Require the exact SAM 2 snapshot before importing model classes."""

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"local SAM 2 model directory does not exist: {root}; stage the pinned snapshot first"
        )
    marker = root / S.SEGMENTATION_CACHE_MARKER
    if not marker.is_file():
        raise RuntimeError(
            f"local SAM 2 model directory {root} is missing {S.SEGMENTATION_CACHE_MARKER}; "
            f"write the exact pinned revision {S.SEGMENTATION_MODEL_REVISION} after staging"
        )
    if marker.read_text().strip() != S.SEGMENTATION_MODEL_REVISION:
        raise RuntimeError(
            f"{marker} must contain the exact pinned model revision {S.SEGMENTATION_MODEL_REVISION}"
        )
    missing = [name for name in S.SEGMENTATION_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"local SAM 2 model directory {root} is missing: {', '.join(missing)}")
    return root


def _load_local_sam(model_dir: Path, device: str):
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "building SAM 2 caches requires the installed torch and Transformers dependencies; "
            "no dependency or model download is attempted"
        ) from exc
    try:
        processor = Sam2Processor.from_pretrained(str(model_dir), local_files_only=True)
        model = Sam2Model.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"could not load staged SAM 2 files from {model_dir} with local_files_only=True"
        ) from exc
    model.to(device).eval()
    return torch, processor, model


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _mask_array(value: Any, *, box_count: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == box_count:
        array = array[0]
    if array.ndim == 4 and array.shape[1] == 1 and array.shape[0] == box_count:
        array = array[:, 0]
    if array.ndim == 3 and array.shape[0] == 1 and box_count == 1:
        array = array[0:1]
    if array.ndim != 3 or array.shape[0] != box_count:
        raise RuntimeError(f"SAM 2 returned masks with shape {array.shape} for {box_count} boxes")
    return array.astype(bool)


def _mask_stats(mask: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if not len(xs):
        return 0.0, {"x": 0.5, "y": 0.5}, {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    area = float(len(xs)) / float(width * height)
    centroid = {"x": float(xs.mean() / width), "y": float(ys.mean() / height)}
    box = {
        "left": float(xs.min() / width),
        "top": float(ys.min() / height),
        "right": float((xs.max() + 1) / width),
        "bottom": float((ys.max() + 1) / height),
    }
    return area, centroid, box


def segment_frame(
    processor: Any,
    model: Any,
    frame: np.ndarray,
    detections: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    device: str,
) -> list[dict[str, Any]]:
    """Segment detector boxes in one frame and return portable mask metadata."""

    if not detections:
        return []
    from PIL import Image

    height, width = int(frame.shape[0]), int(frame.shape[1])
    boxes = []
    for detection in detections:
        box = G.normalise_box(detection["box"])
        boxes.append(
            [
                box["left"] * width,
                box["top"] * height,
                box["right"] * width,
                box["bottom"] * height,
            ]
        )
    image = Image.fromarray(frame)
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
    inputs = _move_inputs(inputs, device)
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    try:
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=True
        )[0]
    except TypeError:
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    masks = _mask_array(masks, box_count=len(detections))
    result: list[dict[str, Any]] = []
    for detection, mask in zip(detections, masks):
        area, centroid, mask_box = _mask_stats(mask)
        result.append(
            {
                "detection_index": int(detection.get("detection_index", 0)),
                "label": str(detection.get("label", "unknown")),
                "confidence": float(detection.get("confidence", 0.0)),
                "area": area,
                "centroid": centroid,
                "box": mask_box,
                "mask": mask,
                "detector_box": G.normalise_box(detection["box"]),
            }
        )
    return result


def _write_contact_sheet(
    path: Path,
    entries: Sequence[tuple[np.ndarray, Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> None:
    from PIL import Image, ImageDraw

    if not entries:
        return
    thumbs = []
    tile_width = 320
    for frame, row, masks in entries[:12]:
        image = Image.fromarray(frame).convert("RGBA")
        scale = tile_width / image.width
        tile_height = max(1, round(image.height * scale))
        image = image.resize((tile_width, tile_height))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for mask in masks:
            box = mask.get("box", {})
            left = round(float(box.get("left", 0)) * tile_width)
            top = round(float(box.get("top", 0)) * tile_height)
            right = round(float(box.get("right", 0)) * tile_width)
            bottom = round(float(box.get("bottom", 0)) * tile_height)
            draw.rectangle((left, top, right, bottom), outline=(255, 70, 30, 255), width=2)
            draw.text((left + 2, top + 2), str(mask.get("label", "?"))[:18], fill=(255, 255, 0, 255))
        tile = Image.alpha_composite(image, overlay).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, tile_width, 20), fill=(0, 0, 0))
        draw.text((4, 4), f"frame={row['frame_index']} time={float(row['timestamp']):.2f}s", fill="white")
        thumbs.append(tile)
    tile_height = max(tile.height for tile in thumbs)
    columns = 4
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "black")
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


@contextmanager
def cache_writer_lock(cache_dir: str | Path):
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".build_segmentation_cache.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"segmentation-cache directory is busy: {root}") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield root
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _offsets(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(not 0 <= item < 1 for item in parsed):
        raise ValueError("offsets must be comma-separated values in [0, 1)")
    return parsed


def _selected_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    return __import__("build_grounding_cache")._selected_rows(
        rows,
        qa_ids_file=args.qa_ids_file,
        video_paths_file=args.video_paths_file,
        split=args.split,
    )


def build_caches(args: argparse.Namespace) -> list[Path]:
    cache_dir = C.work_path(args.segmentation_cache_dir)
    grounding_dir = C.work_path(args.grounding_cache_dir)
    offsets = _offsets(args.offsets)
    mask_policy = S.segmentation_mask_policy(args.max_masks_per_frame)
    with cache_writer_lock(cache_dir):
        split_path = C.work_path(SPLITS[args.split])
        rows = _selected_rows(json.loads(split_path.read_text())["QA"], args)
        manifest = C.load_manifest()
        by_video: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_video.setdefault(row["video_path"], []).append(row)
        pending: list[tuple[str, str, dict[str, Any], dict[str, Any], Path, dict[str, Any]]] = []
        output_paths: list[Path] = []
        prompt = G.normalize_detector_prompt(args.grounding_prompt)
        thresholds = {
            "box_threshold": float(args.grounding_box_threshold),
            "text_threshold": float(args.grounding_text_threshold),
        }
        tracker = {"algorithm": "adjacent-greedy-iou", "iou_threshold": float(args.grounding_iou_threshold)}
        for relative_path in by_video:
            info = manifest[relative_path]
            checksum = C.path_checksum(C.dataset_dir() / relative_path)
            frame_indices: list[int] = []
            for offset in offsets:
                frame_indices.extend(
                    C.sample_index_list(
                        int(info["n_frames"]), float(info["fps"]), offset, mode=C.DEFAULT_FRAME_SAMPLING
                    )
                )
            frame_policy = G.detector_frame_policy(frame_indices, offsets, C.DEFAULT_FRAME_SAMPLING)
            detector_key = G.detector_cache_key(
                checksum,
                G.GROUNDING_DETECTOR_REVISION,
                prompt,
                G.GROUNDING_DETECTOR_PREPROCESSING,
                thresholds,
                frame_policy,
                tracker,
            )
            detector_path = G.detector_cache_path(
                grounding_dir,
                checksum,
                G.GROUNDING_DETECTOR_REVISION,
                prompt,
                G.GROUNDING_DETECTOR_PREPROCESSING,
                thresholds,
                frame_policy,
                tracker,
            )
            detector_cache = G.load_detector_cache(detector_path, expected_cache_key=detector_key)
            segmentation_key = S.segmentation_cache_key(
                checksum, S.SEGMENTATION_MODEL_REVISION_ID, detector_key, frame_policy, mask_policy
            )
            output_path = S.segmentation_cache_path(
                cache_dir,
                checksum,
                S.SEGMENTATION_MODEL_REVISION_ID,
                detector_key,
                frame_policy,
                mask_policy,
            )
            try:
                loaded = S.load_segmentation_cache(output_path, expected_cache_key=segmentation_key)
            except (FileNotFoundError, RuntimeError, ValueError):
                loaded = None
            if loaded is not None:
                output_paths.append(output_path)
                continue
            pending.append((relative_path, checksum, info, frame_policy, output_path, detector_cache))
        if not pending:
            return output_paths
        model_dir = validate_local_model_dir(args.model_dir)
        torch, processor, model = _load_local_sam(model_dir, args.device)
        for relative_path, checksum, info, frame_policy, output_path, detector_cache in pending:
            frame_by_index = {frame["frame_index"]: frame for frame in detector_cache["frames"]}
            frame_indices = frame_policy["frame_indices"]
            cache_frames: list[dict[str, Any]] = []
            contact_entries: list[tuple[np.ndarray, Mapping[str, Any], Sequence[Mapping[str, Any]]]] = []
            frame_iter = C.iter_read_frames(relative_path, frame_indices, args.batch_size)
            for frame_index, frame in zip(frame_indices, (item for batch in frame_iter for item in batch)):
                detector_frame = frame_by_index[frame_index]
                detections = sorted(
                    detector_frame["detections"],
                    key=lambda row: (-float(row.get("confidence", 0.0)), int(row.get("detection_index", 0))),
                )[: args.max_masks_per_frame]
                masks = segment_frame(
                    processor,
                    model,
                    frame,
                    detections,
                    torch=torch,
                    device=args.device,
                )
                output_masks: list[dict[str, Any]] = []
                for mask in masks:
                    mask_id = f"frame-{frame_index:06d}-det-{int(mask['detection_index']):04d}"
                    relative_mask = Path("masks") / output_path.stem / f"{mask_id}.png"
                    mask_path = cache_dir / relative_mask
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    from PIL import Image

                    Image.fromarray((mask.pop("mask").astype(np.uint8) * 255), mode="L").save(mask_path)
                    mask.pop("detector_box", None)
                    output_masks.append(
                        {
                            **mask,
                            "mask_id": mask_id,
                            "frame_index": frame_index,
                            "timestamp": frame_index / max(float(info["fps"]), 1e-6),
                            "mask_path": str(relative_mask),
                        }
                    )
                frame_row = {
                    "frame_index": frame_index,
                    "timestamp": frame_index / max(float(info["fps"]), 1e-6),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "masks": output_masks,
                }
                cache_frames.append(frame_row)
                if len(contact_entries) < 12:
                    contact_entries.append((frame, frame_row, output_masks))
            contact_relative = Path("contacts") / f"{output_path.stem}.jpg"
            _write_contact_sheet(cache_dir / contact_relative, contact_entries)
            written = S.write_segmentation_cache(
                cache_dir,
                video_checksum=checksum,
                segmentation_revision=S.SEGMENTATION_MODEL_REVISION_ID,
                grounding_cache_key=detector_cache["cache_key"],
                frame_policy=frame_policy,
                mask_policy=mask_policy,
                frames=cache_frames,
                contact_sheet=str(contact_relative),
            )
            output_paths.append(written)
            print(f"wrote {written} for {relative_path}")
        return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--model-dir", required=True, help="local pinned SAM 2 snapshot directory")
    parser.add_argument("--grounding-cache-dir", required=True)
    parser.add_argument("--segmentation-cache-dir", required=True)
    parser.add_argument("--grounding-prompt", default=G.DEFAULT_GROUNDING_PROMPT)
    parser.add_argument("--grounding-box-threshold", type=float, default=G.DEFAULT_DETECTOR_THRESHOLD)
    parser.add_argument("--grounding-text-threshold", type=float, default=G.DEFAULT_DETECTOR_TEXT_THRESHOLD)
    parser.add_argument("--grounding-iou-threshold", type=float, default=G.DEFAULT_TRACK_IOU_THRESHOLD)
    parser.add_argument("--offsets", default="0,0.5")
    parser.add_argument("--frame-sampling", choices=C.FRAME_SAMPLING_MODES, default=C.DEFAULT_FRAME_SAMPLING)
    parser.add_argument("--max-masks-per-frame", type=int, default=S.DEFAULT_MAX_MASKS_PER_FRAME)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--qa-ids-file", help="newline-delimited IDs or split-aware pilot JSON manifest")
    parser.add_argument("--video-paths-file")
    args = parser.parse_args()
    if args.frame_sampling != C.DEFAULT_FRAME_SAMPLING:
        parser.error("segmentation requires baseline legacy frame sampling")
    if not 0 <= args.grounding_box_threshold <= 1 or not 0 <= args.grounding_text_threshold <= 1 or not 0 <= args.grounding_iou_threshold <= 1:
        parser.error("grounding thresholds must be in [0, 1]")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    try:
        build_caches(args)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
