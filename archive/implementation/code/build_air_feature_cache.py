"""Build exact endpoint-inclusive AIR candidate caches offline.

Unlike the legacy evidence cache (which is rate-limited and may contain only
24--32 frames on ReVA's short clips), this builder always materializes the
declared AIR candidate count, normally 64. It uses the frozen local SigLIP2
snapshot and never downloads weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import grounding as G
from air_sampler import candidate_grid, candidate_policy


SPLITS = {"valQ": "data/splits/valQ.json", "valT": "data/splits/valT.json", "val": "data/ReVA/val.json"}


def _move(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _feature(output: Any, torch_module: Any) -> list[list[float]]:
    value = getattr(output, "pooler_output", None)
    if value is None:
        value = getattr(output, "image_embeds", None)
    if value is None and isinstance(output, (tuple, list)):
        value = output[0]
    if value is None and hasattr(output, "shape"):
        value = output
    if value is None:
        raise RuntimeError("SigLIP2 output has no image/text feature tensor")
    return torch_module.nn.functional.normalize(value.float(), dim=-1).detach().cpu().tolist()


def _encode_images(processor: Any, model: Any, frames: Sequence[Any], torch_module: Any, device: str, batch_size: int) -> list[list[float]]:
    from PIL import Image

    vectors: list[list[float]] = []
    with torch_module.inference_mode():
        for start in range(0, len(frames), batch_size):
            images = [Image.fromarray(frame) for frame in frames[start : start + batch_size]]
            inputs = processor(images=images, return_tensors="pt")
            output = model.get_image_features(**_move(inputs, device))
            vectors.extend(_feature(output, torch_module))
    return vectors


def _query_text(row: Mapping[str, Any]) -> str:
    options = row.get("options", {})
    choices = "\n".join(f"{letter}. {options[letter]}" for letter in C.LETTERS if letter in options)
    return f"Question: {row.get('question', '')}\nOptions:\n{choices}" if choices else f"Question: {row.get('question', '')}"


def _encode_queries(processor: Any, model: Any, rows: Sequence[Mapping[str, Any]], torch_module: Any, device: str, batch_size: int) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    with torch_module.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = processor(text=[_query_text(row) for row in batch], padding="max_length", truncation=True, return_tensors="pt")
            output = model.get_text_features(**_move(inputs, device))
            for row, vector in zip(batch, _feature(output, torch_module)):
                vectors[str(row["qa_id"])] = vector
    return vectors


def build(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModel, AutoProcessor

    if args.count < 2 or args.batch_size < 1:
        raise ValueError("count must be >= 2 and batch-size positive")
    rows = json.loads(C.work_path(SPLITS[args.split]).read_text())["QA"]
    manifest = C.load_manifest()
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_path"]].append(row)
    policy = candidate_policy(args.count)
    model_revision = G.FEATURE_ENCODER_REVISION
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    cache_dir = C.work_path(args.feature_cache_dir)
    for relative_path, video_rows in by_video.items():
        info = manifest[relative_path]
        checksum = C.path_checksum(C.dataset_dir() / relative_path)
        grid = candidate_grid(int(info["n_frames"]), args.count, float(info["fps"]))
        indices = [int(item["frame_index"]) for item in grid]
        frames = C.read_frames(relative_path, indices)
        image_features = _encode_images(processor, model, frames, torch, args.device, args.batch_size)
        query_features = _encode_queries(processor, model, video_rows, torch, args.device, args.batch_size)
        path = G.write_feature_cache(
            cache_dir,
            video_checksum=checksum,
            encoder_revision=model_revision,
            candidate_policy=policy,
            preprocessing=G.FEATURE_PREPROCESSING,
            candidates=[{**item, "feature": feature} for item, feature in zip(grid, image_features)],
            query_features=query_features,
            query_fingerprints={row["qa_id"]: G.query_input_fingerprint(row) for row in video_rows},
        )
        print(f"wrote {path} candidates={len(indices)} video={relative_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
