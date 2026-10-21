"""Build query-conditioned SigLIP2 feature caches from staged local weights.

The builder is deliberately offline-only. It never resolves a model name or
downloads files; callers must stage one immutable Hugging Face snapshot first.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
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

SIGLIP2_MODEL_REVISION = "022b6f71160ffb0169ca4709e2d7e25be659598a"
REVISION_MARKER = ".reva_encoder_revision"

# Immutable metadata from the HF tree API for the pinned revision.  Regular
# Git files use their Git blob SHA-1; Xet/LFS files use the published SHA-256
# content digest.  The marker below is only provenance and cannot substitute
# for these file checks.  Source: https://huggingface.co/api/models/google/
# siglip2-base-patch16-224/tree/022b6f71160ffb0169ca4709e2d7e25be659598a
SNAPSHOT_FILE_INTEGRITY = {
    "config.json": {"algorithm": "git-blob-sha1", "digest": "c8cd2a20e58a738f44f267ae19f9568ff1095698", "size": 253},
    "model.safetensors": {"algorithm": "sha256", "digest": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b", "size": 1500800904},
    "preprocessor_config.json": {"algorithm": "git-blob-sha1", "digest": "2e52d8e8492b5c496ae04c37bfa09760469fb18b", "size": 394},
    "special_tokens_map.json": {"algorithm": "git-blob-sha1", "digest": "8d6368f7e735fbe4781bf6e956b7c6ad0586df80", "size": 636},
    "tokenizer.json": {"algorithm": "sha256", "digest": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322", "size": 34363039},
    "tokenizer.model": {"algorithm": "sha256", "digest": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2", "size": 4241003},
    "tokenizer_config.json": {"algorithm": "git-blob-sha1", "digest": "d97c5412159422c3b56fbc99076b1dcc25dd7856", "size": 47164},
}


def _file_digest(path: Path, *, algorithm: str, size: int) -> str:
    """Hash one staged file without materializing it in memory."""

    if algorithm == "git-blob-sha1":
        hasher = hashlib.sha1()
        hasher.update(f"blob {size}\0".encode("ascii"))
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:  # pragma: no cover - the manifest is a source-level constant
        raise ValueError(f"unsupported snapshot integrity algorithm: {algorithm}")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def query_text(row: Mapping[str, Any]) -> str:
    """Format only the question and options used by the query fingerprint."""

    question = str(row.get("question", ""))
    options = row.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("question options must be a mapping")
    normalized_options = {str(key): value for key, value in options.items()}
    ordered = [letter for letter in C.LETTERS if letter in normalized_options]
    ordered.extend(sorted(key for key in normalized_options if key not in ordered))
    choices = "\n".join(f"{letter}. {normalized_options[letter]}" for letter in ordered)
    return f"Question: {question}\nOptions:\n{choices}" if choices else f"Question: {question}"


def encoder_revision() -> str:
    """Return the cache identity string for the pinned model snapshot."""

    return f"{G.FEATURE_ENCODER_MODEL_ID}@{SIGLIP2_MODEL_REVISION}"


def validate_local_model_dir(model_dir: str | Path) -> Path:
    """Validate a staged SigLIP2 snapshot before importing Transformers."""

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(
            f"local SigLIP2 model directory does not exist: {root}; "
            "stage google/siglip2-base-patch16-224 at the pinned revision first"
        )
    marker = root / REVISION_MARKER
    if not marker.is_file():
        raise RuntimeError(
            f"local SigLIP2 model directory {root} is missing {REVISION_MARKER}; "
            f"write the exact pinned SHA {SIGLIP2_MODEL_REVISION} after staging"
        )
    if marker.read_text().strip() != SIGLIP2_MODEL_REVISION:
        raise RuntimeError(
            f"{marker} must contain the exact pinned model revision {SIGLIP2_MODEL_REVISION}"
        )
    missing = [name for name in G.FEATURE_MODEL_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"local SigLIP2 model directory {root} is missing: {', '.join(missing)}; "
            "no network download is attempted"
        )
    mismatched = []
    for name, expected in SNAPSHOT_FILE_INTEGRITY.items():
        path = root / name
        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            mismatched.append(
                f"{name} size {actual_size} (expected {expected['size']})"
            )
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
            f"local SigLIP2 files do not match the immutable HF snapshot "
            f"{SIGLIP2_MODEL_REVISION}: {'; '.join(mismatched)}"
        )
    return root


def write_video_cache(
    cache_dir: str | Path,
    *,
    video_checksum: str,
    candidate_indices: Sequence[int],
    candidate_features: Sequence[Sequence[float]],
    queries: Mapping[str, Mapping[str, Any]],
    query_features: Mapping[str, Sequence[float]],
    candidate_policy: Mapping[str, Any],
    encoder_revision_value: str,
    fps: float = 1.0,
) -> Path:
    """Write one cache from already computed features; useful for pure tests."""

    if len(candidate_indices) != len(candidate_features):
        raise ValueError("candidate indices and features must have equal lengths")
    if not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be positive")
    candidates = [
        {
            "frame_index": int(index),
            "timestamp": float(index) / float(fps),
            "feature": list(feature),
        }
        for index, feature in zip(candidate_indices, candidate_features)
    ]
    fingerprints = {
        query_id: G.query_input_fingerprint(query)
        for query_id, query in queries.items()
    }
    return G.write_feature_cache(
        cache_dir,
        video_checksum=video_checksum,
        encoder_revision=encoder_revision_value,
        candidate_policy=candidate_policy,
        preprocessing=G.FEATURE_PREPROCESSING,
        candidates=candidates,
        query_features=query_features,
        query_fingerprints=fingerprints,
    )


def _load_local_encoder(model_dir: Path, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "building SigLIP2 caches requires the installed torch and Transformers "
            "dependencies; no dependency or model download is attempted"
        ) from exc
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_dir), revision=SIGLIP2_MODEL_REVISION, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(model_dir), revision=SIGLIP2_MODEL_REVISION, local_files_only=True
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not load staged SigLIP2 files from {model_dir} with local_files_only=True"
        ) from exc
    model.to(device).eval()
    return torch, processor, model


def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _feature_tensor(output: Any, torch: Any) -> Any:
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if not hasattr(output, "float") or not hasattr(output, "detach"):
        raise RuntimeError("SigLIP2 feature output is not a tensor")
    return torch.nn.functional.normalize(output.float(), dim=-1)


def _batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def encode_images(
    processor: Any,
    model: Any,
    frames: Sequence[Any],
    *,
    torch: Any,
    device: str,
    batch_size: int,
    input_data_format: str | None = None,
) -> list[list[float]]:
    """Encode RGB frames with SigLIP2 and return L2-normalized CPU vectors."""

    from PIL import Image

    vectors: list[list[float]] = []
    with torch.inference_mode():
        for batch in _batches(frames, batch_size):
            if input_data_format is None:
                images = [Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame for frame in batch]
            else:
                import numpy as np

                images = [
                    np.asarray(frame.convert("RGB") if isinstance(frame, Image.Image) else frame)
                    for frame in batch
                ]
            processor_kwargs = {"images": images, "return_tensors": "pt"}
            if input_data_format is not None:
                processor_kwargs["input_data_format"] = input_data_format
            inputs = processor(**processor_kwargs)
            output = model.get_image_features(**_move_inputs(inputs, device))
            vectors.extend(_feature_tensor(output, torch).detach().cpu().tolist())
    return vectors


def encode_queries(
    processor: Any,
    model: Any,
    texts: Sequence[str],
    *,
    torch: Any,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    """Encode query text with the processor's fixed max-length tokenization."""

    vectors: list[list[float]] = []
    with torch.inference_mode():
        for batch in _batches(texts, batch_size):
            inputs = processor(
                text=list(batch),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            output = model.get_text_features(**_move_inputs(inputs, device))
            vectors.extend(_feature_tensor(output, torch).detach().cpu().tolist())
    return vectors


def _cache_matches_queries(cache: Mapping[str, Any], queries: Mapping[str, Mapping[str, Any]]) -> bool:
    fingerprints = cache.get("query_fingerprints", {})
    return set(queries) <= set(fingerprints) and all(
        fingerprints[query_id] == G.query_input_fingerprint(query)
        for query_id, query in queries.items()
    )


@contextmanager
def cache_writer_lock(cache_dir: str | Path):
    """Allow one cache builder process to write a directory at a time."""

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".build_feature_cache.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"feature-cache directory is busy: {root} (remove {lock_path} only after confirming no builder runs)"
        ) from exc
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
    validate_local_model_dir(args.model_dir)
    cache_dir = C.work_path(args.feature_cache_dir)
    with cache_writer_lock(cache_dir):
        return _build_caches(args, cache_dir)


def _build_caches(args: argparse.Namespace, cache_dir: Path) -> list[Path]:
    model_dir = validate_local_model_dir(args.model_dir)
    split_path = C.work_path(SPLITS[args.split])
    with split_path.open() as stream:
        rows = json.load(stream)["QA"]
    manifest = C.load_manifest()
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_path"]].append(row)

    policy = G.candidate_policy(args.candidate_rate, args.candidate_cap)
    revision_value = encoder_revision()
    pending: list[tuple[str, str, dict[str, Any], list[int], Path, Mapping[str, Any] | None]] = []
    output_paths: list[Path] = []
    video_checksums: dict[str, str] = {}
    for relative_path, video_rows in by_video.items():
        try:
            info = manifest[relative_path]
        except KeyError as exc:
            raise KeyError(f"{relative_path!r} is absent from {C.manifest_path()}") from exc
        checksum = video_checksums.get(relative_path)
        if checksum is None:
            checksum = C.path_checksum(C.dataset_dir() / relative_path)
            video_checksums[relative_path] = checksum
        indices = G.candidate_index_list(
            int(info["n_frames"]), float(info["fps"]), policy["rate"], policy["cap"]
        )
        path = G.feature_cache_path(
            cache_dir, checksum, revision_value, policy, G.FEATURE_PREPROCESSING
        )
        queries = {row["qa_id"]: row for row in video_rows}
        try:
            cache = G.load_feature_cache(path, expected_cache_key=path.stem)
        except (FileNotFoundError, RuntimeError, ValueError):
            cache = None
        if (
            cache is not None
            and _cache_matches_queries(cache, queries)
            and G.candidate_grid_matches(cache["candidates"], indices)
        ):
            output_paths.append(path)
            continue
        pending.append((relative_path, checksum, queries, indices, path, cache))

    if not pending:
        return output_paths

    torch, processor, model = _load_local_encoder(model_dir, args.device)
    for relative_path, checksum, queries, indices, path, cache in pending:
        if cache is not None and not G.candidate_grid_matches(cache["candidates"], indices):
            cache = None
        cached_candidates = {}
        if cache is not None and G.candidate_grid_matches(cache["candidates"], indices):
            cached_candidates = {
                int(candidate["frame_index"]): candidate["feature"]
                for candidate in cache["candidates"]
            }
        if len(cached_candidates) == len(indices) and set(cached_candidates) == set(indices):
            candidate_features = [cached_candidates[index] for index in indices]
        else:
            candidate_features = []
            for frame_batch in C.iter_read_frames(relative_path, indices, args.batch_size):
                candidate_features.extend(
                    encode_images(
                        processor,
                        model,
                        frame_batch,
                        torch=torch,
                        device=args.device,
                        batch_size=args.batch_size,
                    )
                )
        query_features = dict((cache or {}).get("query_features", {}))
        query_fingerprints = dict((cache or {}).get("query_fingerprints", {}))
        query_ids = [
            query_id
            for query_id, query in queries.items()
            if query_id not in query_features
            or query_fingerprints.get(query_id) != G.query_input_fingerprint(query)
        ]
        query_features_rows = encode_queries(
            processor,
            model,
            [query_text(queries[query_id]) for query_id in query_ids],
            torch=torch,
            device=args.device,
            batch_size=args.batch_size,
        ) if query_ids else []
        query_features.update(dict(zip(query_ids, query_features_rows)))
        query_fingerprints.update(
            {query_id: G.query_input_fingerprint(queries[query_id]) for query_id in query_ids}
        )
        written = G.write_feature_cache(
            cache_dir,
            video_checksum=checksum,
            encoder_revision=revision_value,
            candidate_policy=policy,
            preprocessing=G.FEATURE_PREPROCESSING,
            candidates=[
                {
                    "frame_index": index,
                    "timestamp": index / max(float(manifest[relative_path]["fps"]), 1e-6),
                    "feature": feature,
                }
                for index, feature in zip(indices, candidate_features)
            ],
            query_features=query_features,
            query_fingerprints=query_fingerprints,
        )
        output_paths.append(written)
        print(f"wrote {written} for {relative_path}")
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--model-dir", required=True, help="local staged SigLIP2 snapshot directory")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--candidate-rate", type=float, default=G.DEFAULT_CANDIDATE_RATE)
    parser.add_argument("--candidate-cap", type=int, default=G.DEFAULT_CANDIDATE_CAP)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.candidate_rate <= 0 or args.candidate_cap <= 0 or args.batch_size <= 0:
        parser.error("candidate-rate, candidate-cap, and batch-size must be positive")
    try:
        build_caches(args)
    except (KeyError, RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
