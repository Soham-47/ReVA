"""Opt-in dual-stream scorer; the production ``score.py`` path is untouched.

This entrypoint loads a locally cached expanded Qwen3-VL tokenizer/base model,
the trained PEFT adapter and temporal bundle, and a local InternVideo2
checkpoint.  It samples the normal appearance view plus a dense motion view,
generates ``time-prefix + A-D`` output, decodes the time prefix, and writes
the same ``qa_id``/``p``/``pred``/``views`` fields consumed by ``evaluate.py``.

The command intentionally uses ``local_files_only=True``.  Downloading model
or data assets remains an explicit operator action outside this scorer.

Example after an approved runtime has populated ``WORK`` and local weights::

    WORK=/runtime/reva python code/score_dualstream_temporal.py \\
      --adapter runs/dualstream_temporal_q3vl8b_r16/final_adapter \\
      --split valT --out runs/dualstream_temporal_q3vl8b_r16/valT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import temporal_grounding as TG


SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def _move_inputs(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _duration(row: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> float:
    for key in ("duration", "video_duration", "duration_sec"):
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0:
            return value
    try:
        value = float(manifest[row["video_path"]]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"no valid duration for {row.get('video_path')!r}") from exc
    if value <= 0:
        raise RuntimeError(f"invalid duration for {row.get('video_path')!r}: {value}")
    return value


def _load_bundle_metadata(bundle: Path) -> dict[str, Any]:
    path = bundle / "temporal_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"temporal bundle metadata is missing: {path}")
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or metadata.get("bundle_version") != TG.TEMPORAL_BUNDLE_VERSION:
        raise RuntimeError(f"unsupported temporal bundle: {path}")
    if not isinstance(metadata.get("temporal_runtime_config"), dict):
        raise RuntimeError(
            "temporal bundle has no runtime config; retrain/save with the opt-in branch "
            "so the InternVideo2 checkpoint and segment layout can be reconstructed"
        )
    return metadata


def _runtime_config(bundle_metadata: dict[str, Any], config_path: str | None, checkpoint: str | None) -> dict[str, Any]:
    runtime = dict(bundle_metadata["temporal_runtime_config"])
    if config_path:
        source = Path(config_path)
        with source.open() as stream:
            payload = yaml.safe_load(stream) or {}
        runtime = dict(payload.get("temporal_stream", payload))
    if checkpoint:
        runtime["checkpoint"] = checkpoint
    required = ("checkpoint", "builder")
    missing = [key for key in required if not runtime.get(key)]
    if missing:
        raise RuntimeError(f"temporal runtime config is missing {missing}")
    return runtime


def _verify_temporal_checkpoint_identity(runtime: dict[str, Any], bundle_metadata: dict[str, Any]) -> None:
    """Reject a runtime checkpoint that cannot reproduce the trained stream."""

    expected = bundle_metadata.get("temporal_checkpoint_identity")
    if not isinstance(expected, dict) or not expected.get("sha256"):
        raise RuntimeError(
            "temporal bundle has no verifiable InternVideo2 checkpoint identity; "
            "refusing to score with an unpinned temporal encoder"
        )
    checkpoint = Path(str(runtime["checkpoint"])).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"InternVideo2 checkpoint is missing: {checkpoint}")
    if C.path_checksum(checkpoint) != expected["sha256"]:
        raise RuntimeError(f"InternVideo2 checkpoint checksum differs from the temporal bundle: {checkpoint}")
    expected_size = expected.get("size")
    if expected_size is not None and checkpoint.stat().st_size != int(expected_size):
        raise RuntimeError(f"InternVideo2 checkpoint size differs from the temporal bundle: {checkpoint}")


def _ensure_output_contract(out: Path, expected: dict[str, Any]) -> set[str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = C.score_manifest_path(out)
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text())
        if actual != expected:
            raise RuntimeError(f"dual-stream score manifest mismatch: {manifest_path}")
    elif out.exists() and out.stat().st_size:
        raise RuntimeError(f"existing score file has no manifest: {out}")
    else:
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n")
    return C.read_completed_jsonl(out)


def _parse_offsets(raw: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("offsets must be comma-separated floats") from exc
    if not values or any(not 0 <= value < 1 for value in values):
        raise ValueError("offsets must be non-empty values in [0, 1)")
    return values


def _build_manifest(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    adapter: Path,
    bundle: Path,
    runtime: dict[str, Any],
    bundle_metadata: dict[str, Any],
    offsets: list[float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "output_kind": "dualstream_temporal_score",
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REV,
        "model_loader": "Qwen3VLForConditionalGeneration+DualStreamQwen3VL",
        "adapter": str(adapter.resolve()),
        "adapter_checksum": C.path_checksum(adapter),
        "temporal_bundle": str(bundle.resolve()),
        "temporal_bundle_checksum": C.path_checksum(bundle),
        "temporal_runtime_config": runtime,
        "temporal_checkpoint_identity": bundle_metadata.get("temporal_checkpoint_identity"),
        "split": args.split,
        "task_filter": args.task,
        "split_checksum": C.split_checksum(rows),
        "offsets": offsets,
        "rots": args.rots,
        "appearance_frame_budget": args.frame_budget,
        "temporal_num_frames": int(runtime.get("num_frames", 96)),
        "temporal_num_segments": int(runtime.get("num_segments", 12)),
        "temporal_frames_per_segment": int(runtime.get("frames_per_segment", 8)),
        "answer_labels": {"letters": list(C.LETTERS), "time_bins": int(runtime.get("num_bins", TG.NUM_BINS))},
        "shard": args.shard,
        "nshards": args.nshards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="trained final_adapter directory")
    parser.add_argument("--temporal-bundle", help="bundle directory; defaults to --adapter")
    parser.add_argument("--temporal-config", help="optional YAML temporal_stream override")
    parser.add_argument("--temporal-checkpoint", help="approved local InternVideo2 checkpoint override")
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument(
        "--task",
        help="optional exact task filter applied before sharding (default: score every split row)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--offsets", default=",".join(str(value) for value in C.DEFAULT_SCORE_OFFSETS))
    parser.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    parser.add_argument("--frame-budget", type=int, default=C.MAX_FRAMES)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.rots <= 4 or not 0 <= args.shard < args.nshards:
        parser.error("rots must be 1..4 and shard must be in [0, nshards)")
    if args.frame_budget < 1 or args.max_new_tokens < 1:
        parser.error("frame-budget and max-new-tokens must be positive")
    if args.task is not None and not args.task.strip():
        parser.error("task must be a non-empty exact task name when provided")
    offsets = _parse_offsets(args.offsets)

    adapter = C.work_path(args.adapter)
    bundle = C.work_path(args.temporal_bundle or args.adapter)
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"invalid adapter directory: {adapter}")
    bundle_metadata = _load_bundle_metadata(bundle)
    runtime = _runtime_config(bundle_metadata, args.temporal_config, args.temporal_checkpoint)
    _verify_temporal_checkpoint_identity(runtime, bundle_metadata)
    num_segments = int(runtime.get("num_segments", 12))
    frames_per_segment = int(runtime.get("frames_per_segment", 8))
    num_frames = int(runtime.get("num_frames", num_segments * frames_per_segment))
    if num_frames != num_segments * frames_per_segment:
        raise ValueError("temporal num_frames must equal num_segments * frames_per_segment")

    split_path = C.work_path(SPLITS[args.split])
    with split_path.open() as stream:
        all_rows = json.load(stream)["QA"]
    if args.task is not None:
        all_rows = [row for row in all_rows if row.get("task") == args.task]
        if not all_rows:
            raise ValueError(f"--task matched no rows in {args.split}: {args.task!r}")
    rows = all_rows[args.shard :: args.nshards]
    expected_manifest = _build_manifest(
        args=args,
        rows=all_rows,
        adapter=adapter,
        bundle=bundle,
        runtime=runtime,
        bundle_metadata=bundle_metadata,
        offsets=offsets,
    )
    out = C.work_path(args.out)
    done = _ensure_output_contract(out, expected_manifest)

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for dual-stream video scoring")
    # Training saves the expanded processor/tokenizer beside the adapter.
    # Prefer that exact artifact; falling back to the pinned base processor is
    # safe only for bundles whose canonical token metadata still matches.
    processor_source = adapter if (adapter / "tokenizer_config.json").is_file() else C.MODEL_ID
    processor_kwargs = {"local_files_only": True}
    if processor_source == C.MODEL_ID:
        processor_kwargs["revision"] = C.MODEL_REV
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    vocabulary = TG.ensure_temporal_tokens(model, processor.tokenizer, num_bins=int(runtime.get("num_bins", TG.NUM_BINS)))
    if list(vocabulary.token_names) != bundle_metadata.get("token_names") or list(vocabulary.token_ids) != bundle_metadata.get("token_ids"):
        raise RuntimeError("expanded tokenizer vocabulary does not match the temporal bundle")
    model = C.load_peft_for_inference(model, adapter, merge=False)
    temporal_encoder = TG.load_internvideo2_encoder(runtime)
    feature_dim = TG.infer_temporal_feature_dim(temporal_encoder, expected=runtime.get("feature_dim"))
    stream = TG.InternVideo2TemporalStream(
        temporal_encoder,
        feature_dim=feature_dim,
        hidden_size=int(model.get_input_embeddings().embedding_dim),
        num_segments=num_segments,
        frames_per_segment=frames_per_segment,
        projector_hidden=runtime.get("projector_hidden"),
    )
    model = TG.DualStreamQwen3VL.from_temporal_bundle(
        model,
        stream,
        processor.tokenizer,
        bundle,
        runtime_config=runtime,
    )
    device = torch.device(args.device)
    model.to(device).eval()
    C.configure_attention_backend("sdpa", torch_module=torch)
    manifest = C.load_manifest()
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(args.rots)]

    with out.open("a") as output, torch.inference_mode():
        for row in rows:
            if row["qa_id"] in done:
                continue
            duration = _duration(row, manifest)
            views = []
            for offset in offsets:
                appearance_frames, appearance_meta = C.get_video(
                    row["video_path"], offset, frame_budget=args.frame_budget
                )
                motion_frames, _ = C.get_video(
                    row["video_path"], offset, frame_budget=num_frames
                )
                motion_pixels = TG.preprocess_temporal_frames(
                    motion_frames,
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    image_size=int(runtime.get("image_size", 224)),
                ).unsqueeze(0).to(device)
                for permutation in rotations:
                    encoded, actual_perm = C.encode(
                        processor,
                        row,
                        appearance_frames,
                        appearance_meta,
                        permutation,
                        enable_thinking=False,
                    )
                    encoded = _move_inputs(encoded, device)
                    encoded["temporal_pixel_values"] = motion_pixels
                    encoded["temporal_segment_mask"] = torch.ones(
                        (1, num_segments), dtype=torch.bool, device=device
                    )
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                    prompt_length = int(encoded["input_ids"].shape[-1])
                    response = processor.tokenizer.decode(
                        generated[0, prompt_length:].tolist(),
                        skip_special_tokens=False,
                    )
                    times, shown_letter = TG.parse_temporal_answer(response, duration)
                    if shown_letter not in C.LETTERS:
                        raise RuntimeError(
                            f"dual-stream generation produced no standalone A-D answer for {row['qa_id']}: {response!r}"
                        )
                    original_letter = actual_perm[C.LETTERS.index(shown_letter)]
                    probabilities = [0.0] * len(C.LETTERS)
                    probabilities[C.LETTERS.index(original_letter)] = 1.0
                    views.append(
                        {
                            "offset": offset,
                            "perm": "".join(actual_perm),
                            "p": probabilities,
                            "argmax_shown": shown_letter,
                            "pred_shown": shown_letter,
                            "response": response,
                            "time_seconds": list(times) if times is not None else None,
                        }
                    )
            mean = np.mean([view["p"] for view in views], axis=0).tolist()
            result = {
                "qa_id": row["qa_id"],
                "p": mean,
                "pred": C.LETTERS[int(np.argmax(mean))],
                "views": views,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
