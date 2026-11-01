"""Standalone Phase 8 pretrained-backbone scorer.

This scorer intentionally owns a separate output contract.  It reuses the
frozen frame sampling, prompt, option permutation, split and resumability
helpers from :mod:`common`, while keeping model-family loading differences in
``phase8_adapters``.  The Stage A ``score.py`` path is not imported or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import phase8_adapters as A


SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def manifest_path(score_path: str | os.PathLike[str]) -> Path:
    return C.score_manifest_path(score_path)


def summary_path(score_path: str | os.PathLike[str]) -> Path:
    """Return the separate runtime telemetry sidecar for a Phase 8 score."""

    path = Path(score_path)
    return path.with_name(path.name + ".summary.json")


def remap_shown_probabilities(shown_prob: Iterable[float], permutation: Iterable[str]) -> list[float]:
    """Map probabilities for shown A-D options back to original letters."""

    values = list(shown_prob)
    actual = list(permutation)
    if len(values) != len(C.LETTERS) or sorted(actual) != sorted(C.LETTERS):
        raise ValueError("probabilities and permutation must each contain exactly four A-D entries")
    original = [0.0] * len(C.LETTERS)
    for shown_index, original_letter in enumerate(actual):
        original[C.LETTERS.index(original_letter)] = float(values[shown_index])
    return original


def build_prompt(q: dict[str, Any], permutation: Iterable[str]) -> tuple[str, list[str]]:
    """Return the shared question text and normalized option permutation."""

    shown, actual = C.permute_options(q, permutation)
    return C.user_text(q, shown), actual


def frame_provenance(indices: Iterable[int], info: dict[str, Any]) -> dict[str, Any]:
    """Bind sampled source indices to canonical timestamps when available."""

    selected = [int(index) for index in indices]
    n_frames = int(info.get("n_frames", 0))
    if not selected or n_frames < 1 or any(index < 0 or index >= n_frames for index in selected):
        raise ValueError("frame indices must be non-empty and within the source video")
    raw = info.get("frame_timestamps_s")
    if raw is None:
        raw = info.get("frame_timestamps", info.get("source_timestamps_s"))
    if raw is not None:
        if not isinstance(raw, list) or len(raw) != n_frames:
            raise ValueError("frame timestamp list must match the source frame count")
        timestamps = [float(raw[index]) for index in selected]
        source = str(info.get("timestamp_source", "presentation_timestamp"))
    else:
        fps = float(info.get("fps", 0.0))
        if fps <= 0:
            raise ValueError("source FPS must be positive when canonical timestamps are absent")
        timestamps = [float(index) / fps for index in selected]
        source = "index_divided_by_fps"
    return {
        "frame_indices": selected,
        "frame_timestamps_s": timestamps,
        "timestamp_source": source,
    }


def _read_selection_file(path: str | None, *, label: str, split: str) -> set[str] | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.casefold() == ".json":
        try:
            payload = json.loads(source.read_text())
            section = payload["splits"][split]
            values = section["ids"] if isinstance(section, dict) else section
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} JSON must contain splits.{split}.ids") from exc
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{label} JSON split {split} must contain a string ids list")
    else:
        values = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} file must contain unique non-empty lines")
    return set(values)


def _selection_metadata(
    path: str | None,
    *,
    label: str,
    split: str,
    selected_question_count: int,
) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    policy = (
        f"unique IDs from splits.{split}.ids"
        if label == "QA IDs"
        else "unique video paths from newline-delimited lines"
    )
    return {
        "path": str(source),
        "sha256": digest,
        "split": split,
        "selected_question_count": int(selected_question_count),
        "ids_policy": policy,
    }


def _load_timestamp_records(path: str | os.PathLike[str] | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    source = C.work_path(path)
    with source.open() as stream:
        payload = json.load(stream)
    records = payload.get("records", payload) if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        raise ValueError("frame manifest must contain a records mapping")
    return {str(video): record for video, record in records.items() if isinstance(record, dict)}


def expected_manifest(
    *,
    family: str,
    model_id: str,
    revision: str,
    split: str,
    split_checksum: str,
    offsets: list[float],
    rots: int,
    frame_sampling: str,
    shard: int,
    nshards: int,
    dataset_revision: str | None = None,
    frame_manifest: str | None = None,
    selection: dict[str, Any] | None = None,
    precision: str = "bfloat16",
    adapter: str | dict[str, Any] = "none",
) -> dict[str, Any]:
    """Construct the immutable Phase 8 run identity."""

    if dataset_revision is None:
        try:
            dataset_revision = C.dataset_revision()
        except RuntimeError:
            # Pure contract checks can run without a runtime WORK tree.  The
            # real scorer always has WORK and therefore records its revision.
            dataset_revision = "unknown"
    return {
        "schema_version": 1,
        "output_kind": "phase8_pretrained_score",
        "phase": "phase8",
        "model_family": family,
        "model_id": model_id,
        "model_revision": revision,
        "model_loader": (
            "AutoModelForMultimodalLM" if family == "qwen38" else "Qwen3VLForConditionalGeneration"
        ),
        "precision": precision,
        "chat_template": {
            "generation_prompt": True,
            "non_thinking": family == "qwen38",
        },
        "adapter": adapter,
        "dataset_revision": dataset_revision,
        "split": split,
        "split_checksum": split_checksum,
        "offsets": list(offsets),
        "rots": int(rots),
        "view_count": len(offsets) * int(rots),
        "frame_sampling": frame_sampling,
        "frames_per_view": C.MAX_FRAMES,
        "frame_metadata": {
            "source_indices": "VideoMetadata.frames_indices",
            "timestamps": "frame_timestamps_s",
            "timestamp_binding": "source_frame_index",
            "canonical_manifest": frame_manifest,
        },
        "answer_labels": {"letters": list(C.LETTERS), "token_count": 1},
        "selection": selection,
        "shard": int(shard),
        "nshards": int(nshards),
    }


def ensure_output_contract(out: Path, expected: dict[str, Any]) -> None:
    """Create or validate a resumable Phase 8 JSONL and its manifest."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = manifest_path(out)
    if sidecar.exists():
        try:
            actual = json.loads(sidecar.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid score manifest: {sidecar}") from exc
        comparable = json.loads(json.dumps(actual))
        answer_labels = comparable.get("answer_labels")
        if isinstance(answer_labels, dict):
            # Token IDs are bound after loading the processor and validated
            # separately by bind_manifest_tokenization(). They are runtime
            # evidence, not part of the static output contract.
            answer_labels.pop("tokenization", None)
        if comparable != expected:
            raise RuntimeError(f"manifest mismatch: {sidecar}; choose a new Phase 8 output")
        return
    if out.exists() and out.stat().st_size:
        raise RuntimeError(f"existing score file has no Phase 8 manifest: {out}")
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    temporary.replace(sidecar)


def bind_manifest_tokenization(out: Path, tokenization: dict[str, Any]) -> None:
    """Bind processor token IDs after loading, while preserving resume checks."""

    sidecar = manifest_path(out)
    try:
        manifest = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid score manifest: {sidecar}") from exc
    labels = manifest.setdefault("answer_labels", {})
    previous = labels.get("tokenization")
    if previous not in (None, tokenization):
        raise RuntimeError(f"tokenization mismatch: {sidecar}; choose a new Phase 8 output")
    labels["tokenization"] = tokenization
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(sidecar)


def _move_inputs(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def input_telemetry(encoded: dict[str, Any]) -> dict[str, Any]:
    """Collect only shape information actually returned by the processor."""

    indicators: dict[str, Any] = {}
    for key in ("input_ids", "video_grid_thw", "pixel_values_videos", "pixel_values"):
        shape = _shape(encoded.get(key))
        if shape is not None:
            indicators[f"{key}_shape"] = shape
    input_shape = indicators.get("input_ids_shape")
    indicators["input_token_count"] = input_shape[-1] if input_shape else None
    indicators["video_grid_thw_values"] = _tensor_values(encoded.get("video_grid_thw"))
    return indicators


def _tensor_values(value: Any) -> Any:
    """Return JSON-safe tensor values when the processor exposes them."""

    if value is None:
        return None
    try:
        detached = value.detach() if hasattr(value, "detach") else value
        cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
        values = cpu_value.tolist() if hasattr(cpu_value, "tolist") else None
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    if values is None:
        return None
    try:
        json.dumps(values)
    except (TypeError, ValueError):
        return None
    return values


def _sync_cuda(torch_module: Any) -> None:
    synchronize = getattr(getattr(torch_module, "cuda", None), "synchronize", None)
    if synchronize is not None:
        synchronize()


def _host_max_rss_bytes() -> int | None:
    """Return Linux/macOS max RSS in bytes when the platform exposes it."""

    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes.  The H200 runtime is Linux.
    return value * 1024 if sys.platform.startswith("linux") else value


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _latency_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _summary_values(previous: dict[str, Any], key: str) -> list[float]:
    samples = previous.get("latency_samples_s", {})
    values = samples.get(key, []) if isinstance(samples, dict) else []
    if not isinstance(values, list):
        return []
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return []


def _model_dtype(model: Any) -> str | None:
    try:
        parameter = next(model.parameters())
        return str(parameter.dtype)
    except (AttributeError, StopIteration, TypeError, RuntimeError):
        return None


def _gpu_name(torch_module: Any) -> str | None:
    try:
        return str(torch_module.cuda.get_device_name(0))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _parse_offsets(raw: str) -> list[float]:
    offsets = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not offsets or any(not 0 <= value < 1 for value in offsets):
        raise ValueError("offsets must be comma-separated values in [0, 1)")
    return offsets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", "--model-family", dest="family", default="qwen38")
    parser.add_argument("--model-id", help="override the family default model ID")
    parser.add_argument(
        "--revision",
        help="explicit Hugging Face revision override; defaults to the family-pinned revision",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        nargs="+",
        required=True,
        help="one or more splits; use an {split} --out template for multiple splits",
    )
    parser.add_argument("--out", required=True, help="new Phase 8 JSONL score path or {split} template under WORK")
    parser.add_argument("--offsets", default=",".join(str(value) for value in C.DEFAULT_SCORE_OFFSETS))
    parser.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    parser.add_argument("--frame-sampling", choices=C.FRAME_SAMPLING_MODES, default=C.DEFAULT_FRAME_SAMPLING)
    parser.add_argument("--frame-manifest", help="optional canonical frame manifest with source timestamps")
    parser.add_argument("--qa-ids-file", help="newline-delimited IDs or split-aware pilot JSON manifest")
    parser.add_argument("--video-paths-file", help="newline-delimited deterministic pilot video paths")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--stop-file", help="stop after the current complete record")
    parser.add_argument(
        "--adapter",
        help="optional PEFT adapter directory to load on top of the selected base model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = A.resolve_model_spec(args.family, model_id=args.model_id, revision=args.revision)
        offsets = _parse_offsets(args.offsets)
    except ValueError as exc:
        _parser().error(str(exc))
    if not 1 <= args.rots <= 4:
        _parser().error("rots must be in [1, 4]")
    if args.nshards < 1 or not 0 <= args.shard < args.nshards:
        _parser().error("shard must be in [0, nshards)")

    if len(args.split) > 1 and "{split}" not in args.out:
        _parser().error("multiple splits require {split} in --out")
    outputs = {
        split: C.work_path(args.out.replace("{split}", split))
        for split in args.split
    }
    adapter_path = C.work_path(args.adapter).resolve() if args.adapter else None
    adapter_record: str | dict[str, Any] = "none"
    if adapter_path is not None:
        if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
            _parser().error(f"adapter must be a PEFT directory with adapter_config.json: {adapter_path}")
        adapter_record = {
            "kind": "peft",
            "path": str(adapter_path),
            "sha256": C.path_checksum(adapter_path),
        }
    frame_manifest_path = str(C.work_path(args.frame_manifest).resolve()) if args.frame_manifest else None
    split_contexts: list[dict[str, Any]] = []
    for split in args.split:
        split_path = C.work_path(SPLITS[split])
        with split_path.open() as stream:
            all_rows = json.load(stream)["QA"]
        split_digest = C.split_checksum(all_rows)
        rows = all_rows[args.shard :: args.nshards]
        qa_ids = _read_selection_file(args.qa_ids_file, label="QA IDs", split=split)
        video_paths = _read_selection_file(args.video_paths_file, label="video paths", split=split)
        if qa_ids is not None or video_paths is not None:
            rows = [
                row
                for row in rows
                if (qa_ids is None or row.get("qa_id") in qa_ids)
                and (video_paths is None or row.get("video_path") in video_paths)
            ]
            if not rows:
                _parser().error("pilot selection files selected no questions")
            if qa_ids is not None and qa_ids - {row.get("qa_id") for row in rows}:
                _parser().error("QA selection includes IDs absent from the selected split/shard")
            if video_paths is not None and video_paths - {row.get("video_path") for row in rows}:
                _parser().error("video selection includes paths absent from the selected split/shard")
        selection = None
        if args.qa_ids_file is not None or args.video_paths_file is not None:
            selection = {
                "qa_ids_file": _selection_metadata(
                    args.qa_ids_file,
                    label="QA IDs",
                    split=split,
                    selected_question_count=len(rows),
                ),
                "video_paths_file": _selection_metadata(
                    args.video_paths_file,
                    label="video paths",
                    split=split,
                    selected_question_count=len(rows),
                ),
            }
        expected = expected_manifest(
            family=spec.family,
            model_id=spec.model_id,
            revision=spec.revision,
            split=split,
            split_checksum=split_digest,
            offsets=offsets,
            rots=args.rots,
            frame_sampling=args.frame_sampling,
            shard=args.shard,
            nshards=args.nshards,
            frame_manifest=frame_manifest_path,
            selection=selection,
            adapter=adapter_record,
        )
        out = outputs[split]
        ensure_output_contract(out, expected)
        split_contexts.append({
            "split": split,
            "rows": rows,
            "out": out,
            "done": C.read_completed_jsonl(out),
        })
    stop_file = Path(args.stop_file) if args.stop_file else (
        Path(os.environ["REVA_STOP_FILE"]) if os.environ.get("REVA_STOP_FILE") else None
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 8 video scoring")
    adapter = A.Qwen38Adapter(spec) if spec.family == "qwen38" else A.Qwen3VL32BAdapter(spec)
    model_load_started = time.perf_counter()
    processor, model = adapter.load()
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        print(f"loaded PEFT adapter={adapter_path}")
    label_ids = A.validate_label_tokens(processor)
    label_tokenization = A.label_tokenization_telemetry(processor)
    for context in split_contexts:
        bind_manifest_tokenization(context["out"], label_tokenization)
    device = torch.device("cuda")
    model.to(device).eval()
    model_load_latency_s = time.perf_counter() - model_load_started
    reset_peak_memory_stats = getattr(torch.cuda, "reset_peak_memory_stats", None)
    if reset_peak_memory_stats is not None:
        reset_peak_memory_stats()
    rotations = [list(C.LETTERS[index:] + C.LETTERS[:index]) for index in range(args.rots)]
    timestamp_records = _load_timestamp_records(args.frame_manifest)
    source_manifest = C.load_manifest()
    dtype = _model_dtype(model)
    gpu_name = _gpu_name(torch)
    telemetry: dict[str, dict[str, Any]] = {
        context["split"]: {
            "forward": [], "question": [], "preprocess": [],
            "grid_values": [], "token_counts": [], "elapsed": 0.0,
        }
        for context in split_contexts
    }
    stop_requested = False

    with torch.inference_mode():
        for context in split_contexts:
            split = context["split"]
            out = context["out"]
            done = context["done"]
            split_started = time.perf_counter()
            with out.open("a") as stream:
                for row in context["rows"]:
                    qa_id = row["qa_id"]
                    if qa_id in done:
                        continue
                    if stop_file is not None and stop_file.exists():
                        print(f"stop requested; resume from {out}")
                        stop_requested = True
                        break
                    question_started = time.perf_counter()
                    views = []
                    for offset in offsets:
                        frames, meta = C.get_video(row["video_path"], offset, mode=args.frame_sampling)
                        source_info = dict(source_manifest[row["video_path"]])
                        source_info.update(timestamp_records.get(row["video_path"], {}))
                        provenance = frame_provenance(meta.frames_indices, source_info)
                        for permutation in rotations:
                            preprocess_started = time.perf_counter()
                            encoded, actual_perm = adapter.encode(processor, row, frames, meta, permutation)
                            preprocess_latency_s = time.perf_counter() - preprocess_started
                            model_inputs = _move_inputs(encoded, device)
                            _sync_cuda(torch)
                            forward_started = time.perf_counter()
                            logits = model(**model_inputs).logits[0, -1, label_ids].float()
                            _sync_cuda(torch)
                            forward_latency_s = time.perf_counter() - forward_started
                            telemetry[split]["forward"].append(forward_latency_s)
                            telemetry[split]["preprocess"].append(preprocess_latency_s)
                            shown_prob = torch.softmax(logits, dim=-1).cpu().numpy()
                            original_prob = remap_shown_probabilities(shown_prob, actual_perm)
                            view = {
                                "offset": float(offset),
                                "perm": "".join(actual_perm),
                                "p": original_prob,
                                "argmax_shown": C.LETTERS[int(np.argmax(shown_prob))],
                                "preprocess_latency_s": preprocess_latency_s,
                                "forward_latency_s": forward_latency_s,
                                "actual_frame_count": len(meta.frames_indices),
                                **provenance,
                            }
                            indicators = input_telemetry(encoded)
                            view["input_telemetry"] = indicators
                            if indicators.get("input_token_count") is not None:
                                telemetry[split]["token_counts"].append(indicators["input_token_count"])
                            if indicators.get("video_grid_thw_values") is not None:
                                telemetry[split]["grid_values"].append(indicators["video_grid_thw_values"])
                            views.append(view)
                    if len(views) != len(offsets) * args.rots:
                        raise RuntimeError(f"incomplete Phase 8 view set for {qa_id}")
                    mean = np.mean([view["p"] for view in views], axis=0).tolist()
                    question_latency_s = time.perf_counter() - question_started
                    telemetry[split]["question"].append(question_latency_s)
                    result = {
                        "qa_id": qa_id,
                        "model_family": spec.family,
                        "model_id": spec.model_id,
                        "model_revision": spec.revision,
                        "p": mean,
                        "pred": C.LETTERS[int(np.argmax(mean))],
                        "question_latency_s": question_latency_s,
                        "views": views,
                    }
                    stream.write(json.dumps(result, separators=(",", ":")) + "\n")
                    stream.flush()
                    done.add(qa_id)
            telemetry[split]["elapsed"] = time.perf_counter() - split_started
            if stop_requested:
                break

    max_memory_allocated = getattr(torch.cuda, "max_memory_allocated", None)
    peak_gpu_memory = None
    if max_memory_allocated is not None:
        try:
            peak_gpu_memory = int(max_memory_allocated())
        except (RuntimeError, TypeError, ValueError):
            peak_gpu_memory = None
    max_rss = _host_max_rss_bytes()
    for context in split_contexts:
        split = context["split"]
        out = context["out"]
        prior = _read_summary(summary_path(out))
        forward_samples = _summary_values(prior, "forward") + telemetry[split]["forward"]
        question_samples = _summary_values(prior, "question") + telemetry[split]["question"]
        preprocess_samples = _summary_values(prior, "preprocess") + telemetry[split]["preprocess"]
        prior_grid_values = prior.get("vision_grid_thw_values")
        grid_values = prior_grid_values if isinstance(prior_grid_values, list) else []
        for value in telemetry[split]["grid_values"]:
            if value not in grid_values:
                grid_values.append(value)
        prior_token_counts = prior.get("input_token_count_values")
        token_counts = prior_token_counts if isinstance(prior_token_counts, list) else []
        for value in telemetry[split]["token_counts"]:
            if value not in token_counts:
                token_counts.append(value)
        previous_model_load = float(prior.get("model_load_latency_s_total", 0.0) or 0.0)
        previous_model_load_count = int(prior.get("model_load_count", 0) or 0)
        total_model_load = previous_model_load + model_load_latency_s
        model_load_count = previous_model_load_count + 1
        model_load_mean = total_model_load / model_load_count
        previous_elapsed = float(prior.get("scoring_elapsed_s_total", 0.0) or 0.0)
        scoring_elapsed = previous_elapsed + telemetry[split]["elapsed"]
        previous_peak_gpu = prior.get("peak_gpu_memory_bytes")
        if isinstance(previous_peak_gpu, (int, float)) and peak_gpu_memory is not None:
            peak_gpu_memory_total = max(int(previous_peak_gpu), peak_gpu_memory)
        else:
            peak_gpu_memory_total = peak_gpu_memory if peak_gpu_memory is not None else previous_peak_gpu
        previous_rss = prior.get("host_max_rss_bytes")
        if isinstance(previous_rss, (int, float)) and max_rss is not None:
            host_max_rss_total = max(int(previous_rss), max_rss)
        else:
            host_max_rss_total = max_rss if max_rss is not None else previous_rss
        summary: dict[str, Any] = {
            "schema_version": 2,
            "output_kind": "phase8_runtime_summary",
            "score_path": str(out.resolve()),
            "manifest_path": str(manifest_path(out).resolve()),
            "model_family": spec.family,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "adapter": adapter_record,
            "precision": "bfloat16",
            "dtype": dtype,
            "label_tokenization": label_tokenization,
            "gpu_name": gpu_name,
            "model_load_latency_s": total_model_load,
            "model_load_latency_s_total": total_model_load,
            "model_load_latency_s_mean": model_load_mean,
            "model_load_latency_s_this_invocation": model_load_latency_s,
            "model_load_count": model_load_count,
            "completed_question_count": len(context["done"]),
            "question_count_this_invocation": len(telemetry[split]["question"]),
            "forward_count_this_invocation": len(telemetry[split]["forward"]),
            "forward_count": len(forward_samples),
            "scoring_elapsed_s_total": scoring_elapsed,
            "questions_per_second": (
                len(question_samples) / scoring_elapsed if scoring_elapsed > 0 else None
            ),
            "forwards_per_second": (
                len(forward_samples) / scoring_elapsed if scoring_elapsed > 0 else None
            ),
            "latency_samples_s": {
                "forward": forward_samples,
                "question": question_samples,
                "preprocess": preprocess_samples,
            },
            "forward_latency_s": _latency_summary(forward_samples),
            "question_latency_s": _latency_summary(question_samples),
            "preprocess_latency_s": _latency_summary(preprocess_samples),
            "vision_grid_thw_values": grid_values or None,
            "input_token_count_values": token_counts or None,
            "vision_grid_thw_values_available": bool(grid_values),
            "input_token_count_available": bool(token_counts),
            "peak_gpu_memory_bytes": peak_gpu_memory_total,
            "host_max_rss_bytes": host_max_rss_total,
        }
        _write_summary(summary_path(out), summary)
        print(f"scored Phase 8 {args.family} {split} shard {args.shard}/{args.nshards}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

