"""Shared paths, frame sampling, prompts, option mapping, and model encoding.

The module keeps optional GPU/video dependencies inside the functions that need
them so split and submission checks can run on a login node with only Python.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REV = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
DATASET_REPO = "ReVA-Benchmark/ReVA"
DATASET_REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"
FPS_TARGET = 4.0
MIN_FRAMES = 16
MAX_FRAMES = 16
LETTERS = ("A", "B", "C", "D")
DEFAULT_SCORE_OFFSETS = (0.0, 0.5)
DEFAULT_SCORE_ROTS = 2
FRAME_SAMPLING_MODES = ("legacy", "endpoints")
DEFAULT_FRAME_SAMPLING = "legacy"
VIDEO_SIZE = {"shortest_edge": 131072, "longest_edge": 16 * 640 * 640}
SYSTEM_PROMPT = (
    "You are an expert analyst of aerial drone (UAV) videos. Watch every frame carefully. "
    "Pay attention to small objects, counts, camera motion versus object motion, spatial layout, "
    "and the timestamp shown before each frame. Then answer the multiple-choice question."
)


def work_dir(work: str | os.PathLike[str] | None = None) -> Path:
    """Return the runtime WORK directory, never a repository-relative artifact path."""

    value = work or os.environ.get("WORK")
    if not value:
        raise RuntimeError("WORK is required; set it to the HPC runtime tree")
    return Path(value).expanduser().resolve()


def work_path(path: str | os.PathLike[str], work: str | os.PathLike[str] | None = None) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else work_dir(work) / p


def dataset_dir(work: str | os.PathLike[str] | None = None) -> Path:
    return work_dir(work) / "data" / "ReVA"


def manifest_path(work: str | os.PathLike[str] | None = None) -> Path:
    return work_dir(work) / "data" / "manifest.json"


def dataset_revision(work: str | os.PathLike[str] | None = None) -> str:
    path = work_dir(work) / "data" / "dataset_revision.json"
    if not path.exists():
        return "unknown"
    with path.open() as f:
        record = json.load(f)
    return str(record.get("revision", "unknown"))


def split_checksum(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def estimate_training_hours(
    samples: int,
    epochs: float,
    grad_accum: int,
    seconds_per_step: float,
    batch_size: int = 1,
) -> float:
    """Estimate wall time from measured optimizer-step seconds."""

    if any(value <= 0 for value in (samples, epochs, grad_accum, seconds_per_step, batch_size)):
        raise ValueError("training estimate inputs must be positive")
    updates = math.ceil(samples / (batch_size * grad_accum)) * epochs
    return updates * seconds_per_step / 3600


def configure_attention_backend(attn_implementation: str, torch_module=None) -> None:
    """Keep Qwen vision SDPA off the unstable cuDNN fused-MHA path on H200."""

    if attn_implementation != "sdpa":
        return
    if torch_module is None:
        import torch as torch_module
    enable_cudnn_sdp = getattr(torch_module.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is None:
        raise RuntimeError("the installed PyTorch build has no cuDNN SDPA control")
    enable_cudnn_sdp(False)
    if os.environ.get("REVA_SDPA_MATH_ONLY") == "1":
        flash = getattr(torch_module.backends.cuda, "enable_flash_sdp", None)
        mem = getattr(torch_module.backends.cuda, "enable_mem_efficient_sdp", None)
        math_sdp = getattr(torch_module.backends.cuda, "enable_math_sdp", None)
        missing = [
            name
            for name, fn in (("flash", flash), ("mem_efficient", mem), ("math", math_sdp))
            if fn is None
        ]
        if missing:
            raise RuntimeError(
                "REVA_SDPA_MATH_ONLY=1 but this PyTorch build lacks SDPA toggles: " + ", ".join(missing)
            )
        flash(False)
        mem(False)
        math_sdp(True)


def path_checksum(path: str | os.PathLike[str]) -> str:
    """Hash an adapter file/directory without including its path or timestamps."""

    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not target.is_dir():
        raise FileNotFoundError(target)
    for child in sorted(p for p in target.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(target)).encode())
        with child.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def score_manifest_path(score_path: str | os.PathLike[str]) -> Path:
    path = Path(score_path)
    return path.with_name(path.name + ".manifest.json")


def read_completed_jsonl(path: Path) -> set[str]:
    """Read completed score IDs; truncate one partial final record after a crash."""

    if not path.exists():
        return set()
    raw = path.read_bytes()
    chunks = raw.splitlines(keepends=True)
    done: set[str] = set()
    offset = 0
    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            offset += len(chunk)
            continue
        try:
            row = json.loads(chunk.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_partial_final = index == len(chunks) - 1 and not chunk.endswith((b"\n", b"\r"))
            if not is_partial_final:
                raise ValueError(f"invalid score JSON at {path}:{index + 1}") from exc
            with path.open("r+b") as f:
                f.truncate(offset)
            break
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or qa_id in done:
            raise ValueError(f"duplicate/invalid qa_id at {path}:{index + 1}")
        done.add(qa_id)
        offset += len(chunk)
    return done


def load_manifest(work: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    with manifest_path(work).open() as f:
        return json.load(f)


def sample_index_list(
    n_frames: int,
    fps: float,
    offset: float,
    *,
    mode: str = DEFAULT_FRAME_SAMPLING,
    frame_budget: int | None = None,
) -> list[int]:
    """Choose an even frame grid without requiring NumPy.

    ``offset`` is a fractional shift of the grid. Clipping and duplication are
    intentional for clips shorter than the minimum frame count.
    """

    n = max(int(n_frames), 1)
    rate = max(float(fps), 1e-6)
    off = float(offset)
    if not 0.0 <= off < 1.0:
        raise ValueError(f"offset must be in [0, 1), got {offset}")
    if mode not in FRAME_SAMPLING_MODES:
        raise ValueError(f"unknown frame sampling mode: {mode}")
    requested = MAX_FRAMES if frame_budget is None else int(frame_budget)
    if requested < 1:
        raise ValueError("frame_budget must be positive")
    k = int(round(n / rate * FPS_TARGET)) if frame_budget is None else requested
    if frame_budget is None:
        k = max(MIN_FRAMES, min(MAX_FRAMES, k))
    k = min(k, max(n, 2))
    k -= k % 2
    k = max(k, 2)
    if mode == "endpoints":
        interior = [
            min(n - 1, max(0, math.floor((i + off) * (n - 1) / (k - 1))))
            for i in range(1, k - 1)
        ]
        return [0, *interior, n - 1]
    return [min(n - 1, max(0, math.floor((i + off) * n / k))) for i in range(k)]


def sample_indices(
    n_frames: int,
    fps: float,
    offset: float,
    *,
    mode: str = DEFAULT_FRAME_SAMPLING,
    frame_budget: int | None = None,
):
    import numpy as np

    return np.asarray(sample_index_list(n_frames, fps, offset, mode=mode, frame_budget=frame_budget), dtype=np.int64)


def _video_metadata(**values: Any) -> Any:
    from transformers.video_utils import VideoMetadata

    try:
        return VideoMetadata(**values)
    except TypeError:
        # Older Transformers releases accepted only these three fields.
        return VideoMetadata(
            total_num_frames=values["total_num_frames"],
            fps=values["fps"],
            frames_indices=values["frames_indices"],
        )


def _read_frames_seek(path, wanted: list[int], fps: float):
    """Fetch sparse frame indices without linearly decoding from frame zero."""

    import av
    import numpy as np

    got: dict[int, Any] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        for index in wanted:
            timestamp = max(0, int((index / fps) / time_base))
            container.seek(timestamp, backward=True, stream=stream)
            for frame in container.decode(stream):
                got[index] = frame.to_ndarray(format="rgb24")
                break
    if not got:
        raise RuntimeError(f"could not decode any requested frame from {path}")
    fallback = got[max(got)]
    return np.stack([got.get(index, fallback) for index in wanted])


def read_frames(
    relative_path: str,
    indices: Iterable[int],
    work: str | os.PathLike[str] | None = None,
    *,
    fps: float | None = None,
):
    import av
    import numpy as np

    wanted = [int(i) for i in indices]
    if not wanted:
        raise ValueError("at least one frame index is required")
    wanted_set = set(wanted)
    path = dataset_dir(work) / relative_path
    max_wanted = max(wanted_set)
    # Long clips: linear decode from t=0 is O(max_index) and can stall training.
    if fps is not None and fps > 0 and (max_wanted >= 512 or len(wanted_set) >= 32):
        return _read_frames_seek(path, wanted, float(fps))

    got: dict[int, Any] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for number, frame in enumerate(container.decode(stream)):
            if number in wanted_set:
                got[number] = frame.to_ndarray(format="rgb24")
            if number >= max_wanted:
                break
    if not got:
        raise RuntimeError(f"could not decode any requested frame from {relative_path}")
    fallback = got[max(got)]
    return np.stack([got.get(index, fallback) for index in wanted])


def iter_read_frames(
    relative_path: str,
    indices: Iterable[int],
    batch_size: int,
    work: str | os.PathLike[str] | None = None,
):
    """Yield requested frames in batches during one decoder pass.

    The decoder stays open while the caller processes each yielded batch, so a
    bounded candidate grid never needs to be materialized as one large array.
    Missing decoded indices use the same last-frame fallback as ``read_frames``.
    """

    import av
    import numpy as np

    wanted = [int(i) for i in indices]
    if not wanted:
        raise ValueError("at least one frame index is required")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    size = int(batch_size)
    batches = [wanted[start : start + size] for start in range(0, len(wanted), size)]
    wanted_set = set(wanted)
    got: dict[int, Any] = {}
    remaining = Counter(wanted)
    fallback_frame = None
    batch_index = 0
    path = dataset_dir(work) / relative_path
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for number, frame in enumerate(container.decode(stream)):
            if number in wanted_set:
                got[number] = frame.to_ndarray(format="rgb24")
                fallback_frame = got[number]
            if batch_index < len(batches) and set(batches[batch_index]) <= got.keys():
                batch = batches[batch_index]
                yield np.stack([got[index] for index in batch])
                for index in batch:
                    remaining[index] -= 1
                    if not remaining[index]:
                        got.pop(index, None)
                batch_index += 1
            if batch_index == len(batches):
                break
    if fallback_frame is None:
        raise RuntimeError(f"could not decode any requested frame from {relative_path}")
    fallback = got[max(got)] if got else fallback_frame
    while batch_index < len(batches):
        yield np.stack([got.get(index, fallback) for index in batches[batch_index]])
        batch_index += 1


def get_video(
    relative_path: str,
    offset: float = 0.0,
    work: str | os.PathLike[str] | None = None,
    *,
    mode: str = DEFAULT_FRAME_SAMPLING,
    frame_budget: int | None = None,
):
    manifest = load_manifest(work)
    try:
        info = manifest[relative_path]
    except KeyError as exc:
        raise KeyError(f"{relative_path!r} is absent from {manifest_path(work)}") from exc
    indices = sample_indices(info["n_frames"], info["fps"], offset, mode=mode, frame_budget=frame_budget)
    frames = read_frames(relative_path, indices, work, fps=float(info["fps"]))
    meta = _video_metadata(
        total_num_frames=int(info["n_frames"]),
        fps=float(info["fps"]),
        frames_indices=indices.tolist(),
        width=int(info["width"]),
        height=int(info["height"]),
        duration=float(info["duration"]),
    )
    return frames, meta


def get_video_by_indices(
    relative_path: str,
    indices: Iterable[int],
    work: str | os.PathLike[str] | None = None,
):
    """Decode an explicit source-frame list and build matching video metadata."""

    manifest = load_manifest(work)
    try:
        info = manifest[relative_path]
    except KeyError as exc:
        raise KeyError(f"{relative_path!r} is absent from {manifest_path(work)}") from exc
    wanted = [int(index) for index in indices]
    if not wanted or any(index < 0 or index >= int(info["n_frames"]) for index in wanted):
        raise ValueError(f"indices must be non-empty and within {relative_path!r}")
    frames = read_frames(relative_path, wanted, work, fps=float(info["fps"]))
    meta = _video_metadata(
        total_num_frames=int(info["n_frames"]),
        fps=float(info["fps"]),
        frames_indices=wanted,
        width=int(info["width"]),
        height=int(info["height"]),
        duration=float(info["duration"]),
    )
    return frames, meta


def downsample_video(frames: Any, meta: Any):
    """Halve a sampled video and rebuild metadata for max-token protection."""

    import numpy as np

    keep = np.arange(0, len(frames), 2, dtype=np.int64)
    keep = keep if len(keep) >= 2 else np.asarray([0, 0], dtype=np.int64)
    indices = [int(meta.frames_indices[i]) for i in keep]
    sampled = frames[keep]
    new_meta = _video_metadata(
        total_num_frames=int(meta.total_num_frames),
        fps=float(meta.fps),
        frames_indices=indices,
        width=int(sampled.shape[2]),
        height=int(sampled.shape[1]),
        duration=float(getattr(meta, "duration", 0.0)),
    )
    return sampled, new_meta


def load_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)


def load_peft_adapters(model, adapter_path: str | os.PathLike[str]):
    """Load the root adapter and every named child adapter in a Phase 6 result.

    The returned names are the exact PEFT adapter names that must be active when
    merging.  Keeping this step separate from ``merge_and_unload`` lets smoke
    checks compare the serialized model before and after merging.
    """

    from peft import PeftModel

    path = Path(adapter_path)
    adapted = PeftModel.from_pretrained(model, str(path))
    child_adapters = sorted(
        child.name
        for child in path.iterdir()
        if child.is_dir() and (child / "adapter_config.json").is_file() and child.name != "default"
    )
    for adapter_name in child_adapters:
        adapted.load_adapter(str(path / adapter_name), adapter_name=adapter_name)
    adapter_names = ["default", *child_adapters]
    adapted.base_model.set_adapter(adapter_names)
    if list(adapted.peft_config) != adapter_names:
        raise RuntimeError(
            f"serialized adapters do not match expected names: {list(adapted.peft_config)} vs {adapter_names}"
        )
    return adapted, adapter_names


def load_peft_for_inference(
    model,
    adapter_path: str | os.PathLike[str],
    *,
    merge: bool = True,
):
    """Load a saved adapter layout, optionally merging all named adapters.

    ``merge=True`` preserves the historical scorer behavior.  Phase 6B
    multi-adapter runs can pass ``merge=False`` because the installed PEFT
    merge path changes Qwen3-VL bfloat16 logits even though the serialized,
    unmerged model is exact.
    """

    adapted, adapter_names = load_peft_adapters(model, adapter_path)
    if merge:
        return adapted.merge_and_unload(adapter_names=adapter_names)
    return adapted


def letter_ids(tokenizer) -> list[int]:
    tokenised = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in tokenised):
        raise ValueError(f"A-D must each encode to one token, got {tokenised}")
    ids = [values[0] for values in tokenised]
    if not all(isinstance(i, int) and i >= 0 for i in ids) or len(set(ids)) != 4:
        raise ValueError(f"A-D must be four distinct tokenizer ids, got {ids}")
    return ids


def normalise_permutation(q: dict[str, Any], permutation: Iterable[str]) -> list[str]:
    perm = list(permutation)
    if sorted(perm) != sorted(LETTERS):
        raise ValueError(f"permutation must contain A-D exactly once, got {perm}")
    pinned = [
        letter
        for letter in LETTERS
        if q["options"][letter].strip().lower().startswith(("all of the above", "none of the above"))
    ]
    if pinned:
        perm = [letter for letter in perm if letter not in pinned] + pinned
    return perm


def permute_options(q: dict[str, Any], permutation: Iterable[str]):
    """Return shown options and the original letter at each shown position."""

    perm = normalise_permutation(q, permutation)
    shown = {letter: q["options"][perm[index]] for index, letter in enumerate(LETTERS)}
    return shown, perm


def user_text(q: dict[str, Any], shown: dict[str, str]) -> str:
    options = "\n".join(f"{letter}. {shown[letter]}" for letter in LETTERS)
    return (
        f"Task: {q['category']} / {q['task']}\n"
        f"Question: {q['question']}\n"
        f"Options:\n{options}\n"
        "Answer with the option's letter from the given choices directly."
    )


def _processor_call(processor, text: str, frames: Any, meta: Any):
    videos_kwargs = {
        "do_sample_frames": False,
        "size": VIDEO_SIZE,
        "cap_pixels_per_frame": True,
        "video_metadata": meta,
    }
    optional = ("cap_pixels_per_frame", "do_sample_frames", "size")
    last_error = None
    for removed_count in range(len(optional) + 1):
        call_kwargs = dict(videos_kwargs)
        for name in optional[:removed_count]:
            call_kwargs.pop(name, None)
        try:
            return processor(
                text=[text],
                videos=[frames],
                return_tensors="pt",
                videos_kwargs=call_kwargs,
            )
        except TypeError as exc:
            last_error = exc
            message = str(exc)
            if not any(name in message for name in (*optional, "video_metadata", "processor_kwargs", "videos_kwargs")):
                raise
            if "video_metadata" in message:
                break
    if last_error is not None:
        raise last_error
    raise RuntimeError("processor call did not execute")


def encode(
    processor,
    q: dict[str, Any],
    frames: Any,
    meta: Any,
    permutation: Iterable[str],
    answer_letter_orig: str | None = None,
    im_end_id: int | None = None,
    ltr_ids: list[int] | None = None,
    grounding_context: str | None = None,
    segmentation_context: str | None = None,
    temporal_context: str | None = None,
    enable_thinking: bool | None = None,
    temporal_token_ids: Iterable[int] | None = None,
    temporal_supervision: bool | None = None,
):
    """Build the exact train/inference encoding and optional answer/time labels.

    ``temporal_token_ids`` is opt-in.  When omitted, the historical one-token
    A-D target and tensor layout are unchanged.
    """

    import torch

    shown, perm = permute_options(q, permutation)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": user_text(q, shown)},
                *(
                    [{"type": "text", "text": grounding_context}]
                    if grounding_context
                    else []
                ),
                *(
                    [{"type": "text", "text": segmentation_context}]
                    if segmentation_context
                    else []
                ),
                *(
                    [{"type": "text", "text": temporal_context}]
                    if temporal_context
                    else []
                ),
            ],
        },
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = bool(enable_thinking)
    text = processor.apply_chat_template(messages, **chat_kwargs)
    raw = _processor_call(processor, text, frames, meta)
    allowed = {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    encoded = {key: value for key, value in raw.items() if key in allowed}
    required = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw"}
    missing = required - encoded.keys()
    if missing:
        raise KeyError(f"processor did not return required model inputs: {sorted(missing)}")
    if answer_letter_orig is not None:
        if answer_letter_orig not in LETTERS or im_end_id is None or ltr_ids is None:
            raise ValueError("training labels require answer letter, im_end_id, and ltr_ids")
        target = ltr_ids[perm.index(answer_letter_orig)]
        temporal_ids = list(temporal_token_ids or ())
        input_ids = encoded["input_ids"][0]
        suffix = torch.tensor((*temporal_ids, target, im_end_id), dtype=input_ids.dtype)
        ids = torch.cat((input_ids, suffix))
        labels = torch.full_like(ids, -100)
        temporal_label_mask = torch.zeros_like(labels, dtype=torch.bool)
        if temporal_ids:
            labels[-2 - len(temporal_ids) : -2] = torch.tensor(temporal_ids, dtype=labels.dtype)
            temporal_label_mask[-2 - len(temporal_ids) : -2] = True
        labels[-2] = target
        encoded["input_ids"] = ids.unsqueeze(0)
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        text_types = torch.zeros((1, len(suffix)), dtype=encoded["mm_token_type_ids"].dtype)
        encoded["mm_token_type_ids"] = torch.cat((encoded["mm_token_type_ids"], text_types), dim=1)
        encoded["labels"] = labels.unsqueeze(0)
        if temporal_token_ids is not None or temporal_supervision is not None:
            encoded["temporal_label_mask"] = temporal_label_mask.unsqueeze(0)
            encoded["temporal_supervision_mask"] = torch.tensor(
                bool(temporal_supervision) if temporal_supervision is not None else bool(temporal_ids),
                dtype=torch.bool,
            )
    return encoded, perm
