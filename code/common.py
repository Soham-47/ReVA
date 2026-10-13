"""Shared paths, frame sampling, prompts, option mapping, and model encoding.

The module keeps optional GPU/video dependencies inside the functions that need
them so split and submission checks can run on a login node with only Python.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
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
    k = int(round(n / rate * FPS_TARGET))
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
):
    import numpy as np

    return np.asarray(sample_index_list(n_frames, fps, offset, mode=mode), dtype=np.int64)


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


def read_frames(relative_path: str, indices: Iterable[int], work: str | os.PathLike[str] | None = None):
    import av
    import numpy as np

    wanted = [int(i) for i in indices]
    if not wanted:
        raise ValueError("at least one frame index is required")
    wanted_set = set(wanted)
    got: dict[int, Any] = {}
    path = dataset_dir(work) / relative_path
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for number, frame in enumerate(container.decode(stream)):
            if number in wanted_set:
                got[number] = frame.to_ndarray(format="rgb24")
            if number >= max(wanted_set):
                break
    if not got:
        raise RuntimeError(f"could not decode any requested frame from {relative_path}")
    fallback = got[max(got)]
    return np.stack([got.get(index, fallback) for index in wanted])


def get_video(
    relative_path: str,
    offset: float = 0.0,
    work: str | os.PathLike[str] | None = None,
    *,
    mode: str = DEFAULT_FRAME_SAMPLING,
):
    manifest = load_manifest(work)
    try:
        info = manifest[relative_path]
    except KeyError as exc:
        raise KeyError(f"{relative_path!r} is absent from {manifest_path(work)}") from exc
    indices = sample_indices(info["n_frames"], info["fps"], offset, mode=mode)
    frames = read_frames(relative_path, indices, work)
    meta = _video_metadata(
        total_num_frames=int(info["n_frames"]),
        fps=float(info["fps"]),
        frames_indices=indices.tolist(),
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
    base = dict(
        text=[text],
        videos=[frames],
        video_metadata=[meta],
        do_sample_frames=False,
        size=VIDEO_SIZE,
        cap_pixels_per_frame=True,
        return_tensors="pt",
    )
    optional = ("cap_pixels_per_frame", "do_sample_frames", "size")
    last_error = None
    for removed_count in range(len(optional) + 1):
        kwargs = dict(base)
        for name in optional[:removed_count]:
            kwargs.pop(name, None)
        try:
            return processor(**kwargs)
        except TypeError as exc:
            last_error = exc
            message = str(exc)
            if not any(name in message for name in (*optional, "video_metadata")):
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
):
    """Build the exact train/inference encoding and optional one-token label."""

    import torch

    shown, perm = permute_options(q, permutation)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": user_text(q, shown)}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = _processor_call(processor, text, frames, meta)
    allowed = {
        "input_ids",
        "attention_mask",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    encoded = {key: value for key, value in raw.items() if key in allowed}
    required = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"}
    missing = required - encoded.keys()
    if missing:
        raise KeyError(f"processor did not return required model inputs: {sorted(missing)}")
    if answer_letter_orig is not None:
        if answer_letter_orig not in LETTERS or im_end_id is None or ltr_ids is None:
            raise ValueError("training labels require answer letter, im_end_id, and ltr_ids")
        target = ltr_ids[perm.index(answer_letter_orig)]
        input_ids = encoded["input_ids"][0]
        ids = torch.cat((input_ids, torch.tensor([target, im_end_id], dtype=input_ids.dtype)))
        labels = torch.full_like(ids, -100)
        labels[-2] = target
        encoded["input_ids"] = ids.unsqueeze(0)
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        encoded["labels"] = labels.unsqueeze(0)
    return encoded, perm
