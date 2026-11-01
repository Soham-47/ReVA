"""Opt-in temporal grounding helpers for the ReVA Qwen3-VL branch.

The normal ReVA path does not import or call this module.  The branch in
``train.py`` enables it only when ``temporal_stream.enabled`` is true.

The timestamp contract deliberately accepts only unambiguous point and
interval expressions in the *correct answer option*.  The returned bins are
relative to the decoded video duration, with ``<t000>`` and ``<t300>`` being
the two endpoints.  A point target is ``<tNNN>``; an interval target is
``<t_start> <tNNN> <t_end> <tMMM>``.  The time prefix is followed by the
existing permuted A-D token and ``<|im_end|>``.

InternVideo2 is an optional local dependency.  ``load_internvideo2_encoder``
uses a caller-provided factory and a local checkpoint; it never downloads a
model.  ``DualStreamQwen3VL`` fuses projected segment features into Qwen's
precomputed video features through the official ``mm_encoder_outputs`` hook.
This keeps the appearance/video path intact and makes incompatibility with an
older pinned Transformers implementation explicit.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # Keep parser and token-contract checks usable on login nodes.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - training environments provide torch.
    torch = None
    nn = None
    F = None


NUM_BINS = 300
TEMPORAL_START_TOKEN = "<t_start>"
TEMPORAL_END_TOKEN = "<t_end>"
TEMPORAL_BINS = tuple(f"<t{index:03d}>" for index in range(NUM_BINS + 1))
TEMPORAL_TOKENS = (TEMPORAL_START_TOKEN, TEMPORAL_END_TOKEN, *TEMPORAL_BINS)
TEMPORAL_BUNDLE_VERSION = 1

_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = r"(?:seconds?|secs?|sec\.?|s)"
_INTERVAL_PATTERNS = (
    re.compile(
        rf"\b(?:between|from)\s*(?P<start>{_NUMBER})\s*(?:{_UNIT}\s*)?"
        rf"(?:to|and|[-–—])\s*(?P<end>{_NUMBER})\s*(?:{_UNIT})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<start>{_NUMBER})\s*{_UNIT}\s*(?:to|[-–—])\s*"
        rf"(?P<end>{_NUMBER})\s*{_UNIT}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:appears?|starts?|visible)\s+(?:at\s+)?(?P<start>{_NUMBER})\s*{_UNIT}\s*"
        rf"(?:and|,)\s*(?:disappears?|ends?|stops?)\s+(?:at\s+)?"
        rf"(?P<end>{_NUMBER})\s*{_UNIT}\b",
        re.IGNORECASE,
    ),
)
_POINT_PATTERNS = (
    re.compile(
        rf"\b(?:at|around|about|approximately|approx\.?|near)\s*(?P<point>{_NUMBER})\s*{_UNIT}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the\s+)?(?:point|timestamp|time)\s*(?:is|at|of)?\s*"
        rf"(?P<point>{_NUMBER})\s*{_UNIT}\b",
        re.IGNORECASE,
    ),
)
_UNIT_VALUE = re.compile(rf"(?P<value>{_NUMBER})\s*{_UNIT}\b", re.IGNORECASE)
_CLOCK_VALUE = re.compile(r"(?P<minutes>\d{1,3}):(?P<seconds>\d{1,2}(?:\.\d+)?)")
_CLOCK_INTERVAL = re.compile(
    r"\b(?:between|from)\s*(?P<start_m>\d{1,3}):(?P<start_s>\d{1,2}(?:\.\d+)?)\s*"
    r"(?:to|and|[-–—])\s*(?P<end_m>\d{1,3}):(?P<end_s>\d{1,2}(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_TIME_DELIMITER = re.compile(r"<t_(?:start|end)>")


@dataclass(frozen=True)
class TemporalAnnotation:
    """A validated absolute-time annotation in seconds."""

    kind: str
    start: float
    end: float
    duration: float

    @property
    def is_interval(self) -> bool:
        return self.kind == "interval"


@dataclass(frozen=True)
class TemporalVocabulary:
    """Tokenizer ids for the temporal token contract."""

    token_names: tuple[str, ...]
    token_ids: tuple[int, ...]
    tokenizer_vocab_size: int
    num_bins: int = NUM_BINS

    def __post_init__(self) -> None:
        if len(self.token_names) != len(self.token_ids):
            raise ValueError("temporal token names and ids must have equal length")
        if len(self.token_names) != self.num_bins + 3:
            raise ValueError("temporal vocabulary must contain two delimiters and every bin")
        if len(set(self.token_ids)) != len(self.token_ids) or any(i < 0 for i in self.token_ids):
            raise ValueError("temporal token ids must be distinct non-negative integers")
        if self.tokenizer_vocab_size <= max(self.token_ids):
            raise ValueError("temporal token ids must fit inside tokenizer vocabulary")

    @property
    def start_id(self) -> int:
        return self.token_ids[0]

    @property
    def end_id(self) -> int:
        return self.token_ids[1]

    @property
    def bin_ids(self) -> tuple[int, ...]:
        return self.token_ids[2:]

    def bin_id(self, index: int) -> int:
        if not 0 <= int(index) <= self.num_bins:
            raise ValueError(f"temporal bin must be in [0, {self.num_bins}], got {index}")
        return self.bin_ids[int(index)]

    def token_name(self, index: int) -> str:
        if not 0 <= int(index) <= self.num_bins:
            raise ValueError(f"temporal bin must be in [0, {self.num_bins}], got {index}")
        return TEMPORAL_BINS[int(index)]


def _as_positive_duration(duration: Any) -> float | None:
    try:
        value = float(duration)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _relative_bin(seconds: float, duration: float, num_bins: int) -> int:
    ratio = min(1.0, max(0.0, float(seconds) / duration))
    return min(num_bins, max(0, int(round(ratio * num_bins))))


def parse_temporal_option(
    text: Any,
    duration: Any,
    *,
    num_bins: int = NUM_BINS,
) -> TemporalAnnotation | None:
    """Parse one correct option, returning ``None`` for ambiguous prose.

    This parser intentionally requires explicit time units and rejects common
    duration-only phrases (``for 3 seconds``, ``lasts 3 seconds``).  It never
    attempts to infer a timestamp from a generic number in an answer option.
    """

    if not isinstance(text, str) or not isinstance(num_bins, int) or num_bins < 1:
        return None
    valid_duration = _as_positive_duration(duration)
    if valid_duration is None:
        return None
    value = " ".join(text.strip().split())
    if not value:
        return None
    # Options often append the clip duration as metadata.  It is not a second
    # target and should not mask an otherwise explicit interval.
    value = re.sub(r"\s*\(\s*duration\s*:[^)]*\)\s*$", "", value, flags=re.IGNORECASE)
    lower = value.casefold()
    duration_only = re.search(
        r"\b(?:for|lasting|lasts|duration|duration\s+of|within|over)\b|\bseconds?\s+long\b",
        lower,
    )

    intervals = []
    for pattern in _INTERVAL_PATTERNS:
        intervals.extend(pattern.finditer(value))
    if intervals and len({(match.group("start"), match.group("end")) for match in intervals}) == 1:
        # The two permissive patterns can both match "from 1 seconds to 2
        # seconds"; treat that as one annotation, not an ambiguity.
        intervals = [intervals[0]]
    if intervals:
        # Multiple interval matches or duration language makes the label unsafe.
        if len(intervals) != 1 or duration_only:
            return None
        match = intervals[0]
        start = float(match.group("start"))
        end = float(match.group("end"))
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end <= valid_duration):
            return None
        return TemporalAnnotation("interval", start, end, valid_duration)
    clock_interval = _CLOCK_INTERVAL.search(value)
    if clock_interval is not None:
        start = float(clock_interval.group("start_m")) * 60.0 + float(clock_interval.group("start_s"))
        end = float(clock_interval.group("end_m")) * 60.0 + float(clock_interval.group("end_s"))
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end <= valid_duration):
            return None
        return TemporalAnnotation("interval", start, end, valid_duration)

    points = []
    for pattern in _POINT_PATTERNS:
        points.extend(pattern.finditer(value))
    unit_values = list(_UNIT_VALUE.finditer(value))
    clock_values = list(_CLOCK_VALUE.finditer(value))
    bare_unit = re.fullmatch(rf"\s*({_NUMBER})\s*{_UNIT}\s*\.?\s*", value, flags=re.IGNORECASE)
    bare_clock = re.fullmatch(r"\s*(\d{1,3}):(\d{1,2}(?:\.\d+)?)\s*", value)
    if len(points) == 1 and len(unit_values) == 1 and not duration_only:
        point = float(points[0].group("point"))
    elif bare_unit and not duration_only:
        point = float(bare_unit.group(1))
    elif bare_clock and not duration_only:
        point = float(bare_clock.group(1)) * 60.0 + float(bare_clock.group(2))
    elif len(clock_values) == 1 and re.search(r"\b(?:at|around|about|approximately|approx\.?|near)\b", lower):
        point = float(clock_values[0].group("minutes")) * 60.0 + float(clock_values[0].group("seconds"))
    else:
        return None
    if not (math.isfinite(point) and 0.0 <= point <= valid_duration):
        return None
    return TemporalAnnotation("point", point, point, valid_duration)


def annotation_to_bins(annotation: TemporalAnnotation, *, num_bins: int = NUM_BINS) -> tuple[int, ...]:
    """Convert a validated annotation to one point or two interval bins."""

    if annotation is None:
        raise ValueError("annotation is required")
    start = _relative_bin(annotation.start, annotation.duration, num_bins)
    if annotation.is_interval:
        return start, _relative_bin(annotation.end, annotation.duration, num_bins)
    return (start,)


def serialize_temporal_target(
    annotation: TemporalAnnotation,
    vocabulary: TemporalVocabulary,
) -> tuple[str, ...]:
    """Return the ordered token strings for a temporal answer prefix."""

    bins = annotation_to_bins(annotation, num_bins=vocabulary.num_bins)
    if len(bins) == 1:
        return (vocabulary.token_name(bins[0]),)
    return (
        vocabulary.token_names[0],
        vocabulary.token_name(bins[0]),
        vocabulary.token_names[1],
        vocabulary.token_name(bins[1]),
    )


def token_ids_for_annotation(
    annotation: TemporalAnnotation,
    vocabulary: TemporalVocabulary,
) -> tuple[int, ...]:
    names = serialize_temporal_target(annotation, vocabulary)
    by_name = dict(zip(vocabulary.token_names, vocabulary.token_ids))
    return tuple(by_name[name] for name in names)


def build_temporal_vocabulary(tokenizer: Any, *, num_bins: int = NUM_BINS) -> TemporalVocabulary:
    """Add the temporal token contract to a tokenizer and verify single ids."""

    if num_bins != NUM_BINS:
        # The names are intentionally fixed to avoid silently incompatible bundles.
        raise ValueError(f"this branch defines exactly {NUM_BINS} relative bins")
    existing_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    missing = [
        token
        for token in TEMPORAL_TOKENS
        if (token not in existing_vocab if existing_vocab else tokenizer.convert_tokens_to_ids(token) in {None, -1})
    ]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    ids = []
    for token in TEMPORAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"tokenizer did not register temporal token {token!r}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != token_id:
            raise ValueError(f"temporal token {token!r} must encode to one stable token, got {encoded}")
        ids.append(token_id)
    return TemporalVocabulary(tuple(TEMPORAL_TOKENS), tuple(ids), len(tokenizer), num_bins)


def ensure_temporal_tokens(model: Any, tokenizer: Any, *, num_bins: int = NUM_BINS) -> TemporalVocabulary:
    """Resize Qwen once, preserving its tied input/output embedding contract."""

    vocabulary = build_temporal_vocabulary(tokenizer, num_bins=num_bins)
    resize = getattr(model, "resize_token_embeddings", None)
    if resize is None:
        raise RuntimeError("the configured Qwen model cannot resize token embeddings")
    input_embeddings = model.get_input_embeddings()
    if int(input_embeddings.num_embeddings) != vocabulary.tokenizer_vocab_size:
        resize(vocabulary.tokenizer_vocab_size)
        input_embeddings = model.get_input_embeddings()
    if int(input_embeddings.num_embeddings) != vocabulary.tokenizer_vocab_size:
        raise RuntimeError("resized Qwen input embeddings disagree with tokenizer vocabulary")
    output_embeddings = getattr(model, "get_output_embeddings", lambda: None)()
    if output_embeddings is not None and hasattr(output_embeddings, "out_features"):
        if int(output_embeddings.out_features) != vocabulary.tokenizer_vocab_size:
            raise RuntimeError("resized Qwen lm_head disagrees with tokenizer vocabulary")
    return vocabulary


class TemporalTargetContract:
    """Train-only conversion from a QA row to temporal target ids."""

    def __init__(self, vocabulary: TemporalVocabulary):
        self.vocabulary = vocabulary

    def for_row(self, row: Mapping[str, Any], duration: Any) -> tuple[int, ...] | None:
        if not str(row.get("qa_id", "")).startswith("train_"):
            raise ValueError("temporal targets may only be derived from train_* rows")
        if row.get("task") != "Temporal Grounding":
            return None
        options = row.get("options")
        answer = row.get("correct_answer")
        if not isinstance(options, Mapping) or answer not in options:
            return None
        annotation = parse_temporal_option(options[answer], duration, num_bins=self.vocabulary.num_bins)
        return None if annotation is None else token_ids_for_annotation(annotation, self.vocabulary)


def _require_torch() -> tuple[Any, Any, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("temporal stream requires the training PyTorch environment")
    return torch, nn, F


def preprocess_temporal_frames(
    frames: Any,
    *,
    num_segments: int = 12,
    frames_per_segment: int = 8,
    image_size: int = 224,
) -> Any:
    """Resize/normalize dense RGB frames into ``[segments, frames, C, H, W]``.

    The preprocessing matches the official InternVideo2 demo (ImageNet mean /
    std and 224px frames) but deliberately uses only PyTorch, avoiding a
    hidden torchvision dependency in DataLoader workers.
    """

    torch_module, _, functional = _require_torch()
    if num_segments < 1 or frames_per_segment < 1 or image_size < 1:
        raise ValueError("temporal frame dimensions must be positive")
    tensor = torch_module.as_tensor(frames)
    if tensor.ndim != 4 or tensor.shape[-1] != 3:
        raise ValueError(f"frames must have shape [T,H,W,3], got {tuple(tensor.shape)}")
    tensor = tensor.float()
    total = num_segments * frames_per_segment
    source_indices = torch_module.linspace(0, tensor.shape[0] - 1, total).round().long()
    tensor = tensor[source_indices].reshape(num_segments, frames_per_segment, *tensor.shape[1:])
    tensor = tensor.permute(0, 1, 4, 2, 3).reshape(total, 3, tensor.shape[2], tensor.shape[3]) / 255.0
    tensor = functional.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((tensor - mean) / std).reshape(num_segments, frames_per_segment, 3, image_size, image_size)


def load_internvideo2_encoder(config: Mapping[str, Any]) -> Any:
    """Build a frozen InternVideo2 encoder from a local checkpoint.

    ``config['builder']`` is a dotted ``module:function`` factory supplied by
    the runtime installation of the official OpenGVLab repository.  The
    factory receives ``builder_config`` and should return either the complete
    ``InternVideo2_Stage2`` object or its ``vision_encoder``.  Official
    checkpoints are state dictionaries (possibly under ``model`` or
    ``module``), so all three layouts are accepted.
    """

    checkpoint_path = Path(str(config.get("checkpoint", ""))).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "InternVideo2 checkpoint is not available locally; set temporal_stream.checkpoint "
            f"to the approved local file, got {checkpoint_path}"
        )
    factory_name = str(config.get("builder", "")).strip()
    if ":" not in factory_name:
        raise RuntimeError(
            "temporal_stream.builder must be a local module:function factory; "
            "the branch never downloads or guesses an InternVideo2 loader"
        )
    module_name, function_name = factory_name.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    encoder = factory(config.get("builder_config", {}))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("module", checkpoint)) if isinstance(checkpoint, Mapping) else checkpoint
    target = getattr(encoder, "vision_encoder", encoder)
    target_keys = set(target.state_dict())
    if isinstance(state_dict, Mapping) and not target_keys.intersection(state_dict):
        for prefix in ("module.vision_encoder.", "vision_encoder.", "module."):
            stripped = {str(key)[len(prefix) :]: value for key, value in state_dict.items() if str(key).startswith(prefix)}
            if target_keys.intersection(stripped):
                state_dict = stripped
                break
    if isinstance(state_dict, Mapping):
        # The published ``...f4.pt`` stage-2 checkpoint carries four-frame
        # temporal position embeddings, while the ReVA segment contract uses
        # eight frames. OpenGVLab ships the interpolation helper for this
        # exact checkpoint-to-runtime transition; use it after prefix
        # normalization and before ``load_state_dict``.
        try:
            from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
        except ImportError:
            interpolate_pos_embed_internvideo2_new = None
        if interpolate_pos_embed_internvideo2_new is not None and any(
            "pos_embed" in str(key) or "clip_pos_embed" in str(key) for key in state_dict
        ):
            interpolate_pos_embed_internvideo2_new(
                state_dict,
                target,
                orig_t_size=int(config.get("checkpoint_num_frames", 4)),
            )
    target.load_state_dict(state_dict, strict=False)
    if isinstance(state_dict, Mapping) and not set(state_dict).intersection(target_keys):
        raise RuntimeError("InternVideo2 checkpoint did not match the configured encoder")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder


class _AttributeConfig(dict):
    """Small adapter for the official InternVideo2 ``config.vision_encoder`` API."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def build_official_internvideo2(builder_config: Mapping[str, Any] | None = None) -> Any:
    """Build the official 1B vision backbone without fetching its checkpoint.

    This factory mirrors OpenGVLab's ``pretrain_internvideo2_1b_patch14_224``
    constructor.  ``load_internvideo2_encoder`` performs the local checkpoint
    load afterward, so the checkpoint is never obtained implicitly.
    """

    try:
        from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
    except ImportError as exc:
        raise RuntimeError(
            "official OpenGVLab InternVideo2 code is required for the temporal branch"
        ) from exc
    values = dict(builder_config or {})
    nested = values.get("vision_encoder", values)
    defaults = {
        "clip_embed_dim": 768,
        "use_flash_attn": False,
        "use_fused_rmsnorm": False,
        "use_fused_mlp": False,
        "num_frames": 8,
        "tubelet_size": 1,
        "sep_image_video_pos_embed": True,
        "use_checkpoint": False,
        "checkpoint_num": 0,
        "clip_teacher_embed_dim": 3200,
        "clip_teacher_final_dim": 768,
        "clip_norm_type": "l2",
        "clip_return_layer": 1,
        "clip_student_return_interval": 1,
        "pretrained": None,
    }
    merged = {**defaults, **dict(nested)}
    # The official factory receives the ``config.vision_encoder`` node, not
    # the node itself. Passing the nested shape also keeps its ``.get`` and
    # attribute access semantics intact without importing its config class.
    return pretrain_internvideo2_1b_patch14_224(
        _AttributeConfig({"vision_encoder": _AttributeConfig(merged)})
    )


def infer_temporal_feature_dim(encoder: Any, expected: int | None = None) -> int:
    """Read the pooled InternVideo2 width from the constructed encoder.

    The official 1B backbone exposes this as the attention-pool projection's
    ``out_features`` (768 for the referenced checkpoint).  We inspect the live
    module rather than silently assuming that width in the projector config.
    """

    candidates = [encoder, getattr(encoder, "vision_encoder", None)]
    for candidate in candidates:
        pooler = getattr(candidate, "clip_projector", None)
        projection = getattr(getattr(pooler, "cross_attn", None), "proj", None)
        width = getattr(projection, "out_features", None)
        if width is not None:
            actual = int(width)
            if expected is not None and actual != int(expected):
                raise ValueError(f"InternVideo2 feature width {actual} disagrees with configured {expected}")
            return actual
    raise RuntimeError("could not infer pooled InternVideo2 feature width from the loaded encoder")


class InternVideo2TemporalStream(nn.Module if nn is not None else object):
    """Frozen InternVideo2 segments plus a trainable Qwen-space projector."""

    def __init__(
        self,
        encoder: Any,
        *,
        feature_dim: int,
        hidden_size: int,
        num_segments: int = 12,
        frames_per_segment: int = 8,
        projector_hidden: int | None = None,
    ):
        _require_torch()
        super().__init__()
        if feature_dim < 1 or hidden_size < 1 or num_segments < 1 or frames_per_segment < 1:
            raise ValueError("temporal stream dimensions must be positive")
        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.num_segments = int(num_segments)
        self.frames_per_segment = int(frames_per_segment)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        intermediate = int(projector_hidden or max(hidden_size, feature_dim // 2))
        self.projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_size),
        )

    def _encode_segments(self, segments: Any) -> Any:
        batch, count, frames = segments.shape[:3]
        flat = segments.reshape(batch * count, frames, *segments.shape[3:])
        with torch.no_grad():
            if hasattr(self.encoder, "encode_vision"):
                result = self.encoder.encode_vision(flat, test=True)
                features = result[1] if isinstance(result, (tuple, list)) else result
            elif hasattr(self.encoder, "vision_encoder"):
                # The official backbone consumes [B,C,T,H,W].
                result = self.encoder.vision_encoder(flat.permute(0, 2, 1, 3, 4), None, True)
                features = result[1] if isinstance(result, (tuple, list)) else result
            else:
                # ``pretrain_internvideo2_1b_patch14_224`` returns the bare
                # official vision encoder, whose forward API consumes
                # [B,C,T,H,W] and returns (tokens, pooled, ...). Keep a
                # one-argument fallback for third-party local builders.
                channels_first = flat.permute(0, 2, 1, 3, 4)
                try:
                    result = self.encoder(channels_first, None, False)
                except TypeError:
                    try:
                        result = self.encoder(channels_first)
                    except TypeError:
                        result = self.encoder(flat)
                features = result[1] if isinstance(result, (tuple, list)) else result
        if features.ndim == 3:
            features = features[:, 0]
        if features.ndim != 2:
            raise RuntimeError(f"InternVideo2 must return pooled [B,D] features, got {tuple(features.shape)}")
        if int(features.shape[-1]) != self.feature_dim:
            raise RuntimeError(
                f"InternVideo2 returned feature width {features.shape[-1]}, expected {self.feature_dim}"
            )
        return features.reshape(batch, count, features.shape[-1])

    def forward(self, temporal_pixel_values: Any, temporal_segment_mask: Any | None = None) -> Any:
        if temporal_pixel_values.ndim == 5:
            if temporal_pixel_values.shape[1] != self.num_segments * self.frames_per_segment:
                raise ValueError(
                    "flat temporal input must contain num_segments * frames_per_segment frames"
                )
            temporal_pixel_values = temporal_pixel_values.reshape(
                temporal_pixel_values.shape[0],
                self.num_segments,
                self.frames_per_segment,
                *temporal_pixel_values.shape[2:],
            )
        if temporal_pixel_values.ndim != 6:
            raise ValueError("temporal_pixel_values must have shape [B,S,T,C,H,W]")
        features = self._encode_segments(temporal_pixel_values)
        projected = self.projector(features)
        if temporal_segment_mask is not None:
            mask = temporal_segment_mask.to(projected.device, dtype=projected.dtype).unsqueeze(-1)
            projected = projected * mask
        return projected


def _fuse_video_pooler(pooler: Any, motion: Any, video_grid_thw: Any) -> Any:
    """Broadcast segment features over Qwen temporal/spatial video tokens."""

    if isinstance(pooler, (list, tuple)):
        pools = list(pooler)
        output_type = type(pooler)
    elif pooler.ndim == 3:
        pools = [pooler[index] for index in range(pooler.shape[0])]
        output_type = torch.stack
    else:
        raise RuntimeError(f"unsupported Qwen video pooler shape {tuple(pooler.shape)}")
    grids = video_grid_thw.detach().cpu().tolist()
    if len(pools) != len(grids):
        raise RuntimeError(f"Qwen video pooler count {len(pools)} disagrees with grids {len(grids)}")
    fused = []
    for index, (pool, grid) in enumerate(zip(pools, grids)):
        temporal_tokens = int(grid[0])
        if temporal_tokens < 1 or pool.shape[0] % temporal_tokens:
            raise RuntimeError("Qwen video feature count is not divisible by temporal grid")
        spatial_tokens = pool.shape[0] // temporal_tokens
        segment = motion[min(index, motion.shape[0] - 1)].to(pool.device, pool.dtype)
        segment = F.interpolate(
            segment.transpose(0, 1).unsqueeze(0),
            size=temporal_tokens,
            mode="linear",
            align_corners=False,
        ).squeeze(0).transpose(0, 1)
        segment = segment.repeat_interleave(spatial_tokens, dim=0)
        fused.append(pool + segment)
    if output_type is torch.stack:
        return torch.stack(fused, dim=0)
    return output_type(fused)


class TemporalEmbedding(nn.Module if nn is not None else object):
    """Qwen input embedding that replaces only temporal ids with a trainable table."""

    def __init__(self, base_embedding: Any, token_ids: Iterable[int], table: Any):
        _require_torch()
        super().__init__()
        self.base_embedding = base_embedding
        self.table = table
        ids = torch.tensor(tuple(int(i) for i in token_ids), dtype=torch.long)
        mapping = torch.full((int(base_embedding.num_embeddings),), -1, dtype=torch.long)
        mapping[ids] = torch.arange(ids.numel(), dtype=torch.long)
        self.register_buffer("token_ids", ids, persistent=True)
        self.register_buffer("id_to_index", mapping, persistent=True)

    @property
    def weight(self) -> Any:
        return self.base_embedding.weight

    @property
    def num_embeddings(self) -> int:
        return self.base_embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base_embedding.embedding_dim

    def forward(self, input_ids: Any) -> Any:
        result = self.base_embedding(input_ids)
        indices = self.id_to_index[input_ids]
        mask = indices >= 0
        safe_indices = indices.clamp_min(0)
        temporal = self.table[safe_indices]
        return torch.where(mask.unsqueeze(-1), temporal, result)


class TemporalOutputHead(nn.Module if nn is not None else object):
    """Qwen lm head with temporal columns tied to the trainable table."""

    def __init__(self, base_head: Any, token_ids: Iterable[int], table: Any):
        _require_torch()
        super().__init__()
        self.base_head = base_head
        self.table = table
        self.register_buffer("token_ids", torch.tensor(tuple(token_ids), dtype=torch.long), persistent=True)

    @property
    def weight(self) -> Any:
        return self.base_head.weight

    @property
    def out_features(self) -> int:
        return self.base_head.out_features

    @property
    def in_features(self) -> int:
        return self.base_head.in_features

    def forward(self, hidden_states: Any) -> Any:
        logits = self.base_head(hidden_states).clone()
        temporal_logits = F.linear(hidden_states, self.table)
        logits[..., self.token_ids] = temporal_logits
        return logits


class DualStreamQwen3VL(nn.Module if nn is not None else object):
    """Qwen3-VL wrapper with a frozen motion stream and temporal token table."""

    def __init__(
        self,
        qwen_model: Any,
        temporal_stream: InternVideo2TemporalStream,
        vocabulary: TemporalVocabulary,
        tokenizer: Any | None = None,
        runtime_config: Mapping[str, Any] | None = None,
    ):
        _require_torch()
        super().__init__()
        self.qwen = qwen_model
        self.temporal_stream = temporal_stream
        self.temporal_vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.temporal_runtime_config = dict(runtime_config or {})
        # Checkpoint identity is immutable for a training run. Cache the
        # digest so Trainer checkpoints do not repeatedly hash a multi-GB
        # frozen InternVideo2 file.
        self._temporal_checkpoint_identity: dict[str, Any] | None = None
        base_input = qwen_model.get_input_embeddings()
        base_output = qwen_model.get_output_embeddings()
        if base_output is None:
            raise RuntimeError("Qwen model has no output embedding/lm_head")
        with torch.no_grad():
            initial = base_input.weight[torch.tensor(vocabulary.token_ids, dtype=torch.long)].detach().clone()
        self.temporal_token_table = nn.Parameter(initial)
        qwen_model.set_input_embeddings(TemporalEmbedding(base_input, vocabulary.token_ids, self.temporal_token_table))
        qwen_model.set_output_embeddings(TemporalOutputHead(base_output, vocabulary.token_ids, self.temporal_token_table))

    @property
    def config(self) -> Any:
        return self.qwen.config

    def train(self, mode: bool = True) -> "DualStreamQwen3VL":
        """Toggle Qwen/projector modes while keeping InternVideo2 frozen/eval."""

        super().train(mode)
        # Trainer calls ``model.train()`` at the start of every epoch.  That
        # recursive call would otherwise put the frozen motion encoder back
        # into training mode (e.g. enabling dropout), despite all parameters
        # being frozen.
        self.temporal_stream.encoder.eval()
        return self

    @property
    def generation_config(self) -> Any:
        return getattr(self.qwen, "generation_config", None)

    def get_input_embeddings(self) -> Any:
        return self.qwen.get_input_embeddings()

    def get_output_embeddings(self) -> Any:
        return self.qwen.get_output_embeddings()

    def enable_input_require_grads(self) -> Any:
        return self.qwen.enable_input_require_grads()

    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> Any:
        return self.qwen.gradient_checkpointing_enable(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> Any:
        return self.qwen.prepare_inputs_for_generation(*args, **kwargs)

    def _fused_qwen_kwargs(
        self,
        kwargs: dict[str, Any],
        temporal_pixel_values: Any,
        temporal_segment_mask: Any | None,
    ) -> dict[str, Any]:
        if "pixel_values_videos" not in kwargs or "video_grid_thw" not in kwargs:
            raise ValueError("dual-stream generation requires Qwen pixel_values_videos and video_grid_thw")
        if not self._supports_precomputed_video():
            raise RuntimeError(
                "the pinned Qwen3-VL forward does not expose mm_encoder_outputs; "
                "cannot safely fuse InternVideo2 features"
            )
        motion = self.temporal_stream(temporal_pixel_values, temporal_segment_mask)
        video_output = self._qwen_video_features(kwargs)
        if not hasattr(video_output, "pooler_output"):
            raise RuntimeError("Qwen video feature output has no pooler_output")
        fused = copy.copy(video_output)
        fused.pooler_output = _fuse_video_pooler(
            video_output.pooler_output,
            motion,
            kwargs["video_grid_thw"],
        )
        qwen_kwargs = dict(kwargs)
        qwen_kwargs.pop("pixel_values_videos", None)
        qwen_kwargs["mm_encoder_outputs"] = {"video": fused}
        return qwen_kwargs

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate with one fused video prefill and Qwen's normal KV cache.

        Temporal features are computed before ``generate`` and passed as
        precomputed video features, so the encoder is not rerun per decoded
        token.  The precomputed features are removed from model inputs after
        the first cache prefill; otherwise Qwen would see video features with
        a one-token cached input and fail its placeholder-count check.
        ``decode_time_prefix`` and ``parse_temporal_answer`` consume the
        resulting prefix before the regular A-D scorer is called.
        """

        temporal_pixel_values = kwargs.pop("temporal_pixel_values", None)
        temporal_segment_mask = kwargs.pop("temporal_segment_mask", None)
        if temporal_pixel_values is None:
            return self.qwen.generate(*args, **kwargs)
        if args:
            raise ValueError("dual-stream generation accepts keyword Qwen inputs only")
        fused_kwargs = self._fused_qwen_kwargs(kwargs, temporal_pixel_values, temporal_segment_mask)
        prepare = getattr(self.qwen, "prepare_inputs_for_generation", None)
        if prepare is None:
            raise RuntimeError("Qwen generation has no prepare_inputs_for_generation cache hook")
        first_prepare = True

        def clear_video_after_prefill(*prepare_args: Any, **prepare_kwargs: Any) -> Any:
            nonlocal first_prepare
            prepared = prepare(*prepare_args, **prepare_kwargs)
            if not isinstance(prepared, dict):
                raise RuntimeError(
                    "Qwen prepare_inputs_for_generation returned a non-dict; "
                    "cannot verify multimodal cache propagation"
                )
            if first_prepare and "mm_encoder_outputs" not in prepared:
                raise RuntimeError(
                    "Qwen prepare_inputs_for_generation dropped precomputed "
                    "mm_encoder_outputs before the multimodal prefill"
                )
            if not first_prepare:
                prepared.pop("mm_encoder_outputs", None)
                prepared.pop("pixel_values_videos", None)
            first_prepare = False
            return prepared

        try:
            # GenerationMixin calls this instance attribute on every cache
            # step.  The original bound method remains responsible for the
            # model-specific position/cache bookkeeping.
            setattr(self.qwen, "prepare_inputs_for_generation", clear_video_after_prefill)
            return self.qwen.generate(**fused_kwargs)
        finally:
            setattr(self.qwen, "prepare_inputs_for_generation", prepare)

    def _qwen_video_features(self, kwargs: dict[str, Any]) -> Any:
        candidates = [self.qwen]
        base_model = getattr(self.qwen, "base_model", None)
        if base_model is not None:
            candidates.append(base_model)
            nested = getattr(base_model, "model", None)
            if nested is not None:
                candidates.append(nested)
        nested = getattr(self.qwen, "model", None)
        if nested is not None:
            candidates.append(nested)
        getter = next((getattr(candidate, "get_video_features", None) for candidate in candidates if hasattr(candidate, "get_video_features")), None)
        if getter is None:
            raise RuntimeError("pinned Qwen3-VL model has no get_video_features precompute hook")
        try:
            return getter(kwargs["pixel_values_videos"], kwargs.get("video_grid_thw"), return_dict=True)
        except TypeError:
            return getter(kwargs["pixel_values_videos"], kwargs.get("video_grid_thw"))

    def _supports_precomputed_video(self) -> bool:
        candidates = [self.qwen]
        for attribute in ("base_model", "model"):
            candidate = getattr(self.qwen, attribute, None)
            if candidate is not None:
                candidates.append(candidate)
                nested = getattr(candidate, "model", None)
                if nested is not None:
                    candidates.append(nested)
        for candidate in candidates:
            try:
                signature = inspect.signature(candidate.forward)
            except (AttributeError, TypeError, ValueError):
                continue
            if "mm_encoder_outputs" in signature.parameters:
                return True
        return False

    def forward(
        self,
        *args: Any,
        temporal_pixel_values: Any | None = None,
        temporal_segment_mask: Any | None = None,
        temporal_supervision_mask: Any | None = None,
        temporal_label_mask: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        # The masks are metadata for bookkeeping; labels already carry -100
        # for rows without a valid temporal annotation.
        del temporal_supervision_mask, temporal_label_mask
        if temporal_pixel_values is None:
            return self.qwen(*args, **kwargs)
        qwen_kwargs = self._fused_qwen_kwargs(kwargs, temporal_pixel_values, temporal_segment_mask)
        if args:
            raise ValueError("dual-stream wrapper accepts keyword Qwen inputs only")
        return self.qwen(**qwen_kwargs)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.qwen.save_pretrained(path, **kwargs)
        save_temporal_bundle(self, path, self.tokenizer)

    @classmethod
    def from_temporal_bundle(
        cls,
        qwen_model: Any,
        temporal_stream: InternVideo2TemporalStream,
        tokenizer: Any,
        directory: str | Path,
        runtime_config: Mapping[str, Any] | None = None,
    ) -> "DualStreamQwen3VL":
        vocabulary = ensure_temporal_tokens(qwen_model, tokenizer)
        wrapper = cls(qwen_model, temporal_stream, vocabulary, tokenizer, runtime_config=runtime_config)
        load_temporal_bundle(wrapper, directory, tokenizer)
        return wrapper


def save_temporal_bundle(wrapper: DualStreamQwen3VL, directory: str | Path, tokenizer: Any | None = None) -> None:
    """Persist projector, temporal token rows, and exact vocabulary metadata."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    vocabulary = wrapper.temporal_vocabulary
    metadata = {
        "bundle_version": TEMPORAL_BUNDLE_VERSION,
        "token_names": list(vocabulary.token_names),
        "token_ids": list(vocabulary.token_ids),
        "tokenizer_vocab_size": vocabulary.tokenizer_vocab_size,
        "num_bins": vocabulary.num_bins,
        "temporal_runtime_config": json.loads(json.dumps(wrapper.temporal_runtime_config)),
    }
    checkpoint = wrapper.temporal_runtime_config.get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(str(checkpoint)).expanduser()
        identity = wrapper._temporal_checkpoint_identity
        if identity is None or identity.get("path") != str(checkpoint_path):
            identity = {"path": str(checkpoint_path)}
            if checkpoint_path.is_file():
                digest = hashlib.sha256()
                with checkpoint_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                identity["sha256"] = digest.hexdigest()
                identity["size"] = checkpoint_path.stat().st_size
            wrapper._temporal_checkpoint_identity = identity
        metadata["temporal_checkpoint_identity"] = identity
    if tokenizer is not None:
        current = build_temporal_vocabulary(tokenizer, num_bins=vocabulary.num_bins)
        if current != vocabulary:
            raise ValueError("tokenizer vocabulary changed before temporal bundle save")
    (path / "temporal_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.save(
        {
            "temporal_token_table": wrapper.temporal_token_table.detach().cpu(),
            # The frozen InternVideo2 checkpoint is an external runtime asset;
            # only the trainable segment projector belongs in the adapter bundle.
            "temporal_projector": {
                name: value.detach().cpu()
                for name, value in wrapper.temporal_stream.projector.state_dict().items()
            },
        },
        path / "temporal_state.pt",
    )


def load_temporal_bundle(wrapper: DualStreamQwen3VL, directory: str | Path, tokenizer: Any) -> None:
    """Load a temporal bundle and reject incompatible token/vocabulary layouts."""

    path = Path(directory)
    metadata_path = path / "temporal_config.json"
    state_path = path / "temporal_state.pt"
    if not metadata_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(f"temporal bundle is incomplete under {path}")
    metadata = json.loads(metadata_path.read_text())
    expected = wrapper.temporal_vocabulary
    actual = build_temporal_vocabulary(tokenizer, num_bins=expected.num_bins)
    if (
        metadata.get("bundle_version") != TEMPORAL_BUNDLE_VERSION
        or metadata.get("token_names") != list(expected.token_names)
        or metadata.get("token_ids") != list(expected.token_ids)
        or metadata.get("tokenizer_vocab_size") != expected.tokenizer_vocab_size
        or actual != expected
    ):
        raise ValueError("temporal checkpoint vocabulary is incompatible with the current tokenizer/model")
    state = torch.load(state_path, map_location="cpu")
    table = state.get("temporal_token_table")
    if table is None or tuple(table.shape) != tuple(wrapper.temporal_token_table.shape):
        raise ValueError("temporal token table shape is incompatible with the current model")
    with torch.no_grad():
        wrapper.temporal_token_table.copy_(table.to(wrapper.temporal_token_table.device))
    if "temporal_projector" not in state:
        raise ValueError("temporal checkpoint does not contain the trainable projector")
    wrapper.temporal_stream.projector.load_state_dict(state["temporal_projector"])


def decode_time_prefix(
    text: str,
    duration: Any,
    *,
    num_bins: int = NUM_BINS,
) -> tuple[float, float] | tuple[float] | None:
    """Decode generated ``<tNNN>`` tokens into absolute seconds."""

    valid_duration = _as_positive_duration(duration)
    if not isinstance(text, str) or valid_duration is None:
        return None
    if _TIME_DELIMITER.search(text):
        match = re.fullmatch(
            rf"\s*{re.escape(TEMPORAL_START_TOKEN)}\s*<t(\d{{3}})>\s*"
            rf"{re.escape(TEMPORAL_END_TOKEN)}\s*<t(\d{{3}})>\s*",
            text,
        )
        if match is None:
            return None
        tokens = match.groups()
    else:
        match = re.fullmatch(r"\s*<t(\d{3})>\s*", text)
        if match is None:
            return None
        tokens = match.groups()
    bins = tuple(int(item) for item in tokens)
    if any(item > num_bins for item in bins) or (len(bins) == 2 and bins[0] > bins[1]):
        return None
    seconds = tuple(valid_duration * item / num_bins for item in bins)
    return seconds


def parse_temporal_answer(
    text: str,
    duration: Any,
    *,
    num_bins: int = NUM_BINS,
) -> tuple[tuple[float, ...] | None, str | None]:
    """Parse a generated time prefix and the first standalone A-D answer."""

    if not isinstance(text, str):
        return None, None
    answer_match = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
    if answer_match is None:
        return decode_time_prefix(text, duration, num_bins=num_bins), None
    prefix = text[: answer_match.start()].strip()
    times = decode_time_prefix(prefix, duration, num_bins=num_bins)
    return times, answer_match.group(1)


def generate_temporal_answer(
    model: DualStreamQwen3VL,
    tokenizer: Any,
    duration: Any,
    **inputs: Any,
) -> tuple[tuple[float, ...] | None, str | None]:
    """Run the opt-in fused generation path, then parse time before A-D.

    Callers provide the normal Qwen processor tensors plus the dense
    ``temporal_pixel_values`` produced by :func:`preprocess_temporal_frames`.
    The returned answer letter can be handed to ReVA's existing scorer; the
    timestamp tuple is an auxiliary grounding prediction.
    """

    generated = model.generate(**inputs)
    if getattr(generated, "ndim", 0) == 1:
        generated = generated.unsqueeze(0)
    if getattr(generated, "ndim", 0) != 2 or generated.shape[0] < 1:
        raise ValueError("Qwen generation must return a non-empty [batch, tokens] tensor")
    prompt_ids = inputs.get("input_ids")
    prompt_length = int(prompt_ids.shape[-1]) if prompt_ids is not None else 0
    text = tokenizer.decode(generated[0, prompt_length:].tolist(), skip_special_tokens=False)
    return parse_temporal_answer(text, duration)

