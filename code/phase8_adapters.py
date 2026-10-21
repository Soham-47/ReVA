"""Model-family adapters for the isolated Phase 8 pretrained scorer.

The Stage A scorer intentionally remains the source of truth for the frozen
8B run.  This module only contains the loader and processor differences needed
by the two Phase 8 candidate backbones.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass, replace
from typing import Any, Iterable

import common as C


@dataclass(frozen=True)
class ModelSpec:
    """A reproducible model identity and its Transformers implementation."""

    family: str
    model_id: str
    loader: str
    non_thinking_chat: bool
    revision: str


_MODEL_SPECS = {
    "qwen38": ModelSpec(
        family="qwen38",
        model_id="Qwen/Qwen3.8-27B",
        loader="AutoModelForMultimodalLM",
        non_thinking_chat=True,
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    ),
    "qwen3_vl_32b": ModelSpec(
        family="qwen3_vl_32b",
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        loader="Qwen3VLForConditionalGeneration",
        non_thinking_chat=False,
        revision="0cfaf48183f594c314753d30a4c4974bc75f3ccb",
    ),
}

_ALIASES = {
    "qwen38": "qwen38",
    "qwen3.8": "qwen38",
    "qwen3.8-27b": "qwen38",
    "qwen3_8_27b": "qwen38",
    "qwen3-vl-32b": "qwen3_vl_32b",
    "qwen3-vl-32b-instruct": "qwen3_vl_32b",
    "qwen3vl32": "qwen3_vl_32b",
    "qwen3vl32b": "qwen3_vl_32b",
    "qwen3_vl_32b": "qwen3_vl_32b",
}


def resolve_model_spec(
    family: str,
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> ModelSpec:
    """Resolve a user-facing family alias without importing Transformers."""

    key = str(family).strip().casefold()
    try:
        spec = _MODEL_SPECS[_ALIASES[key]]
    except KeyError as exc:
        choices = ", ".join(sorted(_MODEL_SPECS))
        raise ValueError(f"unknown Phase 8 model family {family!r}; choose one of {choices}") from exc
    updates = {}
    if model_id is not None:
        if not str(model_id).strip():
            raise ValueError("model_id must be non-empty")
        updates["model_id"] = str(model_id)
    if revision is not None:
        if not str(revision).strip():
            raise ValueError("revision must be non-empty")
        updates["revision"] = str(revision)
    return replace(spec, **updates)


def validate_label_tokens(processor: Any) -> list[int]:
    """Require A-D to be four distinct one-token labels."""

    return C.letter_ids(processor.tokenizer)


def label_tokenization_telemetry(processor: Any) -> dict[str, Any]:
    """Record tokenizer IDs for plain and leading-space A-D forms.

    The scorer always uses the plain ``A``-through-``D`` IDs returned by
    :func:`validate_label_tokens`.  Leading-space forms are diagnostic only;
    some tokenizers do not expose them as one token, which is recorded as
    unavailable without changing the scoring rule.
    """

    tokenizer = processor.tokenizer
    variants: dict[str, Any] = {}
    for name, prefix in (("plain", ""), ("leading_space", " ")):
        texts = [f"{prefix}{letter}" for letter in C.LETTERS]
        token_ids: list[list[int] | None] = []
        for text in texts:
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
                token_ids.append([int(value) for value in encoded])
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                token_ids.append(None)
        available = all(ids is not None for ids in token_ids)
        one_token = available and all(len(ids) == 1 for ids in token_ids if ids is not None)
        distinct = one_token and len({ids[0] for ids in token_ids if ids is not None}) == 4
        variants[name] = {
            "texts": texts,
            "token_ids": token_ids,
            "available": available,
            "one_token": one_token,
            "distinct": distinct,
        }
    return variants


def _cpu_split_sizes(grid_thw: Any, spatial_merge_size: int) -> list[int]:
    """Compute vision split sizes without reducing a CUDA metadata tensor.

    Qwen3.5's visual encoder consumes ``grid_thw`` on the active device, but
    ``get_image_features`` only needs Python integers after the encoder has
    returned.  Keeping this reduction on CPU avoids dispatching a tiny CUDA
    ``prod`` kernel followed immediately by a device-to-host ``tolist``.
    """

    if grid_thw is None:
        raise ValueError("Qwen3.5 vision features require grid_thw metadata")
    detached = grid_thw.detach() if hasattr(grid_thw, "detach") else grid_thw
    cpu_grid = detached.cpu() if hasattr(detached, "cpu") else detached
    products = cpu_grid.prod(-1) if hasattr(cpu_grid, "prod") else None
    if products is None or not hasattr(products, "tolist"):
        raise TypeError("grid_thw must expose detach/cpu/prod/tolist for vision splitting")
    merge = int(spatial_merge_size) ** 2
    values = products.tolist()
    return [int(value) // merge for value in values]


def _cpu_vision_cu_seqlens(grid_thw: Any, torch_module: Any) -> Any:
    """Build Qwen3.5's tiny non-Flash split metadata on CPU."""

    if grid_thw is None:
        raise ValueError("Qwen3.5 vision features require grid_thw metadata")
    detached = grid_thw.detach() if hasattr(grid_thw, "detach") else grid_thw
    cpu_grid = detached.cpu() if hasattr(detached, "cpu") else detached
    rows = cpu_grid.tolist() if hasattr(cpu_grid, "tolist") else None
    if not isinstance(rows, list):
        raise TypeError("grid_thw must expose CPU values for vision attention splitting")
    boundaries = [0]
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("grid_thw rows must contain temporal, height, and width values")
        temporal, height, width = (int(value) for value in row)
        if temporal < 0 or height < 0 or width < 0:
            raise ValueError("grid_thw values must be non-negative")
        for _ in range(temporal):
            boundaries.append(boundaries[-1] + height * width)
    return torch_module.tensor(boundaries, dtype=torch_module.int32)


def _patch_qwen38_vision_split(model: Any, torch_module: Any) -> bool:
    """Keep Qwen3.8's tiny vision split calculation off the CUDA reduction path.

    The official Qwen3.5 implementation uses the same grid tensor for CUDA
    vision position construction and several host-only split calculations.
    Keep the grid on CUDA for the encoder, but provide CPU copies for split
    metadata rather than changing Transformers or the Stage A scorer.
    """

    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "get_image_features"):
        return False

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        # With the Phase 8 adapter's SDPA configuration, the vision attention
        # path consumes ``cu_seqlens`` only to produce Python split sizes.  A
        # CPU tensor avoids the repeated CUDA ``tolist()`` synchronization in
        # every vision block while retaining the CUDA grid for positional data.
        kwargs.setdefault("cu_seqlens", _cpu_vision_cu_seqlens(image_grid_thw, torch_module))
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
        split_sizes = _cpu_split_sizes(image_grid_thw, self.visual.spatial_merge_size)
        vision_outputs.pooler_output = torch_module.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    inner.get_image_features = types.MethodType(get_image_features, inner)
    return True


def _messages(q: dict[str, Any], permutation: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    shown, actual = C.permute_options(q, permutation)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": C.SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": C.user_text(q, shown)},
            ],
        },
    ]
    return messages, actual


class BasePhase8Adapter:
    """Common video encoding and model loading with family-specific hooks."""

    required_keys = {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"}

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def render_chat(self, processor: Any, messages: list[dict[str, Any]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.spec.non_thinking_chat:
            # The Transformers chat-template API takes this as a direct
            # keyword.  ``chat_template_kwargs`` is the nested shape used by
            # API request bodies, not by ``apply_chat_template``.
            kwargs["enable_thinking"] = False
        return processor.apply_chat_template(messages, **kwargs)

    def _model_class(self):
        if self.spec.family == "qwen38":
            from transformers import AutoModelForMultimodalLM

            return AutoModelForMultimodalLM
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration

    def load(self):
        """Load the exact declared processor and model implementation."""

        import torch
        from transformers import AutoProcessor

        cache_dir = os.environ.get("HF_HUB_CACHE")
        cache_kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        processor = AutoProcessor.from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            local_files_only=True,
            **cache_kwargs,
        )
        C.configure_attention_backend("sdpa", torch)
        model = self._model_class().from_pretrained(
            self.spec.model_id,
            revision=self.spec.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            **cache_kwargs,
        )
        if self.spec.family == "qwen38":
            if not _patch_qwen38_vision_split(model, torch):
                raise RuntimeError(
                    "Qwen3.8 model does not expose the expected model.get_image_features API"
                )
        return processor, model

    def encode(
        self,
        processor: Any,
        q: dict[str, Any],
        frames: Any,
        meta: Any,
        permutation: Iterable[str],
    ) -> tuple[dict[str, Any], list[str]]:
        messages, actual = _messages(q, permutation)
        text = self.render_chat(processor, messages)
        raw = C._processor_call(processor, text, frames, meta)
        allowed = {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
            "image_grid_thw",
            "rope_deltas",
            "position_ids",
            "cache_position",
        }
        encoded = {key: value for key, value in raw.items() if key in allowed}
        missing = self.required_keys - encoded.keys()
        if missing:
            raise KeyError(
                f"{self.spec.family} processor did not return required model inputs: {sorted(missing)}"
            )
        return encoded, actual


class Qwen38Adapter(BasePhase8Adapter):
    """Qwen3.8-27B adapter using AutoModelForMultimodalLM and no thinking."""


class Qwen3VL32BAdapter(BasePhase8Adapter):
    """Qwen3-VL-32B-Instruct adapter using its native conditional generator."""

    required_keys = BasePhase8Adapter.required_keys | {"mm_token_type_ids"}


def adapter_for_family(family: str, **kwargs: Any) -> BasePhase8Adapter:
    spec = resolve_model_spec(family, **kwargs)
    if spec.family == "qwen38":
        return Qwen38Adapter(spec)
    return Qwen3VL32BAdapter(spec)

