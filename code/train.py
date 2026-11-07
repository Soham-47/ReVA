"""Resumable LoRA training with an opt-in temporal-prefix answer loss."""

from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import mirror_answers as M
import specialist_data as S


def _temporal_grounding():
    """Load the archived dual-stream helper only when temporal_stream.enabled."""
    archive = Path(__file__).resolve().parents[1] / "archive" / "implementation" / "code"
    sys.path.insert(0, str(archive))
    import temporal_grounding as TG  # noqa: WPS433

    return TG


LANGUAGE_TARGET = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
QWEN38_LANGUAGE_TARGET = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)
VISUAL_MERGER_SUFFIXES = (
    "visual.merger.linear_fc1",
    "visual.merger.linear_fc2",
)
DEEPSTACK_MERGER_SUFFIXES = tuple(
    f"visual.deepstack_merger_list.{index}.{suffix}"
    for index in range(3)
    for suffix in ("linear_fc1", "linear_fc2")
)
DEFAULT_VISION_ADAPTER = "vision_encoder"
DEFAULT_MERGER_ADAPTER = "visual_merger"


def model_family(config: dict) -> str:
    """Return the explicitly selected training backbone family."""

    family = str(config.get("model_family", "qwen3_vl")).strip().casefold()
    aliases = {
        "qwen3_vl": "qwen3_vl",
        "qwen3-vl": "qwen3_vl",
        "qwen3vl": "qwen3_vl",
        "qwen38": "qwen38",
        "qwen3.8": "qwen38",
        "qwen3.8-27b": "qwen38",
    }
    try:
        return aliases[family]
    except KeyError as exc:
        raise ValueError(f"unsupported training model_family: {family!r}") from exc


def chat_template_enable_thinking(config: dict) -> bool | None:
    """Resolve an optional family-specific chat-template setting."""

    template = config.get("chat_template", {})
    if "enable_thinking" in template:
        return bool(template["enable_thinking"])
    return False if model_family(config) == "qwen38" else None


def language_target(config: dict) -> str:
    """Select the complete language projection set for the configured family."""

    return QWEN38_LANGUAGE_TARGET if model_family(config) == "qwen38" else LANGUAGE_TARGET


def _is_linear_module(module) -> bool:
    """Match plain Linear and PEFT-wrapped Linear modules."""

    if isinstance(module, torch.nn.Linear):
        return True
    base_layer = getattr(module, "base_layer", None)
    return isinstance(base_layer, torch.nn.Linear)


def newest_checkpoint(output: Path) -> str | None:
    checkpoints = []
    for path in glob.glob(str(output / "checkpoint-*")):
        try:
            checkpoints.append((int(Path(path).name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints)[1] if checkpoints else None


class ReVATrain(Dataset):
    def __init__(
        self,
        json_path: Path,
        processor,
        config: dict,
        ltr_ids: list[int],
        im_end_id: int,
        temporal_contract=None,
    ):
        with json_path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or not isinstance(payload.get("QA"), list):
            raise ValueError("training JSON must be an object with a QA list")
        expected_split = config.get("require_split")
        if expected_split is not None:
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("split") != expected_split:
                raise ValueError(f"training JSON metadata.split must be {expected_split!r}")
        self.rows = payload["QA"]
        if not self.rows or any(not row["qa_id"].startswith("train_") for row in self.rows):
            raise ValueError("training JSON must contain train_* rows only; val/test leakage is forbidden")
        self.processor = processor
        self.config = config
        self.ltr_ids = ltr_ids
        self.im_end_id = im_end_id
        self.data_seed = int(config.get("data_seed", config["seed"]))
        self.deterministic_sampling = bool(config.get("deterministic_sampling", False))
        self.enable_thinking = chat_template_enable_thinking(config)
        self.temporal_contract = temporal_contract
        self.temporal_config = config.get("temporal_stream", {}) if temporal_contract is not None else {}
        task_filter = self.temporal_config.get("train_task")
        if task_filter is not None:
            if not isinstance(task_filter, str) or not task_filter.strip():
                raise ValueError("temporal_stream.train_task must be a non-empty task name")
            self.rows = [row for row in self.rows if row.get("task") == task_filter]
            if not self.rows:
                raise ValueError(f"temporal_stream.train_task matched no training rows: {task_filter!r}")
        self.temporal_manifest = None
        if self.temporal_contract is not None:
            manifest_path = C.work_path(
                self.temporal_config.get("duration_manifest", "data/manifest.json")
            )
            if manifest_path.is_file():
                with manifest_path.open() as handle:
                    self.temporal_manifest = json.load(handle)
        self.frame_budget = S.normalize_frame_budget(config.get("frame_budget"))
        self.preserve_frame_budget = bool(config.get("preserve_frame_budget", False))
        self.motion_hints_required = bool(config.get("motion_hints_required", False))
        self.strict_hint_vocabulary = bool(config.get("strict_hint_vocabulary", False))
        self.mirror_safe = bool(config.get("mirror_safe", False))
        self.mirror_probability = float(config.get("mirror_probability", 0.0))
        if self.mirror_safe and not 0 < self.mirror_probability <= 1:
            raise ValueError("mirror_safe requires mirror_probability in (0, 1]")
        if self.mirror_safe and self.temporal_contract is not None:
            raise ValueError("mirror_safe cannot be combined with temporal_stream")
        train_tasks = config.get("train_tasks")
        if train_tasks is not None:
            self.rows = S.filter_rows_by_tasks(self.rows, train_tasks)
            S.assert_train_only_rows(self.rows)
        self.hint_formatter = getattr(C, "motion_hint_text", None)
        self.motion_hints = None
        motion_hints_path = config.get("motion_hints")
        if motion_hints_path:
            hint_path = C.work_path(motion_hints_path)
            if not hint_path.is_file():
                if self.motion_hints_required:
                    raise FileNotFoundError(f"required motion hints are missing: {hint_path}")
            else:
                with hint_path.open() as handle:
                    self.motion_hints = S.load_hint_table(json.load(handle))
                if self.motion_hints_required:
                    S.assert_hint_coverage(
                        self.motion_hints,
                        self.rows,
                        strict_vocabulary=self.strict_hint_vocabulary,
                        formatter=self.hint_formatter,
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def _duration(self, row: dict) -> float | None:
        for key in ("duration", "video_duration", "duration_sec"):
            if key in row:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    return value
        if self.temporal_manifest is not None:
            item = self.temporal_manifest.get(row.get("video_path"), {})
            try:
                value = float(item.get("duration"))
            except (TypeError, ValueError, AttributeError):
                return None
            return value if math.isfinite(value) and value > 0 else None
        return None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        # Phase 6 enables this so A1/A2 see the same frame offset and option permutation.
        rng = random.Random(self.data_seed + index) if self.deterministic_sampling else random.Random()
        permutation = list(C.LETTERS)
        rng.shuffle(permutation)
        sample_offset = rng.random()
        row_for_model = row
        if self.mirror_safe and random.Random(self.data_seed + 917_311 * index).random() < self.mirror_probability:
            rewritten = M.mirror_qa(row)
            if rewritten is not None:
                row_for_model = rewritten
        frames, meta = C.get_video(row["video_path"], sample_offset, frame_budget=self.frame_budget)
        if row_for_model is not row:
            frames = M.flip_horizontal(frames)
        motion_context = None
        if self.motion_hints:
            motion_context = S.motion_context_for(
                self.motion_hints,
                row["video_path"],
                required=self.motion_hints_required,
                strict_vocabulary=self.strict_hint_vocabulary,
                formatter=self.hint_formatter,
            )
            if row_for_model is not row and motion_context:
                motion_context = M.swap_left_right(motion_context)
        temporal_ids = None
        temporal_supervision = None
        if self.temporal_contract is not None:
            temporal_ids = self.temporal_contract.for_row(row, self._duration(row))
            temporal_supervision = temporal_ids is not None
        encoded, _ = C.encode(
            self.processor,
            row_for_model,
            frames,
            meta,
            permutation,
            row_for_model["correct_answer"],
            self.im_end_id,
            self.ltr_ids,
            enable_thinking=self.enable_thinking,
            temporal_token_ids=temporal_ids,
            temporal_supervision=temporal_supervision,
            temporal_context=motion_context,
        )
        S.guard_token_budget(self.config, int(encoded["input_ids"].shape[1]), row["qa_id"])
        if encoded["input_ids"].shape[1] > self.config["train"]["max_tokens"]:
            frames, meta = C.downsample_video(frames, meta)
            encoded, _ = C.encode(
                self.processor,
                row_for_model,
                frames,
                meta,
                permutation,
                row_for_model["correct_answer"],
                self.im_end_id,
                self.ltr_ids,
                enable_thinking=self.enable_thinking,
                temporal_token_ids=temporal_ids,
                temporal_supervision=temporal_supervision,
                temporal_context=motion_context,
            )
        if encoded["input_ids"].shape[1] > self.config["train"]["max_tokens"]:
            raise RuntimeError(f"sample {row['qa_id']} still exceeds max_tokens after downsampling")
        if self.temporal_contract is not None:
            total_frames = int(self.temporal_config.get("num_frames", 96))
            num_segments = int(self.temporal_config.get("num_segments", 12))
            frames_per_segment = int(self.temporal_config.get("frames_per_segment", total_frames // num_segments))
            if total_frames != num_segments * frames_per_segment:
                raise ValueError("temporal_stream num_frames must equal num_segments * frames_per_segment")
            temporal_frames, _ = C.get_video(
                row["video_path"],
                sample_offset,
                mode=self.temporal_config.get("sampling_mode", C.DEFAULT_FRAME_SAMPLING),
                frame_budget=total_frames,
            )
            encoded["temporal_pixel_values"] = _temporal_grounding().preprocess_temporal_frames(
                temporal_frames,
                num_segments=num_segments,
                frames_per_segment=frames_per_segment,
                image_size=int(self.temporal_config.get("image_size", 224)),
            )
            encoded["temporal_segment_mask"] = torch.ones(num_segments, dtype=torch.bool)
        return {
            key: value[0] if key in {"input_ids", "attention_mask", "labels", "temporal_label_mask"} else value
            for key, value in encoded.items()
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("video samples use batch size 1 to avoid padding variable visual tensors")
    row = batch[0]
    result = {
        "input_ids": row["input_ids"].unsqueeze(0),
        "attention_mask": row["attention_mask"].unsqueeze(0),
        "labels": row["labels"].unsqueeze(0),
        "mm_token_type_ids": row["mm_token_type_ids"],
        "pixel_values_videos": row["pixel_values_videos"],
        "video_grid_thw": row["video_grid_thw"],
    }
    if "second_per_grid_ts" in row:
        result["second_per_grid_ts"] = row["second_per_grid_ts"]
    for key in ("temporal_pixel_values", "temporal_segment_mask"):
        if key in row:
            result[key] = row[key].unsqueeze(0)
    for key in ("temporal_supervision_mask", "temporal_label_mask"):
        if key in row:
            result[key] = row[key].unsqueeze(0) if row[key].ndim else row[key]
    return result


class Stopper(TrainerCallback):
    def __init__(self, stop_path: Path):
        self.stop_path = stop_path

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_path.exists():
            control.should_save = True
            control.should_training_stop = True
        return control


def make_lora(model, config: dict):
    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=language_target(config),
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapted = get_peft_model(model, peft_config)
    adapted.print_trainable_parameters()
    named = [name for name, param in adapted.named_parameters() if param.requires_grad and "lora_" in name]
    language_count = sum("language_model" in name for name in named)
    print("trainable LoRA language tensors:", language_count)
    if not language_count:
        raise RuntimeError("no language-model LoRA targets matched; check the pinned Transformers model")
    return adapted


def load_training_components(config: dict, attn_implementation: str):
    """Load the configured processor/model while preserving Stage A defaults."""

    family = model_family(config)
    if family == "qwen38":
        from phase8_adapters import adapter_for_family

        adapter = adapter_for_family(
            "qwen38",
            model_id=config.get("model_id"),
            revision=config.get("model_revision"),
        )
        processor, model = adapter.load()
        return processor, model

    processor = C.load_processor()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    return processor, model


def visual_merger_targets(model, include_deepstack: bool = False) -> list[str]:
    """Discover the pinned Qwen visual merger linears and fail closed on drift."""

    expected = set(VISUAL_MERGER_SUFFIXES)
    if include_deepstack:
        expected.update(DEEPSTACK_MERGER_SUFFIXES)
    matched = {
        suffix
        for name, module in model.named_modules()
        if _is_linear_module(module)
        for suffix in expected
        if name == suffix or name.endswith(f".{suffix}")
    }
    if matched != expected:
        raise RuntimeError(
            "pinned Qwen3-VL visual merger targets changed; "
            f"expected {sorted(expected)}, found {sorted(matched)}"
        )
    return sorted(expected)


def visual_encoder_targets(model, config: dict) -> list[str]:
    """Resolve configured vision-block linears against the live model tree."""

    visual = config.get("visual_encoder", {})
    blocks = visual.get("blocks", [])
    modules = visual.get("target_modules", ["attn.qkv", "attn.proj"])
    if not visual.get("enabled", False):
        return []
    if not blocks or not modules:
        raise ValueError("visual_encoder requires non-empty blocks and target_modules")
    targets = [f"visual.blocks.{int(block)}.{module}" for block in blocks for module in modules]
    available = dict(model.named_modules())
    missing = [
        target
        for target in targets
        if not any(
            _is_linear_module(module)
            and (name == target or name.endswith(f".{target}"))
            for name, module in available.items()
        )
    ]
    if missing:
        raise RuntimeError(
            "configured Qwen3-VL vision LoRA targets were not found as Linear modules: "
            f"{missing}"
        )
    if len(set(targets)) != len(targets):
        raise ValueError("visual_encoder target paths must be unique")
    return targets


def _trainable_lora_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name]


def prepare_model(model, config: dict):
    """Build A0, load Stage A, then add the configured visual adapters."""

    init_adapter = config.get("init_adapter")
    if not init_adapter:
        adapted = make_lora(model, config)
        return adapted

    from peft import PeftModel

    adapter_path = C.work_path(init_adapter)
    if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"initial adapter is missing or invalid: {adapter_path}")
    adapted = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    adapter_names = list(adapted.peft_config)
    if adapter_names != ["default"]:
        raise RuntimeError(f"Stage A must load exactly one default adapter, found {adapter_names}")
    language_params = [
        (name, parameter)
        for name, parameter in _trainable_lora_parameters(adapted)
        if "language_model" in name
    ]
    if not language_params:
        raise RuntimeError("loaded Stage A adapter has no trainable language-model LoRA parameters")
    if any("visual." in name for name, _ in _trainable_lora_parameters(adapted)):
        raise RuntimeError("Stage A adapter unexpectedly contains visual LoRA parameters")
    print(f"loaded initial adapter={adapter_path} language_lora_params={sum(p.numel() for _, p in language_params)}")

    visual_encoder = config.get("visual_encoder", {})
    visual_merger = config.get("visual_merger", {})
    adapter_specs = []
    if visual_encoder.get("enabled", False):
        vision_name = str(visual_encoder.get("adapter_name", DEFAULT_VISION_ADAPTER))
        vision_targets = visual_encoder_targets(adapted, config)
        adapter_specs.append((vision_name, vision_targets))
    if visual_merger.get("enabled", False):
        merger_name = str(visual_merger.get("adapter_name", DEFAULT_MERGER_ADAPTER))
        merger_targets = visual_merger_targets(
            adapted, include_deepstack=bool(visual_merger.get("include_deepstack", False))
        )
        adapter_specs.append((merger_name, merger_targets))
    if not adapter_specs:
        return adapted
    names = [name for name, _ in adapter_specs]
    if len(set(names)) != len(names) or "default" in names:
        raise ValueError(f"visual adapter names must be unique and cannot be default: {names}")
    lora = config["lora"]
    for adapter_name, targets in adapter_specs:
        adapter_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        adapted.add_adapter(adapter_name, adapter_config)
    active = ["default", *names]
    # PEFT exposes multi-adapter activation on the underlying tuner.
    adapted.base_model.set_adapter(active)
    adapted.base_model.set_requires_grad(active)
    trainable = _trainable_lora_parameters(adapted)
    for adapter_name in names:
        params = [(name, parameter) for name, parameter in trainable if f".{adapter_name}." in name]
        if not params:
            raise RuntimeError(f"visual adapter {adapter_name!r} did not expose trainable parameters")
        print(
            f"active_adapters={adapted.base_model.active_adapters} adapter={adapter_name} "
            f"targets={dict(adapter_specs)[adapter_name]} lora_params={sum(p.numel() for _, p in params)}"
        )
    return adapted


def make_optimizer(model, config: dict):
    """Create explicit language, vision, merger, and optional temporal groups."""

    train_cfg = config["train"]
    if not train_cfg.get("explicit_optimizer", False):
        return None
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    adapter_names = {
        "language": "default",
        "vision": str(config.get("visual_encoder", {}).get("adapter_name", DEFAULT_VISION_ADAPTER)),
        "merger": str(config.get("visual_merger", {}).get("adapter_name", DEFAULT_MERGER_ADAPTER)),
    }
    temporal_enabled = bool(config.get("temporal_stream", {}).get("enabled", False))
    if temporal_enabled:
        adapter_names["temporal"] = "__temporal__"
    group_keys = ("language", "vision", "merger", "temporal") if temporal_enabled else ("language", "vision", "merger")
    temporal_parameter_ids = set()
    if temporal_enabled:
        table = getattr(model, "temporal_token_table", None)
        if table is not None:
            temporal_parameter_ids.add(id(table))
        stream = getattr(model, "temporal_stream", None)
        if stream is not None:
            temporal_parameter_ids.update(id(parameter) for parameter in stream.projector.parameters())
    buckets = {key: [] for key in adapter_names}
    unexpected = []
    for name, parameter in named:
        matches = [key for key, adapter_name in adapter_names.items() if f".{adapter_name}." in name]
        if temporal_enabled and id(parameter) in temporal_parameter_ids:
            buckets["temporal"].append(parameter)
        elif len(matches) != 1 or "lora_" not in name:
            unexpected.append(name)
        else:
            buckets[matches[0]].append(parameter)
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected[:8]}")
    configured_lrs = {
        "language": train_cfg.get("language_lr", train_cfg["lr"]),
        "vision": train_cfg.get("vision_lr"),
        "merger": train_cfg.get("merger_lr", train_cfg.get("visual_lr")),
        "temporal": train_cfg.get("temporal_lr", train_cfg.get("lr")) if temporal_enabled else None,
    }
    groups = []
    for key in group_keys:
        params = buckets[key]
        lr = configured_lrs[key]
        if params and lr is None:
            raise RuntimeError(f"{key} adapter is trainable but its learning rate is missing")
        if lr is not None and not params:
            raise RuntimeError(f"{key} learning rate is configured but no adapter parameters are trainable")
        if params:
            groups.append({"params": params, "lr": float(lr), "name": key})
    if not groups:
        raise RuntimeError("optimizer has no intended trainable adapter parameters")
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(train_cfg.get("language_lr", train_cfg["lr"])),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print(
        "optimizer_groups "
        + " ".join(
            f"{key}_params={sum(p.numel() for p in buckets[key])} {key}_lr={configured_lrs[key]}"
            for key in group_keys
            if buckets[key]
        )
    )
    return optimizer


class StepBenchmark(TrainerCallback):
    """Print one bounded timing sample and an extrapolated run duration."""

    def __init__(self, total_steps: float, limit: int = 50):
        self.total_steps = total_steps
        self.limit = limit
        self.started = None
        self.times: list[float] = []

    def on_step_begin(self, args, state, control, **kwargs):
        if len(self.times) < self.limit:
            self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is None:
            return control
        self.times.append(time.monotonic() - self.started)
        self.started = None
        if len(self.times) in {1, min(5, self.limit), self.limit}:
            mean_seconds = sum(self.times) / len(self.times)
            hours = C.estimate_training_hours(1, self.total_steps, 1, mean_seconds)
            print(
                f"benchmark steps={len(self.times)} mean_step_seconds={mean_seconds:.3f} "
                f"estimated_train_hours={hours:.2f}",
                flush=True,
            )
        return control


class KernelStepCheck(TrainerCallback):
    """Record that one real optimizer step changed a trainable LoRA tensor."""

    def __init__(self, output: Path):
        self.marker = output / "KERNEL_SMOKE_STEP_OK"
        self.before = None

    def _lora_tensors(self, model):
        return {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        }

    def on_step_begin(self, args, state, control, **kwargs):
        if self.marker.exists() or self.before is not None:
            return control
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("kernel smoke could not see the model to record LoRA weights")
        self.before = self._lora_tensors(model)
        if not self.before:
            raise RuntimeError("kernel smoke found no trainable LoRA tensors")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.marker.exists() or self.before is None:
            return control
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("kernel smoke could not see the model after the optimizer step")
        changed = False
        for name, parameter in model.named_parameters():
            if name not in self.before:
                continue
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"kernel smoke LoRA tensor is non-finite: {name}")
            if not torch.equal(self.before[name], parameter.detach()):
                changed = True
        if not changed:
            raise RuntimeError("kernel smoke optimizer step changed no trainable LoRA tensor")
        for entry in reversed(state.log_history):
            loss = entry.get("loss")
            if loss is None:
                continue
            if not math.isfinite(float(loss)):
                raise RuntimeError("kernel smoke logged a non-finite loss")
            break
        self.marker.write_text(f"global_step={state.global_step}\n")
        self.before = None
        print(f"kernel_smoke_step_ok global_step={state.global_step}", flush=True)
        return control


class TemporalBundleCheckpoint(TrainerCallback):
    """Write the opt-in temporal state beside each Trainer checkpoint.

    ``DualStreamQwen3VL`` is an isolated ``nn.Module`` wrapper rather than a
    Transformers ``PreTrainedModel``. Depending on the installed Trainer
    version, intermediate saves may therefore use a raw state dict instead of
    calling ``save_pretrained``. The callback makes the custom projector,
    temporal token table, vocabulary, and runtime identity available in both
    checkpoint layouts without affecting the historical path.
    """

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        TG = _temporal_grounding()
        if not isinstance(model, TG.DualStreamQwen3VL):
            raise RuntimeError("temporal checkpoint callback received the wrong model wrapper")
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        TG.save_temporal_bundle(model, checkpoint, model.tokenizer)
        return control


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python code/train.py code/configs/stageA.yaml")
    with Path(sys.argv[1]).open() as f:
        config = yaml.safe_load(f)
    if config.get("specialist_contract"):
        S.validate_run_config(config)
    if config.get("require_smoke_marker"):
        marker = os.environ.get("REVA_SMOKE_OK")
        if not marker or not Path(marker).is_file():
            raise RuntimeError("specialist training requires REVA_SMOKE_OK to point at an existing smoke marker")
    work = C.work_dir()
    output = C.work_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if config.get("isolate_output"):
        snapshot = output / "config_snapshot.yaml"
        incoming = Path(sys.argv[1]).read_text()
        if snapshot.exists() and snapshot.read_text() != incoming:
            raise RuntimeError(f"refusing to overwrite {output}; resolved config differs from config_snapshot.yaml")
        if not snapshot.exists():
            snapshot.write_text(incoming)
    if (output / "DONE").exists():
        print(f"already complete: {output}")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL training")
    if config["train"].get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured bf16 training is unavailable on this GPU")
    attn_implementation = config.get("attn_implementation", "sdpa")
    C.configure_attention_backend(attn_implementation)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    processor, model = load_training_components(config, attn_implementation)
    tokenizer = processor.tokenizer
    ltr_ids = C.letter_ids(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise RuntimeError("tokenizer has no <|im_end|> token")

    model.config.use_cache = False
    temporal_contract = None
    if config.get("temporal_stream", {}).get("enabled", False):
        TG = _temporal_grounding()
        temporal_cfg = config["temporal_stream"]
        vocabulary = TG.ensure_temporal_tokens(
            model,
            tokenizer,
            num_bins=int(temporal_cfg.get("num_bins", TG.NUM_BINS)),
        )
        temporal_contract = TG.TemporalTargetContract(vocabulary)
    adapted = prepare_model(model, config)
    if temporal_contract is not None:
        TG = _temporal_grounding()
        hidden_size = int(model.get_input_embeddings().embedding_dim)
        temporal_encoder = TG.load_internvideo2_encoder(config["temporal_stream"])
        feature_dim = TG.infer_temporal_feature_dim(
            temporal_encoder,
            expected=temporal_cfg.get("feature_dim"),
        )
        temporal_stream = TG.InternVideo2TemporalStream(
            temporal_encoder,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            num_segments=int(temporal_cfg.get("num_segments", 12)),
            frames_per_segment=int(temporal_cfg.get("frames_per_segment", 8)),
            projector_hidden=temporal_cfg.get("projector_hidden"),
        )
        adapted = TG.DualStreamQwen3VL(
            adapted,
            temporal_stream,
            vocabulary,
            tokenizer,
            runtime_config=temporal_cfg,
        )
    adapted.enable_input_require_grads()
    train_cfg = config["train"]
    args = TrainingArguments(
        output_dir=str(output),
        run_name=config["run_name"],
        num_train_epochs=train_cfg["epochs"],
        max_steps=int(train_cfg.get("max_steps", -1)),
        learning_rate=train_cfg["lr"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=train_cfg["grad_accum"],
        warmup_steps=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=train_cfg["logging_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        dataloader_num_workers=train_cfg["dataloader_workers"],
        dataloader_persistent_workers=train_cfg["dataloader_workers"] > 0,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        dataloader_pin_memory=True,
    )
    dataset = ReVATrain(
        C.work_path(config["train_json"]),
        processor,
        config,
        ltr_ids,
        im_end_id,
        temporal_contract=temporal_contract,
    )
    configured_steps = int(train_cfg.get("max_steps", -1))
    total_steps = configured_steps if configured_steps > 0 else math.ceil(len(dataset) / train_cfg["grad_accum"]) * train_cfg["epochs"]
    optimizer = make_optimizer(adapted, config)
    if config.get("specialist_contract"):
        print(
            f"specialist_contract={config['specialist_contract']} train_rows={len(dataset)} "
            f"frame_budget={config.get('frame_budget')}",
            flush=True,
        )
    callbacks = [Stopper(output / "STOP"), StepBenchmark(total_steps, train_cfg.get("benchmark_steps", 50))]
    if config.get("kernel_smoke"):
        callbacks.append(KernelStepCheck(output))
    if temporal_contract is not None:
        callbacks.append(TemporalBundleCheckpoint())
    resume_checkpoint = newest_checkpoint(output)
    if temporal_contract is not None and resume_checkpoint is not None:
        raise RuntimeError(
            "automatic resume is disabled for the opt-in temporal wrapper: "
            "Trainer/PEFT checkpoint restoration has not been runtime-validated "
            "in this dependency environment; move the existing checkpoint aside "
            "or obtain an approved integration smoke test before resuming"
        )
    trainer = Trainer(
        model=adapted,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=callbacks,
        optimizers=(optimizer, None) if optimizer is not None else (None, None),
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    if (output / "STOP").exists():
        (output / "STOP").unlink()
        print("stopped after checkpoint; rerun to resume")
        return 0
    adapted.save_pretrained(output / "final_adapter")
    processor.save_pretrained(output / "final_adapter")
    (output / "DONE").write_text("ok\n")
    print(f"training complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
