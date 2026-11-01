"""Run a small GPU forward/backward check before submitting long Slurm jobs."""

from __future__ import annotations

import json
import argparse
import hashlib
import sys
import time
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from train import (
    chat_template_enable_thinking,
    load_training_components,
    make_optimizer,
    prepare_model,
    visual_encoder_targets,
    visual_merger_targets,
)


def _tensor_value_hash(state: dict[str, torch.Tensor]) -> tuple[int, str]:
    """Hash adapter tensor values without depending on PEFT key prefixes."""

    digest = hashlib.sha256()
    count = 0
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
        count += 1
    return count, digest.hexdigest()


def _adapter_state_hashes(model, adapter_names: list[str]) -> dict[str, tuple[int, str]]:
    state = model.state_dict()
    result = {}
    for adapter_name in adapter_names:
        selected = {
            key: value
            for key, value in state.items()
            if f".{adapter_name}." in key
            or (adapter_name == "default" and key.endswith(".default.weight"))
        }
        result[adapter_name] = _tensor_value_hash(selected)
    return result


def _saved_adapter_report(root: Path) -> dict[str, tuple[int, str]]:
    """Print saved adapter layout and return value hashes per adapter directory."""

    from safetensors.torch import load_file

    reports = {}
    adapter_dirs = [root]
    adapter_dirs.extend(sorted(child for child in root.iterdir() if child.is_dir()))
    for directory in adapter_dirs:
        adapter_name = "default" if directory == root else directory.name
        configs = sorted(directory.glob("adapter_config.json"))
        tensor_files = sorted(directory.glob("*.safetensors"))
        if not configs and not tensor_files:
            continue
        state = {}
        for tensor_file in tensor_files:
            loaded = load_file(str(tensor_file), device="cpu")
            state.update(loaded)
            file_digest = hashlib.sha256(tensor_file.read_bytes()).hexdigest()
            print(
                f"saved_adapter_file adapter={directory.name} file={tensor_file.name} "
                f"tensor_keys={len(loaded)} sha256={file_digest}"
            )
        reports[adapter_name] = _tensor_value_hash(state)
        print(
            f"saved_adapter_config adapter={adapter_name} configs={len(configs)} "
            f"tensor_keys={reports[adapter_name][0]} value_sha256={reports[adapter_name][1]}"
        )
    return reports


def _report_logit_diff(label: str, reference: torch.Tensor, candidate: torch.Tensor) -> float:
    difference = (reference - candidate).abs()
    max_diff = difference.max().item()
    last_diff = difference[:, -1].max().item()
    print(
        f"{label} max_abs_logit_diff={max_diff:.6f} "
        f"last_token_max_abs_logit_diff={last_diff:.6f} "
        f"mean_abs_logit_diff={difference.mean().item():.6f}"
    )
    return max_diff


def _report_visual_architecture(model, block_ids: list[int]) -> None:
    """Print live vision projection paths before adapters are attached."""

    linear_paths = {
        name: (
            type(module).__name__,
            tuple(getattr(getattr(module, "base_layer", module), "weight").shape),
        )
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    for block_id in block_ids:
        prefix = f"visual.blocks.{block_id}."
        paths = sorted(name for name in linear_paths if prefix in name)
        print(f"visual_block={block_id} linear_paths={[(path, linear_paths[path]) for path in paths]}")
    deepstack = sorted(
        (
            name,
            type(module).__name__,
            tuple(getattr(getattr(module, "base_layer", module), "weight").shape)
            if hasattr(getattr(module, "base_layer", module), "weight")
            else None,
        )
        for name, module in model.named_modules()
        if "deepstack" in name.casefold()
    )
    print(f"visual_deepstack_modules={deepstack}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "stageA.yaml"))
    parser.add_argument("--qa-id", help="run one known example instead of the multi-source smoke set")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("smoke requires a CUDA GPU")
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)
    attn_implementation = config.get("attn_implementation", "sdpa")
    C.configure_attention_backend(attn_implementation)
    processor, model = load_training_components(config, attn_implementation)
    tokenizer = processor.tokenizer
    ltr_ids = C.letter_ids(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    with C.work_path("data/splits/fit_A.json").open() as f:
        rows = json.load(f)["QA"]
    if args.qa_id:
        picks = [next(row for row in rows if row["qa_id"] == args.qa_id)]
    else:
        picks = []
        for source in ("VisDrone", "Hawk_UAV", "ERA", "UAVDT"):
            picks.append(next(row for row in rows if row["dataset_name"].startswith(source)))
    manifest = C.load_manifest()
    if not args.qa_id:
        worst = max(
            rows,
            key=lambda row: (
                manifest[row["video_path"]]["width"] * manifest[row["video_path"]]["height"],
                manifest[row["video_path"]]["n_frames"],
            ),
        )
        if worst["qa_id"] not in {row["qa_id"] for row in picks}:
            picks.append(worst)
            print("including worst native visual sample:", worst["qa_id"], worst["video_path"])
    vision_blocks = [
        int(block)
        for block in config.get("visual_encoder", {}).get("blocks", [])
    ]
    _report_visual_architecture(model, vision_blocks)
    model.config.use_cache = False
    adapted = prepare_model(model, config)
    adapted.enable_input_require_grads()
    if config["train"].get("gradient_checkpointing", True):
        adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapted = adapted.cuda()
    adapted.train()
    optimizer = make_optimizer(adapted, config)
    visual_encoder = config.get("visual_encoder", {})
    visual_merger = config.get("visual_merger", {})
    expected_adapters = {"default"}
    if visual_encoder.get("enabled", False):
        expected_adapters.add(str(visual_encoder.get("adapter_name", "vision_encoder")))
    if visual_merger.get("enabled", False):
        expected_adapters.add(str(visual_merger.get("adapter_name", "visual_merger")))
    trainable_names = [name for name, parameter in adapted.named_parameters() if parameter.requires_grad]
    if any(
        "lora_" not in name or not any(f".{adapter}." in name for adapter in expected_adapters)
        for name in trainable_names
    ):
        raise AssertionError(f"unexpected trainable parameters: {trainable_names[:8]}")
    if not visual_merger.get("enabled", False) and any(
        parameter.requires_grad
        for name, parameter in adapted.named_parameters()
        if "visual.merger" in name and "lora_" not in name
    ):
        raise AssertionError("visual merger base parameters are unexpectedly trainable")
    reload_inputs = None
    for row in picks:
        frames, meta = C.get_video(row["video_path"], 0.0)
        encoded, _ = C.encode(
            processor,
            row,
            frames,
            meta,
            ["B", "D", "A", "C"],
            row["correct_answer"],
            im_end_id,
            ltr_ids,
            enable_thinking=chat_template_enable_thinking(config),
        )
        decoded = tokenizer.decode(encoded["input_ids"][0])
        if "seconds>" not in decoded or "<|im_start|>assistant\n" not in decoded:
            raise AssertionError("prompt is missing Qwen timestamps or assistant generation tag")
        print(row["qa_id"], row["dataset_name"], frames.shape, encoded["video_grid_thw"].tolist(), encoded["input_ids"].shape[1])
        inputs = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in encoded.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        output = adapted(**inputs)
        if not torch.isfinite(output.loss):
            raise AssertionError("smoke loss is not finite")
        before = {
            name: parameter.detach().clone()
            for name, parameter in adapted.named_parameters()
            if parameter.requires_grad and "lora_" in name
        }
        output.loss.backward()
        gradients = [
            (name, parameter.grad)
            for name, parameter in adapted.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not gradients or any(gradient is None or not torch.isfinite(gradient).all() for _, gradient in gradients):
            raise AssertionError("one or more trainable LoRA tensors has no finite gradient")
        for key, visual in (("vision", visual_encoder), ("merger", visual_merger)):
            if visual.get("enabled", False):
                adapter_name = str(
                    visual.get(
                        "adapter_name",
                        "vision_encoder" if key == "vision" else "visual_merger",
                    )
                )
                if not any(
                    f".{adapter_name}." in name and gradient.abs().sum().item() > 0
                    for name, gradient in gradients
                ):
                    raise AssertionError(f"{key} adapter {adapter_name!r} received no nonzero gradient")
                expected_targets = (
                    visual_encoder_targets(adapted, config)
                    if key == "vision"
                    else visual_merger_targets(
                        adapted,
                        include_deepstack=bool(visual.get("include_deepstack", False)),
                    )
                )
                for target in expected_targets:
                    target_gradients = [
                        gradient.detach().float()
                        for name, gradient in gradients
                        if f".{adapter_name}." in name and target in name
                    ]
                    target_norm = (
                        sum(value.pow(2).sum() for value in target_gradients).sqrt().item()
                        if target_gradients
                        else 0.0
                    )
                    print(
                        f"adapter_target_gradient adapter={adapter_name} target={target} "
                        f"tensors={len(target_gradients)} norm={target_norm:.6e}"
                    )
                    if not target_gradients or target_norm <= 0:
                        raise AssertionError(
                            f"{key} target {target!r} adapter {adapter_name!r} received no nonzero gradient"
                        )
        torch.cuda.synchronize()
        print(f"loss={output.loss.item():.3f} seconds={time.monotonic() - started:.2f} peak_gb={torch.cuda.max_memory_allocated()/2**30:.1f}")
        if optimizer is not None:
            optimizer.step()
            changed = any(
                not torch.equal(before[name], parameter.detach())
                for name, parameter in adapted.named_parameters()
                if name in before
            )
            if not changed:
                raise AssertionError("optimizer step changed no trainable LoRA tensor")
            optimizer.zero_grad(set_to_none=True)
        else:
            adapted.zero_grad(set_to_none=True)
        group_grad_norms = {}
        for adapter_name in sorted(expected_adapters):
            group = [
                gradient.detach().float()
                for name, gradient in gradients
                if f".{adapter_name}." in name
            ]
            if group:
                group_grad_norms[adapter_name] = sum(
                    value.pow(2).sum() for value in group
                ).sqrt().item()
        print(f"adapter_gradient_norms={group_grad_norms}")
        reload_inputs = inputs
    if config.get("smoke_reload_check", False):
        adapted.eval()
        with torch.no_grad():
            expected_logits = adapted(**reload_inputs).logits.float().cpu()
        reload_parent = C.work_path(config["output_dir"])
        reload_parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix="phase6_smoke_reload_", dir=str(reload_parent)))
        adapted.save_pretrained(str(temp))
        adapter_names = ["default"]
        if visual_encoder.get("enabled", False):
            adapter_names.append(str(visual_encoder.get("adapter_name", "vision_encoder")))
        if visual_merger.get("enabled", False):
            adapter_names.append(str(visual_merger.get("adapter_name", "visual_merger")))
        in_memory_hashes = _adapter_state_hashes(adapted, adapter_names)
        print(f"save_reload_dir={temp}")
        print(f"in_memory_adapter_hashes={in_memory_hashes}")
        saved_hashes = _saved_adapter_report(temp)
        print(f"saved_adapter_hashes={saved_hashes}")
        for adapter_name in adapter_names:
            print(
                f"adapter_hash_compare adapter={adapter_name} "
                f"match={in_memory_hashes.get(adapter_name) == saved_hashes.get(adapter_name)} "
                f"in_memory={in_memory_hashes.get(adapter_name)} saved={saved_hashes.get(adapter_name)}"
            )

        reloaded_base = Qwen3VLForConditionalGeneration.from_pretrained(
            C.MODEL_ID,
            revision=C.MODEL_REV,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        reloaded = C.load_peft_for_inference(reloaded_base, temp, merge=False)
        loaded_names = list(reloaded.peft_config)
        print(f"reloaded_adapter_names={list(reloaded.peft_config)} active_adapters={reloaded.base_model.active_adapters}")
        reloaded = reloaded.cuda().eval()
        with torch.no_grad():
            before_merge_logits = reloaded(**reload_inputs).logits.float().cpu()
        before_merge_diff = _report_logit_diff(
            "save_reload_before_merge", expected_logits, before_merge_logits
        )
        merged = reloaded.merge_and_unload(adapter_names=loaded_names).eval()
        with torch.no_grad():
            merged_logits = merged(**reload_inputs).logits.float().cpu()
        merged_diff = _report_logit_diff("save_reload_after_merge", expected_logits, merged_logits)
        if not torch.isfinite(before_merge_logits).all() or not torch.isfinite(merged_logits).all():
            raise AssertionError(f"save/reload produced nonfinite logits; saved_dir={temp}")
        if before_merge_diff > 1e-2:
            raise AssertionError(
                f"save/reload output mismatch before merge: {before_merge_diff:.6f}; saved_dir={temp}"
            )
        if merged_diff > 1e-2:
            print(
                f"merge_equivalence_warning max_abs_logit_diff={merged_diff:.6f}; "
                "use --no-merge-adapters for Phase 6B scoring"
            )
        print(
            f"save/reload check passed unmerged_max_abs_logit_diff={before_merge_diff:.6f} "
            f"explicit_merge_max_abs_logit_diff={merged_diff:.6f} saved_dir={temp}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
