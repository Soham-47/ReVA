"""Resumable LoRA training with a one-token answer loss."""

from __future__ import annotations

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


def newest_checkpoint(output: Path) -> str | None:
    checkpoints = []
    for path in output.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), str(path)))
        except ValueError:
            continue
    return max(checkpoints)[1] if checkpoints else None


class ReVATrain(Dataset):
    def __init__(self, json_path: Path, processor, config: dict, ltr_ids: list[int], im_end_id: int):
        with json_path.open() as f:
            self.rows = json.load(f)["QA"]
        if not self.rows or any(not row["qa_id"].startswith("train_") for row in self.rows):
            raise ValueError("training JSON must contain train_* rows only; val/test leakage is forbidden")
        self.processor = processor
        self.config = config
        self.ltr_ids = ltr_ids
        self.im_end_id = im_end_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        rng = random.Random()
        permutation = list(C.LETTERS)
        rng.shuffle(permutation)
        frames, meta = C.get_video(row["video_path"], rng.random())
        encoded, _ = C.encode(
            self.processor,
            row,
            frames,
            meta,
            permutation,
            row["correct_answer"],
            self.im_end_id,
            self.ltr_ids,
        )
        if encoded["input_ids"].shape[1] > self.config["train"]["max_tokens"]:
            frames, meta = C.downsample_video(frames, meta)
            encoded, _ = C.encode(
                self.processor,
                row,
                frames,
                meta,
                permutation,
                row["correct_answer"],
                self.im_end_id,
                self.ltr_ids,
            )
        if encoded["input_ids"].shape[1] > self.config["train"]["max_tokens"]:
            raise RuntimeError(f"sample {row['qa_id']} still exceeds max_tokens after downsampling")
        return {key: value[0] if key in {"input_ids", "attention_mask", "labels"} else value for key, value in encoded.items()}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError("video samples use batch size 1 to avoid padding variable visual tensors")
    row = batch[0]
    result = {
        "input_ids": row["input_ids"].unsqueeze(0),
        "attention_mask": row["attention_mask"].unsqueeze(0),
        "labels": row["labels"].unsqueeze(0),
        "pixel_values_videos": row["pixel_values_videos"],
        "video_grid_thw": row["video_grid_thw"],
    }
    if "second_per_grid_ts" in row:
        result["second_per_grid_ts"] = row["second_per_grid_ts"]
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
    target = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=target,
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


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python code/train.py code/configs/stageA.yaml")
    with Path(sys.argv[1]).open() as f:
        config = yaml.safe_load(f)
    work = C.work_dir()
    output = C.work_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE").exists():
        print(f"already complete: {output}")
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL training")
    if config["train"].get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("configured bf16 training is unavailable on this GPU")

    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    processor = C.load_processor()
    tokenizer = processor.tokenizer
    ltr_ids = C.letter_ids(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_id, int) or im_end_id < 0:
        raise RuntimeError("tokenizer has no <|im_end|> token")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    adapted = make_lora(model, config)
    adapted.enable_input_require_grads()
    train_cfg = config["train"]
    args = TrainingArguments(
        output_dir=str(output),
        run_name=config["run_name"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["lr"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=train_cfg["grad_accum"],
        warmup_ratio=train_cfg["warmup_ratio"],
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
    dataset = ReVATrain(C.work_path(config["train_json"]), processor, config, ltr_ids, im_end_id)
    total_steps = math.ceil(len(dataset) / train_cfg["grad_accum"]) * train_cfg["epochs"]
    trainer = Trainer(
        model=adapted,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[Stopper(output / "STOP"), StepBenchmark(total_steps, train_cfg.get("benchmark_steps", 50))],
    )
    trainer.train(resume_from_checkpoint=newest_checkpoint(output))
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
