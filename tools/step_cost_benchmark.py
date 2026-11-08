#!/usr/bin/env python3
"""50 optimizer steps in-process (no DataLoader workers).

Measures the same LoRA step as code/train.py without Hugging Face Trainer
prefetch, which stalled on Brev after the first step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C
import train as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    train_cfg = config["train"]
    steps = int(args.steps)
    attn = config.get("attn_implementation", "sdpa")
    C.configure_attention_backend(attn)

    processor, model = T.load_training_components(config, attn)
    tokenizer = processor.tokenizer
    ltr_ids = C.letter_ids(tokenizer)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.config.use_cache = False
    adapted = T.prepare_model(model, config)
    adapted.enable_input_require_grads()
    adapted.train()
    optimizer = T.make_optimizer(adapted, config)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in adapted.parameters() if parameter.requires_grad),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    dataset = T.ReVATrain(
        C.work_path(config["train_json"]),
        processor,
        config,
        ltr_ids,
        im_end_id,
    )
    device = torch.device("cuda")
    adapted.to(device)
    grad_accum = int(train_cfg["grad_accum"])
    full_updates = (len(dataset) + grad_accum - 1) // grad_accum * int(train_cfg["epochs"])

    times: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    sample_index = 0
    while len(times) < steps:
        started = time.monotonic()
        for _ in range(grad_accum):
            batch = T.collate([dataset[sample_index % len(dataset)]])
            sample_index += 1
            moved = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            loss = adapted(**moved).loss / grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), float(train_cfg["max_grad_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        times.append(time.monotonic() - started)
        if len(times) in {1, min(5, steps), steps}:
            mean_seconds = sum(times) / len(times)
            hours = C.estimate_training_hours(1, full_updates, 1, mean_seconds)
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(
                f"benchmark steps={len(times)} mean_step_seconds={mean_seconds:.3f} "
                f"estimated_train_hours={hours:.2f} peak_vram_gb={peak_gb:.2f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
