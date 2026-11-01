"""Answer-only GRPO on ReVA using its verified multiple-choice labels.

This follows RSVideo's group-relative clipped-policy setup and frozen SFT KL
anchor. ReVA does not provide frame/region evidence labels for this split, so
the reward is exact answer correctness with strict one-letter parsing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import common as C


ANSWER_RE = re.compile(r"\s*([A-D])\s*", re.IGNORECASE)
STOP_REQUESTED = False
SPLITS = {"valQ": "data/splits/valQ.json", "valT": "data/splits/valT.json"}


def parse_answer(text: str) -> str | None:
    match = ANSWER_RE.fullmatch(text or "")
    return match.group(1).upper() if match else None


def shown_gold_letter(gold: str, permutation: list[str]) -> str:
    if gold not in C.LETTERS or sorted(permutation) != sorted(C.LETTERS):
        raise ValueError("gold answer and option permutation must each contain A-D exactly once")
    return C.LETTERS[permutation.index(gold)]


def answer_reward(text: str, shown_gold: str) -> tuple[float, bool]:
    prediction = parse_answer(text)
    return float(prediction == shown_gold), prediction is not None


def self_check() -> int:
    assert parse_answer("A") == "A"
    assert parse_answer(" d\n") == "D"
    assert parse_answer("The answer is A") is None
    assert shown_gold_letter("C", ["A", "C", "B", "D"]) == "B"
    assert answer_reward("B", "B") == (1.0, True)
    assert answer_reward("invalid", "B") == (0.0, False)
    print("answer-only GRPO self-check passed")
    return 0


def _request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _install_stop_handler() -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGUSR1, _request_stop)


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("QA") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected a non-empty QA list in {path}")
    seen: set[str] = set()
    for row in rows:
        qa_id = row.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id.startswith("train_") or qa_id in seen:
            raise ValueError("training data must contain unique train_* IDs only")
        if row.get("correct_answer") not in C.LETTERS:
            raise ValueError(f"invalid answer label for {qa_id}")
        if set(row.get("options", {})) != set(C.LETTERS):
            raise ValueError(f"{qa_id} does not have exactly four A-D options")
        if not isinstance(row.get("video_path"), str) or not row["video_path"]:
            raise ValueError(f"missing video path for {qa_id}")
        seen.add(qa_id)
    return rows


def _load_components(adapter: Path, trainable: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter not found: {adapter}")
    processor = AutoProcessor.from_pretrained(C.MODEL_ID, revision=C.MODEL_REV)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return processor, PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)


def _move_inputs(inputs: dict[str, Any], device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _completion_logprob(model, inputs: dict[str, Any], completion_ids, *, grad: bool, temperature: float):
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prompt_ids = inputs["input_ids"]
    completion_ids = completion_ids.to(prompt_ids.device).unsqueeze(0)
    joined = torch.cat((prompt_ids, completion_ids), dim=1)
    joined_inputs = dict(inputs)
    joined_inputs["input_ids"] = joined
    joined_inputs["attention_mask"] = torch.ones_like(joined)
    text_types = torch.zeros(
        (1, completion_ids.shape[1]),
        dtype=inputs["mm_token_type_ids"].dtype,
        device=prompt_ids.device,
    )
    joined_inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], text_types), dim=1)
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        logits = model(**joined_inputs, use_cache=False).logits[:, :-1]
        first = prompt_ids.shape[1] - 1
        token_logps = torch.log_softmax(logits[:, first:] / temperature, dim=-1)
        return token_logps.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)


def _prompt(processor, row: dict[str, Any], *, work: Path, seed: int, update: int, max_tokens: int):
    rng = random.Random(seed + update)
    offset = rng.random()
    permutation = list(C.LETTERS)
    rng.shuffle(permutation)
    frames, meta = C.get_video(row["video_path"], offset, work=work)
    encoded, permutation = C.encode(processor, row, frames, meta, permutation)
    if encoded["input_ids"].shape[1] > max_tokens:
        frames, meta = C.downsample_video(frames, meta)
        encoded, permutation = C.encode(processor, row, frames, meta, permutation)
    if encoded["input_ids"].shape[1] > max_tokens:
        raise RuntimeError(f"sample {row['qa_id']} still exceeds max_tokens after downsampling")
    return encoded, shown_gold_letter(row["correct_answer"], permutation)


def _checkpoint_paths(output: Path) -> list[Path]:
    valid = []
    for path in output.glob("checkpoint-*"):
        if not path.is_dir() or not (path / "training_state.pt").is_file():
            continue
        if not (path / "adapter_model.safetensors").is_file():
            continue
        try:
            int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        valid.append(path)
    return sorted(valid, key=lambda path: int(path.name.rsplit("-", 1)[1]))


def _save_checkpoint(output: Path, update: int, model, optimizer, scheduler, generated_tokens: int, stats: dict):
    import torch

    checkpoint = output / f"checkpoint-{update}"
    temporary = output / f".checkpoint-{update}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "generated_tokens": generated_tokens,
        "stats": stats,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
    }, temporary / "training_state.pt")
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temporary.replace(checkpoint)


def _run_config(args, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "rs-video-style-outcome-only-grpo",
        "reward": "strict_single_letter_exact_match",
        "evidence_rewards": "unavailable_in_reva_fit_A",
        "dataset_revision": C.dataset_revision(),
        "train_json": str(args.train_json.resolve()),
        "train_rows": len(rows),
        "train_rows_sha256": C.split_checksum(rows),
        "init_adapter": str(args.init_adapter.resolve()),
        "init_adapter_sha256": C.path_checksum(args.init_adapter),
        "reference_adapter": str(args.reference_adapter.resolve()),
        "reference_adapter_sha256": C.path_checksum(args.reference_adapter),
        "seed": args.seed,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "kl_beta": args.kl_beta,
        "clip_range": args.clip_range,
        "warmup_ratio": args.warmup_ratio,
        "max_tokens": args.max_tokens,
        "save_steps": args.save_steps,
    }


def _load_training(args, rows: list[dict[str, Any]], output: Path):
    import torch
    from transformers import get_cosine_schedule_with_warmup

    C.configure_attention_backend("sdpa", torch)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ReVA GRPO requires a CUDA GPU with bf16 support")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    valid_checkpoints = _checkpoint_paths(output)
    start_update = int(valid_checkpoints[-1].name.rsplit("-", 1)[1]) if valid_checkpoints else 0
    policy_adapter = valid_checkpoints[-1] if valid_checkpoints else args.init_adapter
    processor, model = _load_components(policy_adapter, True)
    _, reference = _load_components(args.reference_adapter, False)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    device = torch.device("cuda")
    model.to(device).eval()
    reference.to(device).eval()
    reference.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("initial adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    warmup_steps = math.ceil(len(rows) * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(rows)
    )
    generated_tokens = 0
    stats = {"reward_sum": 0.0, "correct": 0, "valid": 0, "groups": 0, "zero_std_groups": 0, "kl_sum": 0.0}
    if valid_checkpoints:
        state = torch.load(valid_checkpoints[-1] / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generated_tokens = int(state["generated_tokens"])
        stats.update(state["stats"])
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        random.setstate(state["python_rng_state"])
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    return torch, processor, model, reference, optimizer, scheduler, device, start_update, generated_tokens, stats, order


def preflight(args: argparse.Namespace) -> int:
    import torch

    rows = _rows(args.train_json)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ReVA GRPO preflight requires a CUDA GPU with bf16 support")
    C.configure_attention_backend("sdpa", torch)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor, model = _load_components(args.init_adapter, True)
    _, reference = _load_components(args.reference_adapter, False)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    device = torch.device("cuda")
    model.to(device).eval()
    reference.to(device).eval().requires_grad_(False)
    inputs, gold = _prompt(processor, rows[0], work=args.work, seed=args.seed, update=0, max_tokens=args.max_tokens)
    inputs = _move_inputs(inputs, device)
    completions = []
    texts = []
    rewards = []
    for _ in range(args.group_size):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, do_sample=True, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        completion = output_ids[0, inputs["input_ids"].shape[1]:]
        text = processor.tokenizer.decode(completion, skip_special_tokens=True)
        completions.append(completion)
        texts.append(text)
        rewards.append(answer_reward(text, gold)[0])
    if not completions or any(completion.numel() == 0 for completion in completions):
        raise RuntimeError("preflight generation returned an empty completion")
    grad_logp = _completion_logprob(model, inputs, completions[0], grad=True, temperature=args.temperature)
    ref_logp = _completion_logprob(reference, inputs, completions[0], grad=False, temperature=args.temperature)
    (-grad_logp.mean()).backward()
    if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("preflight backward pass produced no adapter gradients")
    print(json.dumps({
        "qa_id": rows[0]["qa_id"], "shown_gold": gold, "completions": texts,
        "rewards": rewards, "group_std": float(torch.tensor(rewards).std(unbiased=False)),
        "logprob_tokens": int(grad_logp.numel()), "reference_logprob_tokens": int(ref_logp.numel()),
        "gpu": torch.cuda.get_device_name(), "bf16": torch.cuda.is_bf16_supported(),
    }, indent=2))
    return 0


def train(args: argparse.Namespace) -> int:
    import torch

    global STOP_REQUESTED
    _install_stop_handler()
    rows = _rows(args.train_json)
    if args.group_size < 2 or args.max_new_tokens < 1 or args.save_steps < 1:
        raise ValueError("group_size must be >=2; max_new_tokens and save_steps must be positive")
    if args.learning_rate <= 0 or args.temperature <= 0 or args.kl_beta < 0 or not 0 < args.clip_range < 1:
        raise ValueError("invalid optimizer, sampling, KL, or clipping setting")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest = _run_config(args, rows)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError(f"run manifest mismatch at {manifest_path}; use a new output directory")
    elif any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty output without a matching manifest: {output}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    done = output / "DONE"
    if done.exists():
        print(f"training already complete: {output}")
        return 0

    (
        torch, processor, model, reference, optimizer, scheduler, device,
        start_update, generated_tokens, stats, order,
    ) = _load_training(args, rows, output)
    if start_update >= len(rows):
        raise RuntimeError("checkpoint update count exceeds the one-pass training schedule")
    started = time.monotonic()
    stop_file = Path(os.environ["REVA_STOP_FILE"]) if os.environ.get("REVA_STOP_FILE") else None
    for index in range(start_update, len(rows)):
        update = index + 1
        if STOP_REQUESTED or (stop_file and stop_file.exists()):
            _save_checkpoint(output, index, model, optimizer, scheduler, generated_tokens, stats)
            return 75
        row = rows[order[index]]
        prompt_inputs, shown_gold = _prompt(
            processor, row, work=args.work, seed=args.seed, update=index,
            max_tokens=args.max_tokens,
        )
        base = _move_inputs(prompt_inputs, device)
        completions = []
        rewards: list[float] = []
        valid_count = 0
        old_logps = []
        for _ in range(args.group_size):
            with torch.no_grad():
                output_ids = model.generate(
                    **base, do_sample=True, temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
            completion = output_ids[0, base["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(completion, skip_special_tokens=True)
            reward, valid = answer_reward(text, shown_gold)
            completions.append(completion)
            rewards.append(reward)
            valid_count += int(valid)
            generated_tokens += int(completion.numel())
            old_logps.append(
                _completion_logprob(model, base, completion, grad=False, temperature=args.temperature).detach()
            )

        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        reward_std = reward_tensor.std(unbiased=False)
        advantages = (
            (reward_tensor - reward_tensor.mean()) / (reward_std + 1e-6)
            if float(reward_std) > 0
            else torch.zeros_like(reward_tensor)
        )
        optimizer.zero_grad(set_to_none=True)
        step_kl = 0.0
        step_policy = 0.0
        for completion, old_logp, advantage in zip(completions, old_logps, advantages):
            new_logp = _completion_logprob(model, base, completion, grad=True, temperature=args.temperature)
            with torch.no_grad():
                ref_logp = _completion_logprob(reference, base, completion, grad=False, temperature=args.temperature)
            log_ratio = (new_logp - old_logp).clamp(-20, 20)
            ratio = torch.exp(log_ratio)
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            ref_minus_new = (ref_logp - new_logp).clamp(-20, 20)
            kl = (torch.exp(ref_minus_new) - ref_minus_new - 1.0).mean()
            ((policy_loss + args.kl_beta * kl) / args.group_size).backward()
            step_kl += float(kl.detach()) / args.group_size
            step_policy += float(policy_loss.detach()) / args.group_size
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        scheduler.step()

        stats["reward_sum"] += sum(rewards)
        stats["correct"] += int(sum(rewards))
        stats["valid"] += valid_count
        stats["groups"] += 1
        stats["zero_std_groups"] += int(float(reward_std) == 0.0)
        stats["kl_sum"] += step_kl
        if update % args.log_steps == 0 or update == 1:
            print(json.dumps({
                "update": update, "total_updates": len(rows), "qa_id": row["qa_id"],
                "reward_mean": sum(rewards) / args.group_size, "group_reward_std": float(reward_std),
                "group_correct": int(sum(rewards)), "valid_rate": valid_count / args.group_size,
                "zero_std_groups": stats["zero_std_groups"], "policy_loss": step_policy,
                "kl": step_kl, "lr": scheduler.get_last_lr()[0],
                "generated_tokens": generated_tokens,
                "elapsed_s": round(time.monotonic() - started, 1),
            }), flush=True)
        if update % args.save_steps == 0 or STOP_REQUESTED or (stop_file and stop_file.exists()):
            _save_checkpoint(output, update, model, optimizer, scheduler, generated_tokens, stats)
        if STOP_REQUESTED or (stop_file and stop_file.exists()):
            return 75

    final_adapter = output / "final_adapter"
    model.save_pretrained(final_adapter)
    processor.save_pretrained(final_adapter)
    elapsed = time.monotonic() - started
    summary = {
        "arm": "reva_answer_only_grpo",
        "updates": len(rows),
        "group_size": args.group_size,
        "generated_tokens": generated_tokens,
        "mean_reward": stats["reward_sum"] / max(stats["groups"] * args.group_size, 1),
        "completion_accuracy": stats["correct"] / max(stats["groups"] * args.group_size, 1),
        "valid_answer_rate": stats["valid"] / max(stats["groups"] * args.group_size, 1),
        "zero_std_group_rate": stats["zero_std_groups"] / max(stats["groups"], 1),
        "mean_sampled_kl": stats["kl_sum"] / max(stats["groups"], 1),
        "elapsed_s": elapsed,
        "gpu_hours": elapsed / 3600,
        "adapter": str(final_adapter.resolve()),
    }
    (output / "arm.json").write_text(json.dumps(summary, indent=2) + "\n")
    done.write_text("ok\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def _load_scores(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qa_id = row.get("qa_id")
            if not isinstance(qa_id, str) or qa_id in records or row.get("pred") not in C.LETTERS:
                raise ValueError(f"invalid or duplicate prediction at {path}:{line_number}")
            records[qa_id] = row
    return records


def compare(args: argparse.Namespace) -> int:
    split_path = C.work_path(SPLITS[args.split], args.work)
    gold_rows = json.loads(split_path.read_text())["QA"]
    gold = {row["qa_id"]: row for row in gold_rows}
    baseline = _load_scores(args.baseline)
    candidate = _load_scores(args.candidate)
    expected = set(gold)
    if set(baseline) != expected or set(candidate) != expected:
        raise ValueError("paired comparison requires complete, identical split IDs in both score files")
    records = []
    for qa_id, row in gold.items():
        a0 = baseline[qa_id]["pred"] == row["correct_answer"]
        rl = candidate[qa_id]["pred"] == row["correct_answer"]
        records.append({
            "qa_id": qa_id, "video_path": row["video_path"], "task": row["task"],
            "category": row["category"], "gold": row["correct_answer"],
            "a0": baseline[qa_id]["pred"], "rl": candidate[qa_id]["pred"],
            "a0_ok": a0, "rl_ok": rl,
        })
    fixes = sum(not row["a0_ok"] and row["rl_ok"] for row in records)
    regressions = sum(row["a0_ok"] and not row["rl_ok"] for row in records)
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(group),
            "a0_correct": sum(row["a0_ok"] for row in group),
            "rl_correct": sum(row["rl_ok"] for row in group),
            "a0_accuracy": sum(row["a0_ok"] for row in group) / len(group) if group else 0.0,
            "rl_accuracy": sum(row["rl_ok"] for row in group) / len(group) if group else 0.0,
            "fixes": sum(not row["a0_ok"] and row["rl_ok"] for row in group),
            "regressions": sum(row["a0_ok"] and not row["rl_ok"] for row in group),
        }
    by_task = {
        task: summarize([row for row in records if row["task"] == task])
        for task in sorted({row["task"] for row in records})
    }
    result = {
        "split": args.split,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "overall": summarize(records),
        "net_fixes": fixes - regressions,
        "by_task": by_task,
        "paired": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "paired"}, indent=2))
    return 0


def add_train_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--work", type=Path, default=Path(os.environ.get("WORK", ".")))
    command.add_argument("--train-json", type=Path, required=True)
    command.add_argument("--init-adapter", type=Path, required=True)
    command.add_argument("--reference-adapter", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20261023)
    command.add_argument("--group-size", type=int, default=4)
    command.add_argument("--temperature", type=float, default=0.7)
    command.add_argument("--max-new-tokens", type=int, default=4)
    command.add_argument("--learning-rate", type=float, default=1e-5)
    command.add_argument("--kl-beta", type=float, default=0.03)
    command.add_argument("--clip-range", type=float, default=0.2)
    command.add_argument("--warmup-ratio", type=float, default=0.03)
    command.add_argument("--max-tokens", type=int, default=12000)
    command.add_argument("--save-steps", type=int, default=500)
    command.add_argument("--log-steps", type=int, default=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("self-check")
    pre = sub.add_parser("preflight")
    add_train_options(pre)
    run = sub.add_parser("train")
    add_train_options(run)
    comp = sub.add_parser("compare")
    comp.add_argument("--work", type=Path, default=Path(os.environ.get("WORK", ".")))
    comp.add_argument("--split", choices=sorted(SPLITS), required=True)
    comp.add_argument("--baseline", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-check":
        return self_check()
    if args.command == "preflight":
        return preflight(args)
    if args.command == "train":
        return train(args)
    if args.command == "compare":
        return compare(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
