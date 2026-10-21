#!/usr/bin/env python3
"""Extract machine-readable training diagnostics from a Phase 6 train log."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path
from typing import Any


DICT_RE = re.compile(r"\{\s*'loss':.*\}")
BENCHMARK_RE = re.compile(r"benchmark steps=(\d+) mean_step_seconds=([0-9.eE+-]+) estimated_train_hours=([0-9.eE+-]+)")
RUNTIME_RE = re.compile(r"'train_runtime':\s*'([^']+)'|\"train_runtime\":\s*\"([^\"]+)\"")
TRAIN_LOSS_RE = re.compile(r"'train_loss':\s*'([^']+)'|\"train_loss\":\s*\"([^\"]+)\"")


def parse(path: Path) -> dict[str, Any]:
    records: list[dict[str, float]] = []
    benchmarks: list[dict[str, float]] = []
    runtime = None
    train_loss = None
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        match = DICT_RE.search(line)
        if match:
            try:
                value = ast.literal_eval(match.group(0))
            except (SyntaxError, ValueError):
                value = None
            if isinstance(value, dict) and "loss" in value and "learning_rate" in value:
                try:
                    records.append({key: float(value[key]) for key in ("loss", "grad_norm", "learning_rate", "epoch") if key in value})
                except (TypeError, ValueError):
                    pass
        match = BENCHMARK_RE.search(line)
        if match:
            benchmarks.append({"steps": int(match.group(1)), "mean_step_seconds": float(match.group(2)), "estimated_train_hours": float(match.group(3))})
        match = RUNTIME_RE.search(line)
        if match:
            runtime = float(next(item for item in match.groups() if item is not None))
        match = TRAIN_LOSS_RE.search(line)
        if match:
            train_loss = float(next(item for item in match.groups() if item is not None))
    result: dict[str, Any] = {
        "schema_version": 1,
        "log": str(path),
        "records": records,
        "benchmarks": benchmarks,
        "logged_steps": len(records),
        "train_runtime_seconds": runtime,
        "train_loss": train_loss,
    }
    if records:
        losses = [record["loss"] for record in records]
        lrs = [record["learning_rate"] for record in records]
        grads = [record["grad_norm"] for record in records if "grad_norm" in record]
        result["loss_summary"] = {
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
            "first_half_mean": statistics.mean(losses[: max(1, len(losses) // 2)]),
            "last_half_mean": statistics.mean(losses[len(losses) // 2 :]),
        }
        result["learning_rate_summary"] = {
            "first": lrs[0],
            "last": lrs[-1],
            "maximum": max(lrs),
            "minimum": min(lrs),
        }
        if grads:
            result["grad_norm_summary"] = {
                "first": grads[0],
                "last": grads[-1],
                "minimum": min(grads),
                "maximum": max(grads),
                "mean": statistics.mean(grads),
            }
        result["loss_trend"] = "lower_last_half" if result["loss_summary"]["last_half_mean"] < result["loss_summary"]["first_half_mean"] else "not_lower_last_half"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--arm")
    args = parser.parse_args()
    result = parse(args.log)
    if args.slurm_job_id:
        result["slurm_job_id"] = args.slurm_job_id
    if args.arm:
        result["arm"] = args.arm
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
