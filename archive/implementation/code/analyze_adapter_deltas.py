"""Compute CPU adapter parameter deltas without loading the base model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors.torch import load_file


def _state(path: Path) -> dict:
    return load_file(str(path / "adapter_model.safetensors"), device="cpu")


def _norm(values) -> float:
    return math.sqrt(sum(float(value.float().pow(2).sum()) for value in values))


def _summary(state: dict, reference: dict | None = None) -> dict:
    if reference is None:
        delta = list(state.values())
        shared = len(state)
    else:
        shared_keys = sorted(set(state) & set(reference))
        delta = [state[key].float() - reference[key].float() for key in shared_keys]
        shared = len(shared_keys)
    return {
        "tensor_keys": len(state),
        "shared_keys": shared,
        "l2_norm": _norm(delta),
        "max_abs": max((float(value.float().abs().max()) for value in delta), default=0.0),
        "numel": sum(value.numel() for value in delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    stage_a = Path(args.stage_a)
    stage_state = _state(stage_a)
    results = {}
    for spec in args.arm:
        name, raw_path = spec.split("=", 1)
        root = Path(raw_path)
        root_state = _state(root)
        children = {
            child.name: _state(child)
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / "adapter_model.safetensors").is_file()
        }
        results[name] = {
            "root_default_vs_stageA": _summary(root_state, stage_state),
            "children_vs_zero": {child: _summary(state) for child, state in children.items()},
        }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
