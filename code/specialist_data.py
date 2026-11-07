"""Task filters, motion-hint checks, and A100 specialist config contracts.

The helpers are pure so they can run on a login node. They do not read hidden
test labels and they do not invent split sizes: every count comes from a file
the caller actually opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

import common as C
import mirror_answers as M

B1_TASKS = ("Perspective and Viewpoint", "Change Detection")
B2_TASKS = ("Temporal Grounding",)
MIRROR_TASKS = M.MIRROR_TASKS
HINTS_REL = "data/camera_motion_hints.json"
INIT_ADAPTER = "runs/stageA_q3vl8b_r16/final_adapter"
TRAIN_JSON = "data/splits/fit_A.json"

_LEAKAGE = re.compile(
    r"correct[_\s-]?answer|\bthe answer is\b|\boption [abcd]\b|\bqa_id\b|\btrain_\d+\b|\bval[qt]?_\d+\b",
    re.IGNORECASE,
)
_HINT_WORDS = {
    "camera",
    "motion",
    "unavailable",
    "hovering",
    "stationary",
    "moving",
    "forward",
    "backward",
    "panning",
    "left",
    "right",
    "tilting",
    "down",
    "up",
    "rapid",
    "slow",
}
_PINNED_TRAIN = {
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "scheduler": "cosine",
    "grad_accum": 16,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "max_tokens": 12000,
    "dataloader_workers": 8,
    "save_total_limit": 2,
}
_CONTRACTS = {
    "smoke_b1": {
        "run_name": "smoke_b1_hint16",
        "tasks": B1_TASKS,
        "frame_budget": 16,
        "hints": True,
        "preserve_frame_budget": False,
        "mirror": False,
        "kernel_smoke": True,
        "epochs": 1,
    },
    "smoke_b2": {
        "run_name": "smoke_b2_frames32",
        "tasks": B2_TASKS,
        "frame_budget": 32,
        "hints": False,
        "preserve_frame_budget": True,
        "mirror": False,
        "kernel_smoke": True,
        "epochs": 1,
    },
    "b1": {
        "run_name": "b1_hint16",
        "tasks": B1_TASKS,
        "frame_budget": 16,
        "hints": True,
        "preserve_frame_budget": False,
        "mirror": False,
        "kernel_smoke": False,
        "epochs": 1,
        "requires_smoke_run": "smoke_b1_hint16",
        "requires_smoke_contract": "smoke_b1",
    },
    "b2": {
        "run_name": "b2_frames32",
        "tasks": B2_TASKS,
        "frame_budget": 32,
        "hints": False,
        "preserve_frame_budget": True,
        "mirror": False,
        "kernel_smoke": False,
        "epochs": 1,
        "requires_smoke_run": "smoke_b2_frames32",
        "requires_smoke_contract": "smoke_b2",
    },
    "b1_mirror": {
        "run_name": "b1_hint16_mirror",
        "tasks": MIRROR_TASKS,
        "frame_budget": 16,
        "hints": True,
        "preserve_frame_budget": False,
        "mirror": True,
        "kernel_smoke": False,
        "epochs": 1,
        "requires_smoke_run": "smoke_b1_hint16",
        "requires_smoke_contract": "smoke_b1",
    },
}


def load_qa_payload(path: Path) -> dict:
    """Load a split object and require a QA list."""

    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("QA"), list):
        raise ValueError(f"{path} must be a JSON object with a QA list")
    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{path} metadata must be an object when present")
    return payload


def assert_train_only_rows(rows: list[dict]) -> None:
    if not rows or any(not str(row.get("qa_id", "")).startswith("train_") for row in rows):
        raise ValueError("training rows must be non-empty and use train_* qa ids")


def filter_rows_by_tasks(rows: list[dict], tasks: list[str] | tuple[str, ...]) -> list[dict]:
    if not isinstance(tasks, (list, tuple)) or not tasks or any(not isinstance(task, str) or not task.strip() for task in tasks):
        raise ValueError("train_tasks must be a non-empty list of task names")
    allowed = set(tasks)
    selected = [row for row in rows if row.get("task") in allowed]
    if not selected:
        raise ValueError(f"train_tasks matched no rows: {list(tasks)}")
    return selected


def normalize_frame_budget(value) -> int | None:
    if value is None:
        return None
    budget = int(value)
    if budget < 1:
        raise ValueError("frame_budget must be positive")
    return budget


def load_hint_table(payload: object) -> dict:
    """Accept a flat video-path map and refuse an unaudited wrapper document."""

    if not isinstance(payload, dict) or not payload:
        raise ValueError("camera motion hints must be a non-empty object keyed by video path")
    wrapper_keys = {"hints", "videos", "camera_motion"}
    if any(key in payload for key in wrapper_keys) and set(payload).issubset(wrapper_keys | {"metadata", "schema", "version"}):
        raise ValueError(
            "camera motion hints use a wrapped schema this checkout does not audit; "
            "reconcile the HPC file before training"
        )
    table: dict = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key or key.startswith(("train_", "val_", "test_")) or key in {"QA", "metadata"}:
            raise ValueError(f"hint key is not a video path: {key!r}")
        if not isinstance(value, (str, dict)):
            raise ValueError(f"hint for {key!r} must be a string or object")
        table[key] = value
    return table


def coerce_motion_text(value, formatter=None) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, dict):
        if formatter is None:
            raise RuntimeError(
                "motion hint value is an object and common.motion_hint_text is absent; "
                "reconcile the HPC helper before use"
            )
        formatted = formatter(value)
        if not isinstance(formatted, str):
            raise RuntimeError("motion_hint_text must return a string")
        text = formatted.strip()
    else:
        raise TypeError(f"motion hint must be a string or object, got {type(value).__name__}")
    if not text:
        raise ValueError("motion hint text is empty")
    return text


def validate_hint_text(text: str, *, strict_vocabulary: bool) -> None:
    if _LEAKAGE.search(text):
        raise ValueError("motion hint text contains answer or QA-id language")
    if text.strip() in C.LETTERS:
        raise ValueError("motion hint text is a bare answer letter")
    if not strict_vocabulary:
        return
    if text == "Camera motion: unavailable":
        return
    if not text.startswith("Camera motion:"):
        raise ValueError("strict motion hints must use the audited 'Camera motion:' prefix")
    body = re.sub(r"\d+(?:\.\d+)?s\b", " ", text.split(":", 1)[1])
    words = re.findall(r"[A-Za-z]+", body)
    unexpected = sorted({word.lower() for word in words if word.lower() not in _HINT_WORDS})
    if unexpected:
        raise ValueError(f"motion hint has words outside the pixel-motion vocabulary: {unexpected}")


def motion_context_for(table, video_path: str, *, required: bool, strict_vocabulary: bool, formatter=None) -> str | None:
    if not table:
        if required:
            raise RuntimeError("motion hints are required but no hint table is loaded")
        return None
    if video_path not in table:
        if required:
            raise KeyError(f"motion hint missing for training video {video_path!r}")
        return None
    text = coerce_motion_text(table[video_path], formatter)
    if required or strict_vocabulary:
        validate_hint_text(text, strict_vocabulary=strict_vocabulary)
    return text


def assert_hint_coverage(table: dict, rows: list[dict], *, strict_vocabulary: bool, formatter=None) -> dict:
    missing: list[str] = []
    checked = 0
    for row in rows:
        video = row.get("video_path")
        if not isinstance(video, str):
            raise ValueError("training row is missing video_path")
        if video not in table:
            missing.append(video)
            continue
        motion_context_for(
            table,
            video,
            required=True,
            strict_vocabulary=strict_vocabulary,
            formatter=formatter,
        )
        checked += 1
    if missing:
        sample = sorted(set(missing))[:8]
        raise RuntimeError(f"motion hints missing for {len(set(missing))} training videos, for example {sample}")
    return {"videos_checked": len({row['video_path'] for row in rows}), "hint_values_checked": checked}


def guard_token_budget(config: dict, n_tokens: int, qa_id: str) -> None:
    limit = int(config["train"]["max_tokens"])
    if n_tokens <= limit or not config.get("preserve_frame_budget"):
        return
    raise RuntimeError(
        f"sample {qa_id} has {n_tokens} tokens, above max_tokens={limit}, at "
        f"frame_budget={config.get('frame_budget')}. Refusing to drop frames because "
        "preserve_frame_budget is set. Reconcile HPC video_size_for or the token budget "
        "before training this specialist."
    )


def _ids(rows: list[dict]) -> set[str]:
    return {str(row["qa_id"]) for row in rows}


def _videos(rows: list[dict]) -> set[str]:
    return {str(row["video_path"]) for row in rows}


def _summary(rows: list[dict]) -> dict:
    counts = Counter(str(row.get("task")) for row in rows)
    return {
        "n_qa": len(rows),
        "n_videos": len(_videos(rows)),
        "task_counts": dict(sorted(counts.items())),
    }


def eligibility_report(paths: dict[str, Path]) -> dict:
    """Measure holdout disjointness from the split files that are present."""

    required = ("fit_A", "valQ", "valT", "full_train", "val")
    missing = [name for name in required if name not in paths or not paths[name].is_file()]
    report: dict = {"missing_files": missing, "files": {}}
    if missing:
        report["eligible_for_specialist_holdout"] = False
        report["stageB_family_requires_official_val"] = True
        report["blocked_reason"] = "required split file is absent: " + ", ".join(missing)
        return report

    payloads = {name: load_qa_payload(paths[name]) for name in required}
    for name in ("fit_A", "valQ", "valT", "full_train"):
        metadata = payloads[name].get("metadata")
        if not isinstance(metadata, dict) or metadata.get("split") != name:
            raise ValueError(f"{name} metadata.split must be {name!r}")
    fit_rows = payloads["fit_A"]["QA"]
    assert_train_only_rows(fit_rows)
    rows = {name: payloads[name]["QA"] for name in required}
    fit_ids = _ids(rows["fit_A"])
    fit_videos = _videos(rows["fit_A"])
    overlaps = {
        "fit_A_valQ_qa": len(fit_ids & _ids(rows["valQ"])),
        "fit_A_valT_qa": len(fit_ids & _ids(rows["valT"])),
        "fit_A_val_qa": len(fit_ids & _ids(rows["val"])),
        "fit_A_valT_videos": len(fit_videos & _videos(rows["valT"])),
        "fit_A_valQ_videos": len(fit_videos & _videos(rows["valQ"])),
        "fit_A_val_videos": len(fit_videos & _videos(rows["val"])),
        "full_train_valQ_qa": len(_ids(rows["full_train"]) & _ids(rows["valQ"])),
        "full_train_valT_qa": len(_ids(rows["full_train"]) & _ids(rows["valT"])),
        "full_train_val_qa": len(_ids(rows["full_train"]) & _ids(rows["val"])),
    }
    specialist_ok = (
        overlaps["fit_A_valQ_qa"] == 0
        and overlaps["fit_A_valT_qa"] == 0
        and overlaps["fit_A_val_qa"] == 0
        and overlaps["fit_A_valT_videos"] == 0
    )
    report.update(
        {
            "files": {name: _summary(rows[name]) for name in required},
            "fit_A_task_rows": {
                "Perspective and Viewpoint": sum(row.get("task") == "Perspective and Viewpoint" for row in fit_rows),
                "Change Detection": sum(row.get("task") == "Change Detection" for row in fit_rows),
                "Temporal Grounding": sum(row.get("task") == "Temporal Grounding" for row in fit_rows),
                "Geometric Relation": sum(row.get("task") == "Geometric Relation" for row in fit_rows),
            },
            "overlaps": overlaps,
            "valQ_question_held_out": overlaps["fit_A_valQ_qa"] == 0,
            "valQ_video_held_out": overlaps["fit_A_valQ_videos"] == 0,
            "valT_video_held_out": overlaps["fit_A_valT_videos"] == 0 and overlaps["fit_A_valT_qa"] == 0,
            "official_val_label_held_out_from_fit_A": overlaps["fit_A_val_qa"] == 0,
            "stageB_family_leaks_into_valQ_or_valT": overlaps["full_train_valQ_qa"] > 0 or overlaps["full_train_valT_qa"] > 0,
            "stageB_family_leaks_into_official_val": overlaps["full_train_val_qa"] > 0,
            "stageB_family_requires_official_val": True,
            "eligible_for_specialist_holdout": specialist_ok,
            "blocked_reason": "" if specialist_ok else "fit_A overlaps a holdout QA id or a valT video",
        }
    )
    return report


def _close(actual, expected) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)
    return actual == expected


def validate_run_config(config: dict, *, allow_unresolved: bool = False) -> str:
    """Fail closed unless the config is one of the audited specialist contracts."""

    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    contract = config.get("specialist_contract")
    if contract not in _CONTRACTS:
        raise ValueError(f"unsupported specialist_contract: {contract!r}")
    spec = _CONTRACTS[contract]
    if config.get("run_name") != spec["run_name"]:
        raise ValueError(f"{contract} run_name must be {spec['run_name']}")
    if config.get("train_json") != TRAIN_JSON or config.get("require_split") != "fit_A":
        raise ValueError("specialists must train on data/splits/fit_A.json with require_split fit_A")
    if config.get("init_adapter") != INIT_ADAPTER or config.get("require_init_adapter") is not True:
        raise ValueError("specialists must continue from the Stage A adapter")
    if config.get("seed") != 42 or config.get("attn_implementation") != "sdpa":
        raise ValueError("specialists must keep Stage A seed 42 and sdpa attention")
    if config.get("lora") != {"r": 16, "alpha": 32, "dropout": 0.05}:
        raise ValueError("specialists must keep the Stage A language LoRA recipe")
    if config.get("temporal_stream", {}).get("enabled") or config.get("visual_encoder", {}).get("enabled"):
        raise ValueError("specialist contracts do not enable a new stream or vision adapter")
    if config.get("visual_merger", {}).get("enabled"):
        raise ValueError("specialist contracts do not enable a merger adapter")
    family = config.get("model_family", "qwen3_vl")
    if family not in {"qwen3_vl", "qwen3-vl", "qwen3vl"}:
        raise ValueError(f"unsupported model_family for specialists: {family!r}")
    output = str(config.get("output_dir", ""))
    parts = tuple(Path(output).parts)
    if parts[:3] != ("runs", "specialists", spec["run_name"]) or len(parts) != 4:
        raise ValueError("output_dir must be runs/specialists/<run_name>/<stamp>")
    if parts[3] == "UNRESOLVED":
        if not allow_unresolved:
            raise ValueError("refusing to train into an UNRESOLVED output_dir; launch the A100 sbatch wrapper")
    elif not parts[3] or parts[3] in {".", ".."}:
        raise ValueError("output stamp is empty")
    if list(config.get("train_tasks") or []) != list(spec["tasks"]):
        raise ValueError(f"{contract} train_tasks must be {list(spec['tasks'])}")
    if config.get("frame_budget") != spec["frame_budget"]:
        raise ValueError(f"{contract} frame_budget must be {spec['frame_budget']}")
    if bool(config.get("preserve_frame_budget", False)) != spec["preserve_frame_budget"]:
        raise ValueError("preserve_frame_budget does not match the contract")
    if bool(config.get("motion_hints_required", False)) != spec["hints"]:
        raise ValueError("motion_hints_required does not match the contract")
    if spec["hints"]:
        if config.get("motion_hints") != HINTS_REL or config.get("strict_hint_vocabulary") is not True:
            raise ValueError("hint contracts require data/camera_motion_hints.json and strict vocabulary")
    elif config.get("motion_hints") or config.get("motion_hints_required"):
        raise ValueError("this contract must not load motion hints")
    if bool(config.get("mirror_safe", False)) != spec["mirror"]:
        raise ValueError("mirror_safe does not match the contract")
    if spec["mirror"]:
        if not _close(config.get("mirror_probability"), 0.5):
            raise ValueError("mirror ablation probability must be 0.5")
    elif config.get("mirror_probability") not in (None, 0, 0.0):
        raise ValueError("non-mirror contracts must not set a mirror probability")
    train = config.get("train")
    if not isinstance(train, dict):
        raise ValueError("train block is missing")
    for key, expected in _PINNED_TRAIN.items():
        if not _close(train.get(key), expected):
            raise ValueError(f"train.{key} must stay at the audited Stage A value {expected!r}")
    if not _close(train.get("lr"), 2.0e-4):
        raise ValueError("train.lr must stay at the audited Stage A value 2.0e-4")
    if train.get("explicit_optimizer"):
        raise ValueError("specialists use the Stage A Trainer AdamW path")
    if int(train.get("epochs", 0)) != spec["epochs"]:
        raise ValueError(f"{contract} epochs must be {spec['epochs']}")
    if bool(config.get("kernel_smoke", False)) != spec["kernel_smoke"]:
        raise ValueError("kernel_smoke flag does not match the contract")
    if spec["kernel_smoke"]:
        steps = int(train.get("max_steps", -1))
        if not 5 <= steps <= 20:
            raise ValueError("kernel smoke max_steps must be from 5 through 20")
        if int(train.get("save_steps", 0)) > steps or int(train.get("logging_steps", 0)) != 1:
            raise ValueError("kernel smoke must log every step and checkpoint within the step budget")
        if config.get("require_smoke_marker"):
            raise ValueError("a smoke config must not require an earlier smoke marker")
    else:
        if int(train.get("max_steps", -1)) != -1:
            raise ValueError("one-epoch specialists must not set a positive max_steps")
        if config.get("require_smoke_marker") is not True:
            raise ValueError("one-epoch specialists must require a smoke marker")
        if config.get("requires_smoke_run") != spec["requires_smoke_run"]:
            raise ValueError("requires_smoke_run does not match the contract")
        if int(train.get("logging_steps", 0)) != 10 or int(train.get("save_steps", 0)) != 50:
            raise ValueError("one-epoch specialists keep logging_steps 10 and save_steps 50")
    if config.get("isolate_output") is not True:
        raise ValueError("specialist configs must set isolate_output")
    return contract


def find_smoke_marker(work: Path, run_name: str, expected_contract: str) -> Path:
    root = work / "runs" / "specialists" / run_name
    found: list[Path] = []
    if root.is_dir():
        for marker in root.glob("*/SMOKE_OK"):
            config_path = marker.parent / "resolved_config.yaml"
            snapshot = marker.parent / "config_snapshot.yaml"
            source = config_path if config_path.is_file() else snapshot
            if not source.is_file() or not (marker.parent / "KERNEL_SMOKE_STEP_OK").is_file():
                continue
            if not (marker.parent / "DONE").is_file():
                continue
            config = yaml.safe_load(source.read_text())
            if config.get("specialist_contract") != expected_contract:
                continue
            found.append(marker)
    if not found:
        raise FileNotFoundError(
            f"no completed {expected_contract} smoke marker under {root}. "
            "Run the matching A100 kernel smoke before the one-epoch specialist."
        )
    return max(found, key=lambda path: path.stat().st_mtime)


def materialize_config(template: Path, dest: Path, output_dir: Path, work: Path) -> dict:
    if dest.exists():
        raise FileExistsError(dest)
    config = yaml.safe_load(template.read_text())
    relative = output_dir.resolve().relative_to(work.resolve()).as_posix()
    config["output_dir"] = relative
    config["template_sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()
    validate_run_config(config, allow_unresolved=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    temporary.replace(dest)
    return config


def _work_from_args(args) -> Path:
    value = args.work or os.environ.get("WORK")
    if not value:
        raise SystemExit("WORK or --work is required")
    return Path(value).expanduser().resolve()


def _split_paths(work: Path) -> dict[str, Path]:
    return {
        "fit_A": work / "data" / "splits" / "fit_A.json",
        "valQ": work / "data" / "splits" / "valQ.json",
        "valT": work / "data" / "splits" / "valT.json",
        "full_train": work / "data" / "splits" / "full_train.json",
        "val": work / "data" / "ReVA" / "val.json",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A100 specialist data and config checks")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("config")
    validate.add_argument("--allow-unresolved", action="store_true")

    materialize = sub.add_parser("materialize-config")
    materialize.add_argument("template")
    materialize.add_argument("dest")
    materialize.add_argument("output_dir")
    materialize.add_argument("--work")

    audit = sub.add_parser("audit-eligibility")
    audit.add_argument("--work")
    audit.add_argument("--out", required=True)

    hints = sub.add_parser("audit-hints")
    hints.add_argument("--work")
    hints.add_argument("--hints", default=HINTS_REL)
    hints.add_argument("--tasks", nargs="+", required=True)
    hints.add_argument("--out", required=True)

    marker = sub.add_parser("assert-smoke-marker")
    marker.add_argument("--work")
    marker.add_argument("--run-name", required=True)
    marker.add_argument("--contract", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate-config":
        config = yaml.safe_load(Path(args.config).read_text())
        print(validate_run_config(config, allow_unresolved=args.allow_unresolved))
        return 0
    if args.command == "materialize-config":
        work = _work_from_args(args)
        config = materialize_config(Path(args.template), Path(args.dest), Path(args.output_dir), work)
        print(config["output_dir"])
        return 0
    if args.command == "audit-eligibility":
        work = _work_from_args(args)
        report = eligibility_report(_split_paths(work))
        _write_json(Path(args.out), report)
        print(json.dumps({key: report[key] for key in ("missing_files", "eligible_for_specialist_holdout", "blocked_reason", "overlaps") if key in report}, sort_keys=True))
        return 0 if report.get("eligible_for_specialist_holdout") else 2
    if args.command == "audit-hints":
        work = _work_from_args(args)
        split_path = work / TRAIN_JSON
        hint_path = work / args.hints
        if not split_path.is_file() or not hint_path.is_file():
            raise SystemExit(f"missing {split_path} or {hint_path}")
        rows = filter_rows_by_tasks(load_qa_payload(split_path)["QA"], args.tasks)
        assert_train_only_rows(rows)
        table = load_hint_table(json.loads(hint_path.read_text()))
        formatter = getattr(C, "motion_hint_text", None)
        coverage = assert_hint_coverage(table, rows, strict_vocabulary=True, formatter=formatter)
        report = {"hints": str(hint_path), "tasks": list(args.tasks), "train_rows": len(rows), **coverage}
        _write_json(Path(args.out), report)
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.command == "assert-smoke-marker":
        work = _work_from_args(args)
        print(find_smoke_marker(work, args.run_name, args.contract))
        return 0
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
