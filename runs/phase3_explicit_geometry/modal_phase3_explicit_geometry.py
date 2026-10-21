"""Run the approved Phase 3 pilot on one Modal A100.

The local entrypoint refuses to submit without an explicitly named Modal
secret containing ``REPLICATE_API_TOKEN``.  The remote function checks that
token before staging or invoking any detector, VLM, or Replicate model call.
All remote inputs are pinned to the new Phase 3 selection manifest and the
existing Stage A adapter; returned artifacts are copied back into this run
directory only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
from typing import Any

import modal


def _repository_root(module_path: str | os.PathLike[str], *, container_root: Path = Path("/root/ReVA")) -> Path:
    """Resolve the repo for both the local file and Modal's flattened module."""

    path = Path(module_path).expanduser().resolve()
    # Locally this file is runs/phase3_explicit_geometry/<name>.py.  Modal
    # uploads the module as /root/<name>.py, where parents[2] does not exist.
    if len(path.parents) > 2:
        local_candidate = path.parents[2]
        if (local_candidate / "code").is_dir() and (local_candidate / "runs").is_dir():
            return local_candidate
    if (container_root / "code").is_dir() and (container_root / "runs").is_dir():
        return container_root
    raise RuntimeError(f"cannot resolve ReVA repository root for module path: {path}")


def _self_check_root_resolution(root: Path) -> None:
    """Exercise both path shapes without touching Modal or model services."""

    local = Path(__file__).resolve()
    local_expected = local.parents[2] if len(local.parents) > 2 else root
    assert _repository_root(local, container_root=local_expected) == local_expected
    flat = Path("/root/modal_phase3_explicit_geometry.py")
    assert _repository_root(flat, container_root=root) == root


ROOT = _repository_root(__file__, container_root=Path("/root/ReVA"))
_self_check_root_resolution(ROOT)
WORK = "/root/ReVA"
RUN = "runs/phase3_explicit_geometry"
SELECTION = f"{RUN}/selection_manifest.json"
SPLITS = ("valQ", "valT")
DATASET_REPO = "ReVA-Benchmark/ReVA"
DATASET_REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"

# Keep the Modal object graph identical when the local module is imported by
# the CLI and when Modal hydrates the uploaded module.  The secret value is
# resolved only inside the remote container and is never read or printed here.
APPROVED_SECRET_NAME = "reva-phase3-replicate"
SECRETS = [modal.Secret.from_name(APPROVED_SECRET_NAME)]


def _self_check_secret_setup() -> None:
    assert APPROVED_SECRET_NAME == "reva-phase3-replicate"
    assert len(SECRETS) == 1


_self_check_secret_setup()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "transformers==5.19.0",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "huggingface-hub==2.2.0",
        "av==19.0.1",
        "numpy",
        "pillow",
        "pyyaml",
        "tqdm",
        "replicate",
    )
    .env({"WORK": WORK, "HF_HOME": "/root/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
    .add_local_dir(str(ROOT / "data" / "splits"), f"{WORK}/data/splits", copy=True)
    .add_local_dir(str(ROOT / RUN), f"{WORK}/{RUN}", copy=True)
    .add_local_dir(
        str(ROOT / "runs" / "stageA_q3vl8b_r16" / "final_adapter"),
        f"{WORK}/runs/stageA_q3vl8b_r16/final_adapter",
        copy=True,
    )
)

app = modal.App("reva-phase3-explicit-geometry", image=image)


def _scope_selection(work: Path, scope: str) -> tuple[Path, Path]:
    """Write report and execution selections with tranche reuse explicit."""

    source = json.loads((work / SELECTION).read_text())
    if scope == "full":
        selected = source["selected_questions"]
        tranche = source["tranche"]
        tranche_ids = set(qa_id for ids in tranche["ids"].values() for qa_id in ids)
        execution_selected = {
            split: [row for row in selected[split] if row["qa_id"] not in tranche_ids]
            for split in SPLITS
        }
    else:
        tranche_ids = source["tranche"]["ids"]
        selected = {
            split: [row for row in source["selected_questions"][split] if row["qa_id"] in set(tranche_ids[split])]
            for split in SPLITS
        }
        tranche = {
            "questions_per_split": 2,
            "ids": {split: [row["qa_id"] for row in selected[split]] for split in SPLITS},
            "composition": source["tranche"]["composition"],
        }
        execution_selected = selected
    if any(len(selected[split]) != (30 if scope == "full" else 2) for split in SPLITS):
        raise RuntimeError(f"{scope} selection does not have the expected per-split question count")
    payload = dict(source)
    payload["selected_questions"] = selected
    payload["tranche"] = tranche
    if scope == "full":
        tranche_root = work / RUN / "tranche" / "sam3"
        prior_paid = 0
        for candidate in (tranche_root / "sam3_api_manifest.json", tranche_root / "sam3_api_manifest.partial.json"):
            if candidate.is_file():
                try:
                    prior_paid = max(prior_paid, int(json.loads(candidate.read_text()).get("paid_attempt_count_total", 0)))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
        payload["cost_gate"] = dict(source["cost_gate"])
        payload["cost_gate"]["prior_paid_attempt_count"] = prior_paid
    destination = work / RUN / scope / "scope_selection.json"
    execution_destination = work / RUN / scope / "execution_selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload["selected_questions"] = selected
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    execution_payload = dict(payload)
    execution_payload["selected_questions"] = execution_selected
    execution_destination.write_text(json.dumps(execution_payload, indent=2) + "\n")
    return destination, execution_destination


def _prepare_data(work: Path, scope: str) -> tuple[dict[str, Any], Path, Path]:
    """Download only selected released clips and create the local manifest."""

    import av
    from huggingface_hub import hf_hub_download, snapshot_download

    scope_selection, execution_selection = _scope_selection(work, scope)
    payload = json.loads(execution_selection.read_text())
    rows = [row for split in SPLITS for row in payload["selected_questions"][split]]
    split_rows = {
        split: json.loads((work / "data" / "splits" / f"{split}.json").read_text())["QA"]
        for split in SPLITS
    }
    # The copied split files define the selected rows; the released train
    # index confirms their exact content at the pinned revision.
    dataset_root = work / "data" / "ReVA"
    dataset_root.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename="train.json",
        revision=DATASET_REVISION,
        local_dir=str(dataset_root),
    )
    train_by_id = {row["qa_id"]: row for row in json.loads((dataset_root / "train.json").read_text())["QA"]}
    for split in SPLITS:
        available = {row["qa_id"]: row for row in split_rows[split]}
        for selected in payload["selected_questions"][split]:
            if train_by_id.get(selected["qa_id"]) != available.get(selected["qa_id"]):
                raise RuntimeError(f"selected {split} row differs from pinned train.json: {selected['qa_id']}")
    paths = sorted({row["video_path"] for row in rows})
    if any(PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts for path in paths):
        raise RuntimeError("unsafe selected video path")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(dataset_root),
        allow_patterns=paths,
        max_workers=8,
    )
    manifest: dict[str, dict[str, Any]] = {}
    for relative in paths:
        path = dataset_root / PurePosixPath(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            try:
                fps = float(stream.average_rate or stream.base_rate or 30.0)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 30.0
            timestamps = []
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is not None and frame.time_base is not None:
                    timestamps.append(float(frame.pts * frame.time_base))
                else:
                    timestamps.append(index / max(fps, 1e-6))
            if not timestamps:
                raise RuntimeError(f"selected video has no decodable frames: {relative}")
            timestamps = [value - timestamps[0] for value in timestamps]
            if any(a > b for a, b in zip(timestamps, timestamps[1:])):
                timestamps = [index / max(fps, 1e-6) for index in range(len(timestamps))]
                source = "index_over_average_fps"
            else:
                source = "presentation_timestamp"
            manifest[relative] = {
                "n_frames": len(timestamps),
                "fps": fps,
                "width": int(stream.width or stream.codec_context.width),
                "height": int(stream.height or stream.codec_context.height),
                "duration": len(timestamps) / max(fps, 1e-6),
                "frame_timestamps": timestamps,
                "timestamp_source": source,
            }
    (work / "data" / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    (work / "data" / "dataset_revision.json").write_text(
        json.dumps({"repo": DATASET_REPO, "revision": DATASET_REVISION, "referenced_clips": len(paths)}, indent=2) + "\n"
    )
    return manifest, scope_selection, execution_selection


def _run(work: Path, command: list[str], logs: list[str]) -> None:
    rendered = " ".join(shlex.quote(item) for item in command)
    print(f"$ {rendered}", flush=True)
    result = subprocess.run(
        command,
        cwd=work,
        env=os.environ.copy(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logs.append(f"$ {rendered}\n{result.stdout}")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {rendered}")


def _detector_prompt(selection: dict[str, Any], split: str) -> str:
    entities: set[str] = set()
    for row in selection["selected_questions"][split]:
        entities.update(row["prompt_entities_from_question_only"])
    ordered = sorted(entities)
    if not ordered:
        raise RuntimeError(f"no question entities for {split}")
    return ". ".join(ordered) + "."


def _write_execution_qa_ids(work: Path, scope: str, selection: dict[str, Any]) -> Path:
    """Write the split-aware ID file required by build_grounding_cache.py."""

    if scope not in {"tranche", "full"}:
        raise ValueError("scope must be tranche or full")
    expected_count = 2 if scope == "tranche" else 28
    tranche_ids = {
        split: set(selection["tranche"]["ids"][split]) for split in SPLITS
    }
    ids_by_split: dict[str, list[str]] = {}
    for split in SPLITS:
        ids = [row["qa_id"] for row in selection["selected_questions"][split]]
        if len(ids) != expected_count or len(ids) != len(set(ids)):
            raise RuntimeError(f"{scope} execution selection has invalid {split} ID count")
        if scope == "tranche" and set(ids) != tranche_ids[split]:
            raise RuntimeError(f"tranche execution IDs drifted for {split}")
        if scope == "full" and set(ids) & tranche_ids[split]:
            raise RuntimeError(f"full execution IDs overlap the completed tranche for {split}")
        ids_by_split[split] = ids
    destination = work / RUN / scope / "qa_ids.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"splits": {split: {"ids": ids} for split, ids in ids_by_split.items()}}, indent=2) + "\n")
    return destination


def _merge_tranche_artifacts(work: Path, scope: str, selection: dict[str, Any]) -> None:
    """Combine unchanged tranche rows with newly scored remaining rows."""

    if scope != "full":
        return
    prior_root = work / RUN / "tranche"
    output_root = work / RUN / scope
    for condition in ("g1", "g2"):
        for split in SPLITS:
            for suffix in ("contexts.jsonl", "scores.jsonl"):
                destination = output_root / f"{condition}_{split}_{suffix}"
                prior = prior_root / destination.name
                if not prior.is_file() or not destination.is_file():
                    raise RuntimeError(f"cannot merge missing tranche artifact: {prior} or {destination}")
                rows_by_id: dict[str, str] = {}
                for source in (prior, destination):
                    for line in source.read_text().splitlines():
                        row = json.loads(line)
                        qa_id = row.get("qa_id")
                        if not isinstance(qa_id, str) or qa_id in rows_by_id:
                            raise RuntimeError(f"duplicate or invalid row while merging {destination}")
                        rows_by_id[qa_id] = json.dumps(row, separators=(",", ":"))
                ordered_ids = [row["qa_id"] for row in selection["selected_questions"][split]]
                if set(rows_by_id) != set(ordered_ids):
                    raise RuntimeError(f"merged artifact IDs do not match full selection: {destination}")
                destination.write_text("\n".join(rows_by_id[qa_id] for qa_id in ordered_ids) + "\n")


@app.function(gpu="A100", cpu=8, memory=65536, timeout=18 * 60 * 60, secrets=SECRETS)
def run_phase3(scope: str) -> dict[str, bytes]:
    """Run evidence construction and paired scoring, returning only run files."""

    if scope not in {"tranche", "full"}:
        raise ValueError("scope must be tranche or full")
    # This check precedes dataset/model work and therefore guarantees that a
    # missing Replicate credential cannot trigger any paid or model call.
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise RuntimeError("Modal secret did not provide REPLICATE_API_TOKEN; no model/API call was made")
    work = Path(WORK)
    output_dir = work / RUN / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    # These markers describe this invocation only.  Remove stale markers from
    # an earlier attempt before entering the run's error-reporting block.
    (output_dir / "phase3_error.txt").unlink(missing_ok=True)
    (output_dir / "phase3_complete.json").unlink(missing_ok=True)
    logs: list[str] = []
    try:
        manifest, scope_selection_path, execution_selection_path = _prepare_data(work, scope)
        selection = json.loads(scope_selection_path.read_text())
        execution_selection = json.loads(execution_selection_path.read_text())
        (output_dir / "scope_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        (output_dir / "execution_selection.json").write_text(json.dumps(execution_selection, indent=2) + "\n")
        _write_execution_qa_ids(work, scope, execution_selection)
        detector_dir = work / "hf_cache" / "grounding-dino-tiny-a2bb814"
        sys.path.insert(0, str(work / "code"))
        import grounding as grounding_module
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=grounding_module.GROUNDING_DETECTOR_MODEL_ID,
            revision=grounding_module.GROUNDING_DETECTOR_MODEL_REVISION,
            local_dir=str(detector_dir),
            allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
        )
        (detector_dir / grounding_module.GROUNDING_CACHE_MARKER).write_text(
            f"{grounding_module.GROUNDING_DETECTOR_MODEL_REVISION}\n"
        )
        detector_cache = f"{RUN}/{scope}/grounding_cache"
        for split in SPLITS:
            _run(
                work,
                [
                    sys.executable,
                    "code/build_grounding_cache.py",
                    "--split",
                    split,
                    "--model-dir",
                    str(detector_dir),
                    "--grounding-cache-dir",
                    detector_cache,
                    "--qa-ids-file",
                    f"{RUN}/{scope}/qa_ids.json",
                    "--prompt",
                    _detector_prompt(execution_selection, split),
                    "--offsets",
                    "0,0.5",
                    "--frame-sampling",
                    "legacy",
                    "--device",
                    "cuda",
                    "--batch-size",
                    "4",
                ],
                logs,
            )
        for split in SPLITS:
            _run(
                work,
                [
                    sys.executable,
                    "runs/phase3_explicit_geometry/build_contexts.py",
                    "--split",
                    split,
                    "--condition",
                    "G1",
                    "--selection",
                    f"{RUN}/{scope}/execution_selection.json",
                    "--out",
                    f"{RUN}/{scope}/g1_{split}_contexts.jsonl",
                    "--grounding-cache-dir",
                    detector_cache,
                    "--detector-prompt",
                    _detector_prompt(execution_selection, split),
                ],
                logs,
            )
        if scope == "full":
            prior_raw = work / RUN / "tranche" / "sam3" / "sam3_raw"
            if prior_raw.is_dir():
                shutil.copytree(prior_raw, output_dir / "sam3" / "sam3_raw", dirs_exist_ok=True)
        _run(
            work,
            [
                sys.executable,
                "runs/phase3_explicit_geometry/replicate_phase3.py",
                "--scope",
                scope,
                "--data-root",
                "data/ReVA",
                "--manifest",
                "data/manifest.json",
                "--selection",
                f"{RUN}/{scope}/execution_selection.json",
                "--output-root",
                f"{RUN}/{scope}/sam3",
            ],
            logs,
        )
        for split in SPLITS:
            _run(
                work,
                [
                    sys.executable,
                    "runs/phase3_explicit_geometry/build_contexts.py",
                    "--split",
                    split,
                    "--condition",
                    "G2",
                    "--selection",
                    f"{RUN}/{scope}/execution_selection.json",
                    "--out",
                    f"{RUN}/{scope}/g2_{split}_contexts.jsonl",
                    "--sam3-cache-dir",
                    f"{RUN}/{scope}/sam3/overlay_cache",
                ],
                logs,
            )
        for condition in ("G1", "G2"):
            for split in SPLITS:
                _run(
                    work,
                    [
                        sys.executable,
                        "runs/phase3_explicit_geometry/score_explicit.py",
                        "--split",
                        split,
                        "--condition",
                        condition,
                        "--scope",
                        scope,
                        "--selection",
                        f"{RUN}/{scope}/execution_selection.json",
                        "--context",
                        f"{RUN}/{scope}/{condition.casefold()}_{split}_contexts.jsonl",
                        "--out",
                        f"{RUN}/{scope}/{condition.casefold()}_{split}_scores.jsonl",
                    ],
                    logs,
                )
        _merge_tranche_artifacts(work, scope, selection)
        _run(
            work,
            [
                sys.executable,
                "runs/phase3_explicit_geometry/analyze_phase3.py",
                "--scope-dir",
                f"{RUN}/{scope}",
                "--selection",
                f"{RUN}/{scope}/scope_selection.json",
            ],
            logs,
        )
        (output_dir / "phase3_complete.json").write_text(
            json.dumps({"scope": scope, "status": "complete", "selected": {split: len(selection["selected_questions"][split]) for split in SPLITS}}, indent=2) + "\n"
        )
    except Exception as exc:
        logs.append(f"{type(exc).__name__}: {exc}\n")
        (output_dir / "phase3_error.txt").write_text("".join(logs))
        print(logs[-1], end="", flush=True)
    outputs = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    outputs["modal_phase3.log"] = "\n".join(logs).encode()
    return outputs


@app.local_entrypoint()
def main(scope: str = "tranche") -> None:
    if scope not in {"tranche", "full"}:
        raise ValueError("scope must be tranche or full")
    caller_secret_name = os.environ.get("REVA_REPLICATE_MODAL_SECRET", "").strip()
    if caller_secret_name != APPROVED_SECRET_NAME:
        raise RuntimeError(
            f"Set REVA_REPLICATE_MODAL_SECRET={APPROVED_SECRET_NAME} before submitting; "
            "no remote call was made"
        )
    if scope == "full" and not (ROOT / RUN / "tranche_complete.json").is_file():
        raise RuntimeError("full scope is locked until the 4-question tranche is complete")
    destination = ROOT / RUN / scope
    result = run_phase3.remote(scope)
    for name, contents in result.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe Modal result path: {name}")
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        print(f"saved {path} ({len(contents):,} bytes)")
    if "phase3_error.txt" in result:
        raise RuntimeError(f"Phase 3 {scope} run failed; partial outputs and error log were saved")
    expected = [f"{condition.casefold()}_{split}_scores.jsonl" for condition in ("G1", "G2") for split in SPLITS]
    if scope == "tranche" and all(name in result for name in expected):
        (ROOT / RUN / "tranche_complete.json").write_text(
            json.dumps({"scope": "tranche", "status": "complete", "score_files": expected}, indent=2) + "\n"
        )

