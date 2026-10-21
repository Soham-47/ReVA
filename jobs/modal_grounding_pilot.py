"""Run the fixed Phase 3 grounding-only pilot on one Modal A100 function."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN = "runs/grounding_pilot_stageA_modal"
PILOT = "tasks/phase3_grounding_pilot_ids.json"
SPLITS = ("valQ", "valT")

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
        "pyyaml",
        "tqdm",
    )
    .env({"WORK": WORK, "HF_HOME": f"{WORK}/hf_cache"})
    .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
    .add_local_file(str(ROOT / PILOT), f"{WORK}/{PILOT}", copy=True)
    .add_local_dir(str(ROOT / "data" / "splits"), f"{WORK}/data/splits", copy=True)
    .add_local_dir(
        str(ROOT / "runs" / "stageA_q3vl8b_r16" / "final_adapter"),
        f"{WORK}/runs/stageA_q3vl8b_r16/final_adapter",
        copy=True,
    )
)

manifest_source = ROOT / "data" / "manifest.json"
if manifest_source.is_file():
    image = image.add_local_file(
        str(manifest_source), f"{WORK}/data/manifest.json", copy=True
    )

app = modal.App("reva-grounding-pilot", image=image)


@app.function(
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=12 * 60 * 60,
)
def grounding_pilot() -> dict[str, bytes]:
    """Download pinned inputs, run the fixed pilot, and return score artifacts."""

    work = Path(WORK)
    logs: list[str] = []

    def run(*args: str) -> str:
        command = " ".join(shlex.quote(arg) for arg in args)
        print(f"$ {command}", flush=True)
        result = subprocess.run(
            [sys.executable, *args],
            cwd=WORK,
            env=os.environ.copy(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logs.append(f"$ {command}\n{result.stdout}")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {command}")
        return result.stdout

    sys.path.insert(0, str(work / "code"))
    import build_grounding_cache as B
    import common as C
    import grounding as G

    pilot_path = work / PILOT
    pilot = json.loads(pilot_path.read_text())
    expected_checksums = pilot.get("source_split_checksums", {})
    if set(expected_checksums) != set(SPLITS):
        raise RuntimeError("pilot manifest must provide source checksums for valQ and valT")
    split_rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        split_path = work / "data" / "splits" / f"{split}.json"
        rows = json.loads(split_path.read_text())["QA"]
        split_rows[split] = rows
        actual_checksum = C.split_checksum(rows)
        expected_checksum = expected_checksums[split]
        if actual_checksum != expected_checksum:
            raise RuntimeError(
                f"{split} checksum mismatch: {actual_checksum} (expected {expected_checksum})"
            )
        ids = pilot.get("splits", {}).get(split, {}).get("ids")
        if not isinstance(ids, list) or len(ids) != 16 or len(ids) != len(set(ids)):
            raise RuntimeError(f"pilot manifest must contain exactly 16 unique IDs for {split}")
        available = {row.get("qa_id") for row in rows}
        missing = sorted(set(ids) - available)
        if missing:
            raise RuntimeError(f"pilot IDs absent from remote {split}: {', '.join(missing)}")
        print(f"verified {split}: checksum={actual_checksum} selected_ids=16", flush=True)

    # Fetch only the released train index and the videos named by the fixed
    # 32-question manifest. The split JSONs are copied from the local/HPC
    # checkout so their valQ ordering and checksums cannot drift remotely.
    from huggingface_hub import hf_hub_download, snapshot_download

    dataset_root = work / "data" / "ReVA"
    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"downloading train.json at dataset revision {C.DATASET_REVISION}", flush=True)
    hf_hub_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        filename="train.json",
        revision=C.DATASET_REVISION,
        local_dir=str(dataset_root),
    )
    train_rows = json.loads((dataset_root / "train.json").read_text())["QA"]
    train_by_id = {row["qa_id"]: row for row in train_rows}
    selected_rows: list[dict] = []
    selected_ids: list[str] = []
    for split in SPLITS:
        for qa_id in pilot["splits"][split]["ids"]:
            row = next(item for item in split_rows[split] if item["qa_id"] == qa_id)
            if train_by_id.get(qa_id) != row:
                raise RuntimeError(f"fixed {split} row differs from dataset train.json: {qa_id}")
            selected_ids.append(qa_id)
            selected_rows.append(row)
    if len(selected_ids) != 32 or len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("pilot manifest must contain 32 unique IDs across valQ and valT")
    selected_paths = sorted({row["video_path"] for row in selected_rows})
    print(f"downloading {len(selected_paths)} pilot videos", flush=True)
    snapshot_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        revision=C.DATASET_REVISION,
        local_dir=str(dataset_root),
        allow_patterns=selected_paths,
        max_workers=8,
    )
    missing_videos = [path for path in selected_paths if not (dataset_root / path).is_file()]
    if missing_videos:
        raise RuntimeError(f"missing selected dataset videos: {', '.join(missing_videos[:5])}")
    (work / "data" / "dataset_revision.json").write_text(
        json.dumps(
            {
                "repo": C.DATASET_REPO,
                "revision": C.DATASET_REVISION,
                "referenced_clips": len(selected_paths),
            },
            indent=2,
        )
        + "\n"
    )

    manifest_path = work / "data" / "manifest.json"
    if not manifest_path.is_file():
        # A full HPC manifest is bundled when available. Otherwise decode only
        # these selected clips; the full released dataset is never scanned.
        import av

        manifest: dict[str, dict] = {}
        for relative in selected_paths:
            path = dataset_root / relative
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or stream.base_rate or 30.0)
                n_frames = sum(1 for _ in container.decode(stream))
                width = int(stream.width or stream.codec_context.width)
                height = int(stream.height or stream.codec_context.height)
            if n_frames < 1 or width < 1 or height < 1:
                raise RuntimeError(f"invalid selected video metadata: {relative}")
            manifest[relative] = {
                "n_frames": n_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": n_frames / fps,
            }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"wrote selected-video manifest: {manifest_path}", flush=True)
    else:
        manifest = json.loads(manifest_path.read_text())
        missing_manifest = [path for path in selected_paths if path not in manifest]
        if missing_manifest:
            raise RuntimeError(
                f"bundled data/manifest.json is missing selected videos: {', '.join(missing_manifest[:5])}"
            )

    model_dir = work / "hf_cache" / "grounding-dino-tiny-a2bb814"

    print(
        f"downloading {G.GROUNDING_DETECTOR_MODEL_ID}@{G.GROUNDING_DETECTOR_MODEL_REVISION}",
        flush=True,
    )
    snapshot_download(
        repo_id=G.GROUNDING_DETECTOR_MODEL_ID,
        revision=G.GROUNDING_DETECTOR_MODEL_REVISION,
        local_dir=str(model_dir),
        allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
    )
    (model_dir / G.GROUNDING_CACHE_MARKER).write_text(
        f"{G.GROUNDING_DETECTOR_MODEL_REVISION}\n"
    )
    B.validate_local_model_dir(model_dir)
    print(f"validated detector snapshot: {model_dir}", flush=True)

    # Grounding DINO itself is now local and loaded with local_files_only=True.
    # Keep HF network access available for the existing Stage A base model,
    # which score.py loads by its pinned model revision.
    cache_dir = f"{RUN}/cache"
    adapter = "runs/stageA_q3vl8b_r16/final_adapter"
    outputs: dict[str, bytes] = {}
    for split in SPLITS:
        run(
            "code/build_grounding_cache.py",
            "--split",
            split,
            "--model-dir",
            str(model_dir),
            "--grounding-cache-dir",
            cache_dir,
            "--qa-ids-file",
            PILOT,
            "--offsets",
            "0,0.5",
            "--frame-sampling",
            "legacy",
            "--device",
            "cuda",
            "--batch-size",
            "4",
        )

    for split in SPLITS:
        score_stem = f"{RUN}/scores_{split}_grounding_pilot32.jsonl"
        run(
            "code/score.py",
            "--adapter",
            adapter,
            "--split",
            split,
            "--grounding-context",
            "--grounding-cache-dir",
            cache_dir,
            "--qa-ids-file",
            PILOT,
            "--offsets",
            "0,0.5",
            "--rots",
            "2",
            "--frame-sampling",
            "legacy",
            "--out",
            score_stem,
        )
        score_path = G.grounding_output_path(work / score_stem)
        manifest_path = C.score_manifest_path(score_path)
        for path in (score_path, manifest_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"expected non-empty result file: {path}")
            outputs[path.name] = path.read_bytes()

    print("grounding pilot complete; command logs are in the Modal function log", flush=True)
    return outputs


@app.local_entrypoint()
def main() -> None:
    expected = {
        name
        for split in SPLITS
        for name in (
            f"scores_{split}_grounding_pilot32.grounding.detector-v1.jsonl",
            f"scores_{split}_grounding_pilot32.grounding.detector-v1.jsonl.manifest.json",
        )
    }
    returned = grounding_pilot.remote()
    if set(returned) != expected:
        raise RuntimeError(f"Modal returned unexpected result files: {sorted(returned)}")

    destination = ROOT / RUN
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in returned.items():
        path = destination / name
        path.write_bytes(contents)
        print(f"saved {path} ({len(contents):,} bytes)")
