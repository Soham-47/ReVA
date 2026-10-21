"""Run the fixed Phase 3 SAM 2 mask pilot on one Modal A100 function."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import modal


ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN = "runs/segmentation_pilot_stageA_modal"
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
        "pillow",
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
    image = image.add_local_file(str(manifest_source), f"{WORK}/data/manifest.json", copy=True)

app = modal.App("reva-segmentation-pilot", image=image)


@app.function(gpu="A100", cpu=8, memory=65536, timeout=18 * 60 * 60)
def segmentation_pilot() -> dict[str, bytes]:
    """Stage pinned models, smoke-test one clip, run the fixed 32-question pilot."""

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
    import common as C
    import grounding as G
    import segmentation as S

    pilot = json.loads((work / PILOT).read_text())
    expected_checksums = pilot.get("source_split_checksums", {})
    split_rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        rows = json.loads((work / "data" / "splits" / f"{split}.json").read_text())["QA"]
        split_rows[split] = rows
        actual = C.split_checksum(rows)
        if actual != expected_checksums.get(split):
            raise RuntimeError(f"{split} checksum mismatch: {actual}")
        ids = pilot["splits"][split]["ids"]
        if len(ids) != 16 or len(ids) != len(set(ids)):
            raise RuntimeError(f"pilot manifest must contain exactly 16 unique IDs for {split}")
        if set(ids) - {row.get("qa_id") for row in rows}:
            raise RuntimeError(f"pilot IDs absent from remote {split}")
        print(f"verified {split}: checksum={actual} selected_ids=16", flush=True)

    from huggingface_hub import hf_hub_download, snapshot_download

    dataset_root = work / "data" / "ReVA"
    dataset_root.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        filename="train.json",
        revision=C.DATASET_REVISION,
        local_dir=str(dataset_root),
    )
    train_rows = json.loads((dataset_root / "train.json").read_text())["QA"]
    train_by_id = {row["qa_id"]: row for row in train_rows}
    selected_rows = []
    for split in SPLITS:
        for qa_id in pilot["splits"][split]["ids"]:
            row = next(item for item in split_rows[split] if item["qa_id"] == qa_id)
            if train_by_id.get(qa_id) != row:
                raise RuntimeError(f"fixed {split} row differs from train.json: {qa_id}")
            selected_rows.append(row)
    selected_paths = sorted({row["video_path"] for row in selected_rows})
    snapshot_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        revision=C.DATASET_REVISION,
        local_dir=str(dataset_root),
        allow_patterns=selected_paths,
        max_workers=8,
    )
    if not all((dataset_root / path).is_file() for path in selected_paths):
        raise RuntimeError("one or more selected pilot videos are missing")
    (work / "data" / "dataset_revision.json").write_text(
        json.dumps(
            {"repo": C.DATASET_REPO, "revision": C.DATASET_REVISION, "referenced_clips": len(selected_paths)},
            indent=2,
        )
        + "\n"
    )

    manifest_path = work / "data" / "manifest.json"
    if not manifest_path.is_file() or any(path not in json.loads(manifest_path.read_text()) for path in selected_paths):
        import av

        manifest: dict[str, dict] = {}
        for relative in selected_paths:
            with av.open(str(dataset_root / relative)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or stream.base_rate or 30.0)
                frames = list(container.decode(stream))
                n_frames = len(frames)
                width, height = int(stream.width), int(stream.height)
            manifest[relative] = {
                "n_frames": n_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration": n_frames / fps,
            }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    detector_dir = work / "hf_cache" / "grounding-dino-tiny-a2bb814"
    snapshot_download(
        repo_id=G.GROUNDING_DETECTOR_MODEL_ID,
        revision=G.GROUNDING_DETECTOR_MODEL_REVISION,
        local_dir=str(detector_dir),
        allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
    )
    (detector_dir / G.GROUNDING_CACHE_MARKER).write_text(f"{G.GROUNDING_DETECTOR_MODEL_REVISION}\n")

    sam_dir = work / "hf_cache" / "sam2-hiera-tiny-7c218be"
    snapshot_download(
        repo_id=S.SEGMENTATION_MODEL_ID,
        revision=S.SEGMENTATION_MODEL_REVISION,
        local_dir=str(sam_dir),
        allow_patterns=["*.json", "model.safetensors"],
    )
    (sam_dir / S.SEGMENTATION_CACHE_MARKER).write_text(f"{S.SEGMENTATION_MODEL_REVISION}\n")

    grounding_cache_dir = f"{RUN}/grounding_cache"
    segmentation_cache_dir = f"{RUN}/segmentation_cache"
    for split in SPLITS:
        run(
            "code/build_grounding_cache.py",
            "--split",
            split,
            "--model-dir",
            str(detector_dir),
            "--grounding-cache-dir",
            grounding_cache_dir,
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

    # Smoke one clip and inspect files before the full mask pass.
    smoke_row = next(row for row in selected_rows if row["video_path"] in selected_paths)
    smoke_video_file = work / "tasks" / "phase3_segmentation_smoke_video.txt"
    smoke_video_file.parent.mkdir(parents=True, exist_ok=True)
    smoke_video_file.write_text(smoke_row["video_path"] + "\n")
    run(
        "code/build_segmentation_cache.py",
        "--split",
        "valQ",
        "--model-dir",
        str(sam_dir),
        "--grounding-cache-dir",
        grounding_cache_dir,
        "--segmentation-cache-dir",
        segmentation_cache_dir,
        "--video-paths-file",
        str(smoke_video_file),
        "--offsets",
        "0,0.5",
        "--frame-sampling",
        "legacy",
        "--device",
        "cuda",
        "--batch-size",
        "1",
    )
    smoke_json = list((work / segmentation_cache_dir).glob("*.json"))
    smoke_contacts = list((work / segmentation_cache_dir / "contacts").glob("*.jpg"))
    smoke_masks = list((work / segmentation_cache_dir / "masks").rglob("*.png"))
    if not smoke_json or not smoke_contacts or not smoke_masks:
        raise RuntimeError("SAM 2 smoke pass produced no inspectable cache/contact/mask artifacts")
    print(
        f"SAM2 smoke artifacts: caches={len(smoke_json)} contacts={len(smoke_contacts)} masks={len(smoke_masks)}",
        flush=True,
    )

    for split in SPLITS:
        run(
            "code/build_segmentation_cache.py",
            "--split",
            split,
            "--model-dir",
            str(sam_dir),
            "--grounding-cache-dir",
            grounding_cache_dir,
            "--segmentation-cache-dir",
            segmentation_cache_dir,
            "--qa-ids-file",
            PILOT,
            "--offsets",
            "0,0.5",
            "--frame-sampling",
            "legacy",
            "--device",
            "cuda",
            "--batch-size",
            "1",
        )

    adapter = "runs/stageA_q3vl8b_r16/final_adapter"
    outputs: dict[str, bytes] = {}
    for split in SPLITS:
        score_stem = f"{RUN}/scores_{split}_segmentation_pilot32.jsonl"
        run(
            "code/score.py",
            "--adapter",
            adapter,
            "--split",
            split,
            "--grounding-context",
            "--grounding-cache-dir",
            grounding_cache_dir,
            "--segmentation-context",
            "--segmentation-cache-dir",
            segmentation_cache_dir,
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
        import segmentation as Seg

        score_path = Seg.segmentation_output_path(work / score_stem)
        manifest = C.score_manifest_path(score_path)
        for path in (score_path, manifest):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"expected non-empty score artifact: {path}")
            outputs[path.name] = path.read_bytes()

    # Return the inspectable cache as one compressed artifact alongside scores.
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        archive.add(work / segmentation_cache_dir, arcname="segmentation_cache")
    outputs["segmentation_cache.tar.gz"] = archive_buffer.getvalue()
    print("SAM 2 segmentation pilot complete", flush=True)
    return outputs


@app.local_entrypoint()
def main() -> None:
    expected = {
        name
        for split in SPLITS
        for name in (
            f"scores_{split}_segmentation_pilot32.segmentation.sam2-box-overlay-v1.jsonl",
            f"scores_{split}_segmentation_pilot32.segmentation.sam2-box-overlay-v1.jsonl.manifest.json",
        )
    }
    expected.add("segmentation_cache.tar.gz")
    returned = segmentation_pilot.remote()
    if set(returned) != expected:
        raise RuntimeError(f"Modal returned unexpected result files: {sorted(returned)}")
    destination = ROOT / RUN
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in returned.items():
        path = destination / name
        path.write_bytes(contents)
        print(f"saved {path} ({len(contents):,} bytes)")

