"""Run the approved Phase 2 temporal comparison on one Modal A100."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN = "runs/phase2_temporal_coarsefine"
QUESTION_SET = ROOT / RUN / "question_set.json"
DATASET = "ReVA-Benchmark/ReVA"
REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"

for mode in ("tranche", "full"):
    (ROOT / RUN / mode).mkdir(parents=True, exist_ok=True)

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
    .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
    .add_local_dir(
        str(ROOT / "runs/stageA_q3vl8b_r16/final_adapter"),
        f"{WORK}/runs/stageA_q3vl8b_r16/final_adapter",
        copy=True,
    )
    .add_local_file(str(QUESTION_SET), f"{WORK}/{RUN}/question_set.json", copy=True)
    .env({"WORK": WORK, "HF_HOME": "/root/hf_cache", "TOKENIZERS_PARALLELISM": "false"})
)
for mode in ("tranche", "full"):
    image = image.add_local_dir(str(ROOT / RUN / mode), f"{WORK}/{RUN}/{mode}", copy=True)

app = modal.App("reva-phase2-temporal", image=image)


def prepare_data(work: Path) -> dict:
    import av
    from huggingface_hub import snapshot_download

    question_set = json.loads((work / RUN / "question_set.json").read_text())
    paths = sorted({row["video_path"] for rows in question_set["splits"].values() for row in rows})
    if len(paths) != 135:
        raise RuntimeError(f"expected 135 question videos, found {len(paths)}")
    if any(PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts for path in paths):
        raise RuntimeError("unsafe video path in selected question set")

    data_root = work / "data/ReVA"
    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(data_root),
        allow_patterns=paths,
        max_workers=8,
    )
    manifest = {}
    for relative in paths:
        path = data_root / PurePosixPath(relative)
        if not path.is_file():
            raise FileNotFoundError(f"missing selected video: {relative}")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            try:
                fps = float(stream.average_rate or stream.base_rate or 30.0)
            except (TypeError, ValueError, ZeroDivisionError):
                fps = 30.0
            frames = []
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is not None and frame.time_base is not None:
                    frames.append(float(frame.pts * frame.time_base))
                else:
                    frames.append(index / max(fps, 1e-6))
            if not frames:
                raise RuntimeError(f"no decodable frames in {relative}")
            frames = [value - frames[0] for value in frames]
            if any(a > b for a, b in zip(frames, frames[1:])):
                frames = [index / max(fps, 1e-6) for index in range(len(frames))]
                timestamp_source = "index_over_average_fps"
            else:
                timestamp_source = "presentation_timestamp"
            manifest[relative] = {
                "n_frames": len(frames),
                "fps": fps,
                "width": int(stream.width or stream.codec_context.width),
                "height": int(stream.height or stream.codec_context.height),
                "duration": len(frames) / max(fps, 1e-6),
                "frame_timestamps": frames,
                "timestamp_source": timestamp_source,
            }
    revision_file = work / "data/dataset_revision.json"
    revision_file.write_text(json.dumps({"repo": DATASET, "revision": REVISION, "referenced_clips": len(paths)}, indent=2) + "\n")
    manifest_file = work / "data/manifest.json"
    manifest_file.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return manifest


@app.function(gpu="A100", cpu=8, memory=65536, timeout=12 * 60 * 60)
def score_phase2(mode: str) -> dict[str, bytes]:
    if mode not in {"tranche", "full"}:
        raise ValueError("mode must be tranche or full")
    work = Path(WORK)
    output_dir = work / RUN / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    try:
        manifest = prepare_data(work)
        (output_dir / "source_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        command = [
            sys.executable,
            "code/score_temporal.py",
            "--questions",
            f"{RUN}/question_set.json",
            "--output-dir",
            f"{RUN}/{mode}",
        ]
        if mode == "tranche":
            command.extend(["--limit-per-split", "4"])
        result = subprocess.run(
            command,
            cwd=WORK,
            env=os.environ.copy(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logs.append(result.stdout)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            (output_dir / "phase2_error.txt").write_text(
                f"score_temporal.py exited {result.returncode}\n{result.stdout}"
            )
    except Exception as exc:
        logs.append(f"{type(exc).__name__}: {exc}\n")
        (output_dir / "phase2_error.txt").write_text("".join(logs))
        print(logs[-1], end="", flush=True)

    outputs = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    outputs["modal_phase2.log"] = "\n".join(logs).encode()
    return outputs


@app.local_entrypoint()
def main(mode: str = "tranche") -> None:
    if mode not in {"tranche", "full"}:
        raise ValueError("mode must be tranche or full")
    destination = ROOT / RUN / mode
    result = score_phase2.remote(mode)
    for name, contents in result.items():
        path = destination / PurePosixPath(name)
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise RuntimeError(f"unsafe Modal result path: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        print(f"saved {path} ({len(contents):,} bytes)")
    if "phase2_error.txt" in result:
        raise RuntimeError(f"Phase 2 {mode} run failed; partial outputs and error log were saved")
    if "phase2_timing.json" in result:
        timing = json.loads(result["phase2_timing.json"])
        print("measured inference seconds:", json.dumps(timing, indent=2))
