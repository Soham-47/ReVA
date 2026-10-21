"""Run the endpoint-sampling validation on Modal with the existing Stage A adapter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN = "runs/stageA_q3vl8b_r16"
SPLITS = ("valQ", "valT", "val")

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
        str(ROOT / RUN / "final_adapter"),
        f"{WORK}/{RUN}/final_adapter",
        copy=True,
    )
    .env({"WORK": WORK, "HF_HOME": "/root/hf_cache"})
)

app = modal.App("reva-endpoint-validation", image=image)


@app.function(
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
)
def score_endpoints() -> dict[str, bytes]:
    work = Path(WORK)
    logs = []

    def run(*args: str) -> str:
        result = subprocess.run(
            [sys.executable, *args],
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
            raise RuntimeError(f"command failed ({result.returncode}): {args}")
        return result.stdout

    run("code/download_data.py", "--work", WORK)
    run("code/make_splits.py", "--work", WORK)
    run("code/build_manifest.py", "--work", WORK)

    outputs = {}
    for split in SPLITS:
        score = f"{RUN}/scores_{split}_endpoints.jsonl"
        run(
            "code/score.py",
            "--adapter",
            f"{RUN}/final_adapter",
            "--split",
            split,
            "--frame-sampling",
            "endpoints",
            "--out",
            score,
        )
        run("code/evaluate.py", score, split)
        for path in (work / score, work / f"{score}.manifest.json"):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"expected non-empty result file: {path}")
            outputs[path.name] = path.read_bytes()

    outputs["endpoint_evaluation.log"] = "\n".join(logs).encode()
    return outputs


@app.local_entrypoint()
def main() -> None:
    expected = {
        name
        for split in SPLITS
        for name in (
            f"scores_{split}_endpoints.jsonl",
            f"scores_{split}_endpoints.jsonl.manifest.json",
        )
    } | {"endpoint_evaluation.log"}
    returned = score_endpoints.remote()
    if returned.keys() != expected:
        raise RuntimeError(f"Modal returned unexpected result files: {sorted(returned)}")

    destination = ROOT / RUN
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in returned.items():
        (destination / name).write_bytes(contents)
        print(f"saved {destination / name} ({len(contents):,} bytes)")
