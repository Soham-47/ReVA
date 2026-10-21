"""Run the existing Qwen3.8 Stage A recipe on Modal with resumable volumes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


PROJECT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "soham-reva-qwen38-stagea"
WORK = Path("/mnt/reva")
CODE = Path("/root/reva/code")
MODEL_CACHE = WORK / "hf_cache"

app = modal.App("reva-qwen38-stagea")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(str(PROJECT / "requirements.txt"))
    .add_local_dir(
        str(PROJECT / "code"),
        remote_path=str(CODE),
        ignore=["**/__pycache__/**"],
    )
)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "WORK": str(WORK),
            "HF_HOME": str(MODEL_CACHE),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    return env


def _run(*args: str, gpu: str | None = None) -> None:
    if gpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Modal did not attach the requested CUDA GPU")
        print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    subprocess.run(args, cwd="/root/reva", env=_environment(), check=True)


@app.function(
    image=image,
    volumes={str(WORK): volume},
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=30),
)
def prepare_data_and_weights() -> str:
    ready = WORK / "MODAL_DATA_AND_WEIGHTS_READY"
    if ready.exists():
        print("pinned dataset and model snapshot already staged")
        return "ready"

    split = WORK / "data/splits/fit_A.json"
    manifest = WORK / "data/manifest.json"
    if not split.is_file() or not manifest.is_file():
        raise FileNotFoundError("upload the pinned fit_A split and ReVA manifest before preparation")

    _run(sys.executable, str(CODE / "download_data.py"), "--work", str(WORK))

    import yaml
    from huggingface_hub import snapshot_download

    config_path = CODE / "configs/stageA_qwen38_a100.yaml"
    config = yaml.safe_load(config_path.read_text())
    snapshot = snapshot_download(
        repo_id=config["model_id"],
        revision=config["model_revision"],
        cache_dir=str(MODEL_CACHE / "hub"),
    )
    print(f"pinned model snapshot staged: {snapshot}")
    ready.write_text("dataset revision and model revision staged\n")
    volume.commit()
    return "ready"


def _require_prepared() -> None:
    if not (WORK / "MODAL_DATA_AND_WEIGHTS_READY").is_file():
        raise RuntimeError("run prepare_data_and_weights before the GPU jobs")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=12,
    memory=131072,
    volumes={str(WORK): volume},
    timeout=6 * 60 * 60,
    single_use_containers=True,
)
def smoke_a100() -> str:
    _require_prepared()
    marker = WORK / "runs/stageA_qwen38_27b_r16_a100_smoke/SMOKE_OK"
    if marker.exists():
        return "smoke already passed"
    _run(
        sys.executable,
        str(CODE / "smoke.py"),
        "--config",
        str(CODE / "configs/stageA_qwen38_a100_smoke.yaml"),
        gpu="A100",
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")
    volume.commit()
    return "A100 smoke passed"


def _train(config_name: str, *, checkpoint_step: int | None = None) -> str:
    _require_prepared()
    config_path = CODE / "configs" / config_name
    with config_path.open() as f:
        import yaml

        config = yaml.safe_load(f)
    output = WORK / config["output_dir"]
    if checkpoint_step is not None:
        checkpoint = output / f"checkpoint-{checkpoint_step}"
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"required resume checkpoint is missing: {checkpoint}")
        state = json.loads(state_path.read_text())
        if int(state.get("global_step", -1)) != checkpoint_step:
            raise RuntimeError(f"expected checkpoint step {checkpoint_step}, got {state.get('global_step')}")
    if config_name == "stageA_qwen38_a100.yaml":
        smoke_marker = WORK / "runs/stageA_qwen38_27b_r16_a100_smoke/SMOKE_OK"
        if not smoke_marker.is_file():
            raise RuntimeError("A100 training is gated on a passing A100 smoke")
    _run(sys.executable, str(CODE / "train.py"), str(config_path), gpu=config_name)
    volume.commit()
    return f"completed {config['run_name']}"


@app.function(
    image=image,
    gpu="H200",
    cpu=12,
    memory=131072,
    volumes={str(WORK): volume},
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=15),
    single_use_containers=True,
)
def train_h200_from_checkpoint() -> str:
    return _train("stageA_qwen38_h200.yaml", checkpoint_step=1641)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=12,
    memory=131072,
    volumes={str(WORK): volume},
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=15),
    single_use_containers=True,
)
def train_a100() -> str:
    return _train("stageA_qwen38_a100.yaml")
