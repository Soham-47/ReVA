"""Small, read-only GPU probe for the Qwen3-VL vision forward path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import Qwen3VLForConditionalGeneration, __version__ as transformers_version

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--metadata-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--disable-cudnn-sdp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic requires a CUDA GPU")
    device = torch.device("cuda")
    if args.disable_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers_version,
                "device": torch.cuda.get_device_name(device),
                "capability": torch.cuda.get_device_capability(device),
                "arch_list": torch.cuda.get_arch_list(),
                "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "attn": args.attn,
                "metadata_device": args.metadata_device,
                "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            sort_keys=True,
        )
    )
    with (Path(__file__).resolve().parent / "configs" / "stageA.yaml").open() as f:
        config = yaml.safe_load(f)
    processor = C.load_processor()
    tokenizer = processor.tokenizer
    with C.work_path("data/splits/fit_A.json").open() as f:
        row = next(row for row in json.load(f)["QA"] if row["dataset_name"].startswith("VisDrone"))
    frames, meta = C.get_video(row["video_path"], 0.0)
    encoded, _ = C.encode(
        processor,
        row,
        frames,
        meta,
        ["B", "D", "A", "C"],
        row["correct_answer"],
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        C.letter_ids(tokenizer),
    )
    print("encoded", frames.shape, encoded["video_grid_thw"].tolist())
    inputs = {}
    for key, value in encoded.items():
        if not hasattr(value, "to"):
            inputs[key] = value
        elif "grid_thw" in key and args.metadata_device == "cpu":
            inputs[key] = value
        else:
            inputs[key] = value.to(device)
        if hasattr(inputs[key], "device"):
            print("input", key, inputs[key].device, inputs[key].dtype, tuple(inputs[key].shape))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(**inputs)
    if not torch.isfinite(output.loss):
        raise AssertionError("diagnostic loss is not finite")
    print("loss", float(output.loss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
