"""Score a split with letter logits and frame/option test-time augmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def move_inputs(inputs: dict, device: torch.device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="final_adapter directory, or none for zero-shot")
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=','.join(str(value) for value in C.DEFAULT_SCORE_OFFSETS))
    ap.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if not 1 <= args.rots <= 4 or not 0 <= args.shard < args.nshards:
        ap.error("rots must be 1..4, and shard must be in [0, nshards)")
    offsets = [float(value) for value in args.offsets.split(",") if value.strip()]
    if not offsets or any(not 0 <= value < 1 for value in offsets):
        ap.error("offsets must be comma-separated values in [0, 1)")

    split_path = C.work_path(SPLITS[args.split])
    with split_path.open() as f:
        all_rows = json.load(f)["QA"]
    split_digest = C.split_checksum(all_rows)
    rows = all_rows
    rows = rows[args.shard :: args.nshards]
    out = C.work_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapter_path = None if args.adapter == "none" else C.work_path(args.adapter)
    expected_manifest = {
        "schema_version": 1,
        "adapter": "none" if adapter_path is None else str(adapter_path.resolve()),
        "adapter_checksum": "none" if adapter_path is None else C.path_checksum(adapter_path),
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REV,
        "dataset_revision": C.dataset_revision(),
        "split": args.split,
        "split_checksum": split_digest,
        "offsets": offsets,
        "rots": args.rots,
        "shard": args.shard,
        "nshards": args.nshards,
    }
    manifest_path = C.score_manifest_path(out)
    if manifest_path.exists():
        with manifest_path.open() as f:
            actual_manifest = json.load(f)
        if actual_manifest != expected_manifest:
            raise RuntimeError(f"score manifest mismatch: {manifest_path}; choose a new output or remove stale scores")
    elif out.exists() and out.stat().st_size:
        raise RuntimeError(f"existing score file has no manifest: {out}; choose a new output or remove stale scores")
    else:
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with temporary.open("w") as f:
            json.dump(expected_manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        temporary.replace(manifest_path)
    done = C.read_completed_jsonl(out)
    stop_file = os.environ.get("REVA_STOP_FILE")
    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video scoring")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    device = torch.device("cuda")
    model.to(device).eval()
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(args.rots)]
    with out.open("a") as stream, torch.inference_mode():
        for row in rows:
            if row["qa_id"] in done:
                continue
            if stop_file and Path(stop_file).exists():
                print(f"stop requested; resume from {out}")
                break
            views = []
            for offset in offsets:
                frames, meta = C.get_video(row["video_path"], offset)
                for permutation in rotations:
                    encoded, actual_perm = C.encode(processor, row, frames, meta, permutation)
                    logits = model(**move_inputs(encoded, device)).logits[0, -1, ltr_ids].float()
                    shown_prob = torch.softmax(logits, dim=-1).cpu().numpy()
                    original_prob = [0.0] * 4
                    for shown_index, original_letter in enumerate(actual_perm):
                        original_prob[C.LETTERS.index(original_letter)] = float(shown_prob[shown_index])
                    views.append(
                        {
                            "offset": offset,
                            "perm": "".join(actual_perm),
                            "p": original_prob,
                            "argmax_shown": C.LETTERS[int(np.argmax(shown_prob))],
                        }
                    )
            mean = np.mean([view["p"] for view in views], axis=0).tolist()
            result = {
                "qa_id": row["qa_id"],
                "p": mean,
                "pred": C.LETTERS[int(np.argmax(mean))],
                "views": views,
            }
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
    print(f"scored {args.split} shard {args.shard}/{args.nshards}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
