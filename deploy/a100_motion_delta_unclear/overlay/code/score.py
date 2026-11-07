"""Score a split with letter logits and frame/option test-time augmentation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import specialist_data as S

SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def move_inputs(inputs: dict, device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="final_adapter directory, or none for zero-shot")
    ap.add_argument(
        "--no-merge-adapters",
        action="store_true",
        help="keep named PEFT adapters active instead of merging them into bfloat16 base weights",
    )
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=",".join(str(value) for value in C.DEFAULT_SCORE_OFFSETS))
    ap.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    ap.add_argument("--frame-sampling", choices=C.FRAME_SAMPLING_MODES, default=C.DEFAULT_FRAME_SAMPLING)
    ap.add_argument("--frame-budget", type=int, default=C.MAX_FRAMES)
    ap.add_argument("--tasks", default="", help="comma-separated exact task names; default scores every row")
    ap.add_argument(
        "--video-max-pixels",
        type=int,
        default=None,
        help="per-frame pixel cap (default: historical 640×640); must match adapter run_meta.json",
    )
    ap.add_argument(
        "--motion-hints",
        default=None,
        help="JSON cache of camera-motion sentences (same temporal_context path as train.py)",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if not 1 <= args.rots <= 4 or not 0 <= args.shard < args.nshards:
        ap.error("rots must be 1..4, and shard must be in [0, nshards)")
    if args.frame_budget < 1:
        ap.error("frame-budget must be positive")
    offsets = [float(value) for value in args.offsets.split(",") if value.strip()]
    if not offsets or any(not 0 <= value < 1 for value in offsets):
        ap.error("offsets must be comma-separated values in [0, 1)")

    split_path = C.work_path(SPLITS[args.split])
    with split_path.open() as f:
        all_rows = json.load(f)["QA"]
    task_names = [part.strip() for part in args.tasks.split(",") if part.strip()]
    if task_names:
        allowed_tasks = set(task_names)
        all_rows = [row for row in all_rows if row.get("task") in allowed_tasks]
        if not all_rows:
            raise RuntimeError(f"tasks matched no rows in {args.split}: {task_names}")
    split_digest = C.split_checksum(all_rows)
    rows = all_rows[args.shard :: args.nshards]
    out = C.work_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapter_path = None if args.adapter == "none" else C.work_path(args.adapter)
    video_size = C.video_size_for(args.frame_budget, args.video_max_pixels)
    motion_hints = None
    if args.motion_hints:
        hints_path = C.work_path(args.motion_hints)
        with hints_path.open() as handle:
            motion_hints = json.load(handle)
    if adapter_path is not None:
        C.assert_score_video_contract(
            adapter_path,
            frame_budget=args.frame_budget,
            video_max_pixels=args.video_max_pixels,
        )
    expected_manifest = {
        "schema_version": 1,
        "adapter": "none" if adapter_path is None else str(adapter_path.resolve()),
        "adapter_checksum": "none" if adapter_path is None else C.path_checksum(adapter_path),
        "adapter_merge_mode": (
            "none" if adapter_path is None else "unmerged" if args.no_merge_adapters else "merged"
        ),
        "model_id": C.MODEL_ID,
        "model_revision": C.MODEL_REV,
        "dataset_revision": C.dataset_revision(),
        "split": args.split,
        "split_checksum": split_digest,
        "offsets": offsets,
        "rots": args.rots,
        "frame_sampling": args.frame_sampling,
        "frame_budget": args.frame_budget,
        "tasks": task_names,
        "video_max_pixels": args.video_max_pixels,
        "video_size": video_size,
        "motion_hints": None if args.motion_hints is None else str(C.work_path(args.motion_hints).resolve()),
        "shard": args.shard,
        "nshards": args.nshards,
    }
    manifest_path = C.score_manifest_path(out)
    if manifest_path.exists():
        actual_manifest = json.loads(manifest_path.read_text())
        actual_manifest.setdefault("frame_sampling", C.DEFAULT_FRAME_SAMPLING)
        actual_manifest.setdefault("frame_budget", C.MAX_FRAMES)
        actual_manifest.setdefault("tasks", [])
        actual_manifest.setdefault("video_max_pixels", None)
        actual_manifest.setdefault("video_size", C.VIDEO_SIZE)
        actual_manifest.setdefault("motion_hints", None)
        actual_manifest.setdefault(
            "adapter_merge_mode",
            "none" if adapter_path is None else "merged",
        )
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

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video scoring")
    attn_implementation = "sdpa"
    C.configure_attention_backend(attn_implementation)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    device = torch.device("cuda")
    model.to(device)
    if adapter_path is not None:
        model = C.load_peft_for_inference(model, adapter_path, merge=not args.no_merge_adapters)
        model.to(device)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model left CUDA before scoring; refusing a CPU adapter merge")
    model.eval()
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(args.rots)]

    with out.open("a") as stream, torch.inference_mode():
        for row in rows:
            if row["qa_id"] in done:
                continue
            if stop_file and Path(stop_file).exists():
                print(f"stop requested; resume from {out}")
                break
            print(f"scoring {row['qa_id']}", flush=True)
            views = []
            for offset in offsets:
                frames, meta = C.get_video(
                    row["video_path"], offset, mode=args.frame_sampling, frame_budget=args.frame_budget
                )
                motion_context = C.motion_hint_text(motion_hints, row["video_path"])
                if motion_hints is not None:
                    if not motion_context:
                        raise KeyError(f"motion hint missing for scoring video {row['video_path']!r}")
                    motion_context = S.checked_motion_hint(motion_context)
                for permutation in rotations:
                    encoded, actual_perm = C.encode(
                        processor,
                        row,
                        frames,
                        meta,
                        permutation,
                        temporal_context=motion_context,
                        video_size=video_size,
                    )
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
