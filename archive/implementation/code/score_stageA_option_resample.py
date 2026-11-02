"""Stage A scorer with Option-Conditioned Two-Pass Resampling for Temporal Grounding.

For standard questions, evaluates 16 frames uniformly with 4-view TTA.
For questions with numeric time options (e.g. Temporal Grounding), Pass 2
densely resamples 16 frames inside the bounding option window [t_min, t_max],
effectively halving discretization error (~0.1s spacing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Bind core code/ directory
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "archive" / "implementation" / "code"))

import common as C
import option_conditioned_resampling as OCR


SPLITS = {"valQ", "valT", "val", "test"}


def move_inputs(data: Any, device: Any) -> Any:
    import torch
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_inputs(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_inputs(v, device) for v in data]
    return data


def _get_video_window(
    relative_path: str,
    window: tuple[float, float],
    offset: float = 0.0,
    work: str | os.PathLike[str] | None = None,
    k: int = 16,
):
    manifest = C.load_manifest(work)
    info = manifest[relative_path]
    indices = OCR.sample_window_indices(
        info["n_frames"], info["fps"], window, offset=offset, k=k
    )
    indices_arr = np.asarray(indices, dtype=np.int64)
    frames = C.read_frames(relative_path, indices_arr, work, fps=float(info["fps"]))
    meta = C._video_metadata(
        total_num_frames=int(info["n_frames"]),
        fps=float(info["fps"]),
        frames_indices=indices_arr.tolist(),
        width=int(info["width"]),
        height=int(info["height"]),
        duration=float(info["duration"]),
    )
    return frames, meta


def score_single_pass(
    model,
    processor,
    device,
    row: dict[str, Any],
    offsets: list[float],
    rotations: list[list[str]],
    ltr_ids: list[int],
    window: tuple[float, float] | None = None,
    frame_budget: int = 16,
) -> tuple[list[float], list[dict[str, Any]]]:
    import torch

    views = []
    for offset in offsets:
        if window is not None:
            frames, meta = _get_video_window(row["video_path"], window, offset=offset, k=frame_budget)
        else:
            frames, meta = C.get_video(row["video_path"], offset, frame_budget=frame_budget)

        for permutation in rotations:
            encoded, actual_perm = C.encode(
                processor,
                row,
                frames,
                meta,
                permutation,
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

    avg_p = np.mean([v["p"] for v in views], axis=0).tolist()
    return avg_p, views


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="final_adapter directory, or none for zero-shot")
    ap.add_argument("--no-merge-adapters", action="store_true")
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa-ids-file", help="JSON split manifest or newline-delimited qa_id list")
    ap.add_argument("--offsets", default=",".join(str(v) for v in C.DEFAULT_SCORE_OFFSETS))
    ap.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    ap.add_argument("--fusion-mode", choices=["pass2_only", "ensemble_average", "confidence_weighted", "pass1_fallback"], default="confidence_weighted")
    ap.add_argument("--window-margin-s", type=float, default=0.8)
    ap.add_argument("--force-resample-all-tg", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")

    split_path = REPO_ROOT / "data" / "splits" / f"{args.split}.json"
    with split_path.open() as f:
        meta = json.load(f)
    rows = meta["QA"] if "QA" in meta else meta

    if args.qa_ids_file:
        cohort_path = Path(args.qa_ids_file).expanduser().resolve()
        with cohort_path.open() as cf:
            cohort_data = json.load(cf)
        if isinstance(cohort_data, dict) and "splits" in cohort_data:
            allowed_ids = set(cohort_data["splits"].get(args.split, {}).get("ids", []))
        elif isinstance(cohort_data, list):
            allowed_ids = set(cohort_data)
        else:
            allowed_ids = set(cohort_data.get("ids", []))
        rows = [r for r in rows if r["qa_id"] in allowed_ids]

    expected_manifest = {
        "split": args.split,
        "n_rows": len(rows),
        "adapter": args.adapter,
        "fusion_mode": args.fusion_mode,
        "window_margin_s": args.window_margin_s,
    }
    with manifest_path.open("w") as mf:
        json.dump(expected_manifest, mf, indent=2, sort_keys=True)
        mf.write("\n")

    done = C.read_completed_jsonl(out)
    offsets = [float(x) for x in args.offsets.split(",")]
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(args.rots)]

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for video scoring")

    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)

    attn_implementation = "sdpa"
    C.configure_attention_backend(attn_implementation)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    adapter_path = None if args.adapter.strip().lower() == "none" else Path(args.adapter).expanduser().resolve()
    if adapter_path is not None:
        model = C.load_peft_for_inference(model, adapter_path, merge=not args.no_merge_adapters)

    device = torch.device("cuda")
    model.to(device).eval()

    manifest = C.load_manifest()

    print(f"Scoring {len(rows)} rows for split={args.split} into {out} (fusion={args.fusion_mode})", flush=True)

    with out.open("a") as stream, torch.inference_mode():
        for row in rows:
            qa_id = row["qa_id"]
            if qa_id in done:
                continue

            # Pass 1: Global uniform sampling
            p_pass1, views_pass1 = score_single_pass(
                model, processor, device, row, offsets, rotations, ltr_ids
            )

            is_tg = row.get("task") == "Temporal Grounding"
            window = None
            p_final = p_pass1
            pass2_applied = False

            if is_tg or args.force_resample_all_tg:
                duration = float(manifest.get(row["video_path"], {}).get("duration", 0.0) or 0.0)
                window = OCR.extract_option_time_window(
                    row.get("options", {}), duration, margin_s=args.window_margin_s
                )
                if window is not None:
                    p_pass2, views_pass2 = score_single_pass(
                        model, processor, device, row, offsets, rotations, ltr_ids, window=window
                    )
                    p_final = OCR.fuse_pass_probabilities(p_pass1, p_pass2, mode=args.fusion_mode)
                    pass2_applied = True

            pred_letter = C.LETTERS[int(np.argmax(p_final))]
            record = {
                "qa_id": qa_id,
                "p": p_final,
                "pred": pred_letter,
                "views": views_pass1,
                "pass2_applied": pass2_applied,
                "window": list(window) if window else None,
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    print(f"Completed scoring {args.split} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
