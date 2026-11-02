"""Frozen Stage A scoring with training-free pilot flags (overlays, prompts, REZE).

Production ``code/score.py`` stays minimal; Brev/HPC pilots invoke this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import reze as R  # noqa: E402
import spatial_contract as SC  # noqa: E402
import visual_prompts as VP  # noqa: E402

SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
    "test": "data/ReVA/test.json",
}


def move_inputs(inputs: dict, device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _read_selection_file(path: str | None, *, label: str, split: str) -> set[str] | None:
    if path is None:
        return None
    source = C.work_path(path)
    if source.suffix.casefold() == ".json":
        try:
            payload = json.loads(source.read_text())
            section = payload["splits"][split]
            values = section["ids"] if isinstance(section, dict) else section
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} JSON must contain splits.{split}.ids") from exc
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{label} JSON split {split} must contain a string ids list")
    else:
        values = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} file must contain unique non-empty entries")
    return set(values)


def _reze_frame_yes(model, processor, frame, meta, question, *, device, yes_id, no_id):
    prompt = (
        "If there is any doubt, answer No. Output a single word Yes or No.\n"
        f"Question event: {question}\n"
        "Is this event CLEARLY occurring in this frame? Answer:"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": C.SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = C._processor_call(processor, text, frame, meta)
    allowed = {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    encoded = {key: value for key, value in encoded.items() if key in allowed}
    logits = model(**move_inputs(encoded, device)).logits[0, -1].float()
    return R.yes_from_logits(logits, yes_id, no_id)


def _pilot_flags(args) -> bool:
    return any(
        (
            args.progress_bar,
            args.camera_motion_hint,
            args.cotr_timestamp_burnin,
            args.spatial_contract,
            args.reze,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none", help="final_adapter directory, or none for zero-shot")
    ap.add_argument("--no-merge-adapters", action="store_true")
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa-ids-file", help="JSON split manifest or newline-delimited qa_id list")
    ap.add_argument("--offsets", default=",".join(str(value) for value in C.DEFAULT_SCORE_OFFSETS))
    ap.add_argument("--rots", type=int, default=C.DEFAULT_SCORE_ROTS)
    ap.add_argument("--frame-sampling", choices=C.FRAME_SAMPLING_MODES, default=C.DEFAULT_FRAME_SAMPLING)
    ap.add_argument("--frame-budget", type=int, default=C.MAX_FRAMES)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--progress-bar", action="store_true")
    ap.add_argument("--camera-motion-hint", action="store_true")
    ap.add_argument(
        "--cotr-timestamp-burnin",
        action="store_true",
        help="CoTR-style mm:ss + [t s] glyphs burned into the top-right of each frame",
    )
    ap.add_argument("--spatial-contract", action="store_true")
    ap.add_argument("--reze", action="store_true")
    ap.add_argument("--reze-alpha", type=float, default=1.0)
    args = ap.parse_args()

    if not _pilot_flags(args):
        ap.error("at least one pilot flag is required")
    if args.spatial_contract and args.reze:
        ap.error("--spatial-contract and --reze are separate pilots")
    if args.progress_bar and args.cotr_timestamp_burnin:
        ap.error("use either progress-bar or cotr-timestamp-burnin, not both")
    if args.reze_alpha < 0:
        ap.error("--reze-alpha must be non-negative")
    if (args.qa_ids_file is None) and _pilot_flags(args):
        pass  # full split allowed for cotr full TG runs
    if not 1 <= args.rots <= 4 or not 0 <= args.shard < args.nshards:
        ap.error("rots must be 1..4, and shard must be in [0, nshards)")

    offsets = [float(value) for value in args.offsets.split(",") if value.strip()]
    if not offsets or any(not 0 <= value < 1 for value in offsets):
        ap.error("offsets must be comma-separated values in [0, 1)")

    split_path = C.work_path(SPLITS[args.split])
    with split_path.open() as f:
        all_rows = json.load(f)["QA"]
    split_digest = C.split_checksum(all_rows)
    rows = all_rows[args.shard :: args.nshards]
    qa_ids = _read_selection_file(args.qa_ids_file, label="QA IDs", split=args.split)
    if qa_ids is not None:
        rows = [row for row in rows if row["qa_id"] in qa_ids]
        if not rows:
            raise SystemExit(f"no rows selected for split {args.split}")

    out = C.work_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adapter_path = None if args.adapter == "none" else C.work_path(args.adapter)

    output_kind = "pilot_score"
    if args.progress_bar or args.camera_motion_hint:
        output_kind = "visual_prompt_score"
    elif args.cotr_timestamp_burnin:
        output_kind = "cotr_timestamp_score"
    elif args.spatial_contract:
        output_kind = "spatial_contract_score"
    elif args.reze:
        output_kind = "reze_score"

    expected_manifest = {
        "schema_version": 1,
        "output_kind": output_kind,
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
        "shard": args.shard,
        "nshards": args.nshards,
        "pilot_qa_ids_file": args.qa_ids_file,
        "pilot_qa_ids_file_checksum": C.path_checksum(args.qa_ids_file) if args.qa_ids_file else None,
        "selected_questions": len(rows),
        "visual_prompts": {
            "progress_bar": bool(args.progress_bar),
            "camera_motion_hint": bool(args.camera_motion_hint),
            "cotr_timestamp_burnin": bool(args.cotr_timestamp_burnin),
        },
        "spatial_contract": bool(args.spatial_contract),
        "reze": {"alpha": args.reze_alpha} if args.reze else None,
    }

    manifest_path = C.score_manifest_path(out)
    if manifest_path.exists():
        actual_manifest = json.loads(manifest_path.read_text())
        for key in ("frame_sampling", "frame_budget", "adapter_merge_mode"):
            actual_manifest.setdefault(key, expected_manifest.get(key))
        if actual_manifest.get("output_kind") != output_kind:
            raise RuntimeError(f"manifest output_kind mismatch: {manifest_path}")
        if actual_manifest != expected_manifest:
            raise RuntimeError(f"score manifest mismatch: {manifest_path}")
    elif out.exists() and out.stat().st_size:
        raise RuntimeError(f"existing score file has no manifest: {out}")
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
    yes_id = no_id = None
    if args.reze:
        yes_id, no_id = R.yes_no_ids(processor.tokenizer)
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
    if adapter_path is not None:
        model = C.load_peft_for_inference(model, adapter_path, merge=not args.no_merge_adapters)
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
            print(f"scoring {row['qa_id']}", flush=True)
            views = []
            reze_frames = None
            reze_meta = None
            for offset in offsets:
                frames, meta = C.get_video(
                    row["video_path"], offset, mode=args.frame_sampling, frame_budget=args.frame_budget
                )
                if args.reze and reze_frames is None:
                    reze_frames, reze_meta = frames, meta
                motion_hint = None
                if args.camera_motion_hint:
                    motion_hint = VP.camera_motion_hint(frames, meta)
                if args.progress_bar:
                    frames, meta = VP.overlay_progress_bar(frames, meta)
                if args.cotr_timestamp_burnin:
                    frames = VP.overlay_cotr_timestamp(frames, meta)
                temporal_context = SC.SPATIAL_CONTRACT if args.spatial_contract else motion_hint
                for permutation in rotations:
                    encoded, actual_perm = C.encode(
                        processor,
                        row,
                        frames,
                        meta,
                        permutation,
                        temporal_context=temporal_context,
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
            if args.reze:
                times = R.frame_times(reze_meta)
                duration = R.clip_duration(reze_meta)
                yes_scores = []
                for index in range(len(reze_frames)):
                    frame = reze_frames[index : index + 1]
                    frame_meta = C._video_metadata(
                        total_num_frames=int(reze_meta.total_num_frames),
                        fps=float(reze_meta.fps),
                        frames_indices=[int(reze_meta.frames_indices[index])],
                        width=int(reze_frames.shape[2]),
                        height=int(reze_frames.shape[1]),
                        duration=duration,
                    )
                    yes_scores.append(
                        _reze_frame_yes(
                            model,
                            processor,
                            frame,
                            frame_meta,
                            row["question"],
                            device=device,
                            yes_id=yes_id,
                            no_id=no_id,
                        )
                    )
                energy = R.option_energy(row, times, yes_scores, duration)
                fused = R.fuse_probs(mean, energy, args.reze_alpha)
                result["p_letter"] = mean
                result["reze_times"] = times
                result["reze_yes"] = yes_scores
                result["reze_energy"] = energy
                result["reze_alpha"] = args.reze_alpha
                result["p"] = fused
                result["pred"] = C.LETTERS[int(np.argmax(fused))]
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
    print(f"scored {args.split} shard {args.shard}/{args.nshards}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
