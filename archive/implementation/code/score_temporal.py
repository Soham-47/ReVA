"""Question-aware coarse-to-fine temporal scoring for the approved Phase 2 run."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from bisect import bisect_left
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OFFSETS = C.DEFAULT_SCORE_OFFSETS
ROTATIONS = [list(C.LETTERS), list(C.LETTERS[1:] + C.LETTERS[:1])]
SELECTOR_VERSION = "qwen3vl-question-only-coarse-bracket-v1"
MAX_RUNTIME_SECONDS = 11 * 60 * 60


def move_inputs(inputs: dict, device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    C.read_completed_jsonl(path)
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["qa_id"]] = row
    return rows


def _timelines(manifest: dict, video_path: str) -> tuple[list[float], float]:
    info = manifest[video_path]
    fps = max(float(info["fps"]), 1e-6)
    times = info.get("frame_timestamps")
    if not isinstance(times, list) or len(times) != int(info["n_frames"]):
        times = [index / fps for index in range(int(info["n_frames"]))]
    times = [float(value) for value in times]
    if any(not math.isfinite(value) for value in times) or any(a > b for a, b in zip(times, times[1:])):
        raise ValueError(f"invalid presentation timestamps for {video_path}")
    return times, float(info.get("duration", len(times) / fps))


def _nearest_index(times: list[float], target: float) -> int:
    pos = bisect_left(times, target)
    if pos <= 0:
        return 0
    if pos >= len(times):
        return len(times) - 1
    return pos - 1 if target - times[pos - 1] <= times[pos] - target else pos


def _select_indices(
    n_frames: int, fps: float, offset: float, times: list[float], start_s: float, end_s: float
) -> dict:
    legacy = C.sample_index_list(n_frames, fps, offset, mode="legacy")
    global_indices = legacy[::2]
    if len(global_indices) != 8:
        raise RuntimeError(f"expected 8 coarse anchors, got {len(global_indices)}")
    targets = [start_s + (end_s - start_s) * i / 7 for i in range(8)]
    local_indices = [_nearest_index(times, target) for target in targets]

    chosen = set(global_indices + local_indices)
    for index in legacy:
        if len(chosen) >= C.MAX_FRAMES:
            break
        chosen.add(index)
    if len(chosen) < C.MAX_FRAMES:
        center = (start_s + end_s) / 2
        for index in sorted(range(n_frames), key=lambda i: abs(times[i] - center)):
            chosen.add(index)
            if len(chosen) >= C.MAX_FRAMES:
                break
    selected = sorted(chosen)[: C.MAX_FRAMES]
    while len(selected) < C.MAX_FRAMES:
        selected.append(legacy[len(selected) % len(legacy)])
    if len(selected) != C.MAX_FRAMES:
        raise AssertionError("temporal sampler must return 16 frames")
    return {
        "mode": "coarse8+local8",
        "coarse_indices": global_indices,
        "local_indices": local_indices,
        "frame_indices": selected,
        "frame_times_s": [times[index] for index in selected],
    }


def _localize(processor, model, device, row: dict, offset: float, manifest: dict) -> dict:
    video_path = row["video_path"]
    info = manifest[video_path]
    n_frames, fps = int(info["n_frames"]), float(info["fps"])
    times, duration = _timelines(manifest, video_path)
    legacy = C.sample_index_list(n_frames, fps, offset, mode="legacy")
    if len(legacy) < C.MAX_FRAMES:
        exact_times = [times[index] for index in legacy]
        return {
            "mode": "legacy-fallback-short-clip",
            "frame_indices": legacy,
            "frame_times_s": exact_times,
            "coarse_indices": [],
            "coarse_times_s": [],
            "fallback": True,
            "confidence": "low",
            "candidate_window_s": None,
            "raw_localizer_output": "",
            "timestamp_context": (
                f"Only {len(legacy)} legacy frames are available for this short clip; no reliable "
                "coarse-to-fine window was estimated. Legacy frame sampling was used. Source "
                f"presentation times (seconds): {json.dumps([round(value, 4) for value in exact_times])}."
            ),
        }
    coarse_indices = legacy[::2]
    frames, meta = C.get_video_by_indices(video_path, coarse_indices)
    exact_times = [times[index] for index in coarse_indices]
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You locate events in aerial video. Use only the video, the question, and the listed "
                "frame times. Do not answer a multiple-choice question. Return one JSON object only."
            )}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": (
                    f"Question: {row['question']}\n"
                    f"Exact source presentation times for these 8 coarse frames (seconds): "
                    f"{json.dumps([round(value, 4) for value in exact_times])}.\n"
                    "Estimate a short time window that contains the visual event or visibility boundary "
                    "needed by the question. For duration/visibility questions, bracket the relevant "
                    "visible interval if both ends can be supported. Return keys confidence (high, medium, "
                    "or low), window_start_s, and window_end_s; use nulls and low confidence if the target "
                    "or event cannot be localized from these frames. A window is approximate, not an exact "
                    "ground-truth time. Do not infer answer choices."
                )},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = C._processor_call(processor, text, frames, meta)
    allowed = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"}
    encoded = {key: value for key, value in raw.items() if key in allowed}
    required = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values_videos", "video_grid_thw"}
    missing = required - encoded.keys()
    if missing:
        raise KeyError(f"localizer input missing model fields: {sorted(missing)}")
    generated = model.generate(**move_inputs(encoded, device), do_sample=False, max_new_tokens=128)
    answer = processor.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    parsed = None
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    confidence = str((parsed or {}).get("confidence", "low")).lower()
    start = (parsed or {}).get("window_start_s")
    end = (parsed or {}).get("window_end_s")
    valid = confidence in {"high", "medium"}
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        point = (parsed or {}).get("candidate_time_s", (parsed or {}).get("event_time_s"))
        try:
            point = float(point)
            start, end = point - 0.5, point + 0.5
        except (TypeError, ValueError):
            valid = False
            start = end = 0.0
    valid = valid and math.isfinite(start) and math.isfinite(end) and end >= start
    if valid:
        start, end = max(0.0, min(duration, start)), max(0.0, min(duration, end))
        if end < start:
            valid = False
    if valid:
        if end - start < 7 / fps:
            center = (start + end) / 2
            start, end = max(0.0, center - 0.5), min(duration, center + 0.5)
        selection = _select_indices(n_frames, fps, offset, times, start, end)
        selection.update({
            "fallback": False,
            "confidence": confidence,
            "candidate_window_s": [start, end],
            "coarse_indices": coarse_indices,
            "coarse_times_s": exact_times,
            "raw_localizer_output": answer[:1000],
        })
        selection["timestamp_context"] = (
            "Question-only coarse scan estimated an approximate event window "
            f"[{start:.3f}, {end:.3f}] seconds ({confidence} confidence). "
            "This is a search bracket, not an exact event boundary. Source presentation times for "
            "the selected frames (seconds): "
            f"{json.dumps([round(value, 4) for value in selection['frame_times_s']])}. "
            "Use the visible frames to choose the closest supported answer."
        )
    else:
        selection = {
            "mode": "legacy-fallback",
            "frame_indices": legacy,
            "frame_times_s": [times[index] for index in legacy],
            "coarse_indices": coarse_indices,
            "coarse_times_s": exact_times,
            "fallback": True,
            "confidence": confidence if confidence in {"high", "medium", "low"} else "low",
            "candidate_window_s": None,
            "raw_localizer_output": answer[:1000],
            "timestamp_context": (
                "The question-only coarse scan could not confidently localize this event; "
                "legacy sampling was used. No supported event-time bracket is available. Exact "
                "source presentation times for the selected frames (seconds): "
                f"{json.dumps([round(value, 4) for value in [times[index] for index in legacy]])}."
            ),
        }
    return selection


def _score_variant(processor, model, device, ltr_ids, row, selection_by_offset, variant: str) -> dict:
    import torch

    views = []
    for offset in OFFSETS:
        selection = selection_by_offset[str(offset)]
        frames, meta = C.get_video_by_indices(row["video_path"], selection["frame_indices"])
        for permutation in ROTATIONS:
            encoded, actual_perm = C.encode(
                processor,
                row,
                frames,
                meta,
                permutation,
                temporal_context=selection["timestamp_context"] if variant == "T2" else None,
            )
            logits = model(**move_inputs(encoded, device)).logits[0, -1, ltr_ids].float()
            shown_prob = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
            original_prob = [0.0] * 4
            for shown_index, letter in enumerate(actual_perm):
                original_prob[C.LETTERS.index(letter)] = float(shown_prob[shown_index])
            views.append({
                "offset": offset,
                "perm": "".join(actual_perm),
                "p": original_prob,
                "frame_indices": selection["frame_indices"],
                "frame_times_s": selection["frame_times_s"],
            })
    mean = np.mean([view["p"] for view in views], axis=0).tolist()
    return {"qa_id": row["qa_id"], "p": mean, "pred": C.LETTERS[int(np.argmax(mean))], "views": views,
            "temporal_selection_by_offset": selection_by_offset}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit-per-split", type=int, help="score the first N questions of each split for timing")
    args = ap.parse_args()
    if args.limit_per_split is not None and args.limit_per_split < 1:
        ap.error("limit-per-split must be positive")
    question_set = json.loads(Path(args.questions).read_text())
    expected_revision = C.DATASET_REVISION
    if question_set.get("dataset_revision") != expected_revision:
        raise RuntimeError("question set dataset revision does not match the pinned code revision")
    rows_by_split = question_set.get("splits", {})
    if set(rows_by_split) != {"valQ", "valT"}:
        raise ValueError("question set must contain exactly valQ and valT")
    for split, expected in (("valQ", 69), ("valT", 133)):
        rows = rows_by_split[split]
        if any(row.get("task") != "Temporal Grounding" for row in rows):
            raise ValueError(f"non-temporal QA in {split} question set")
        if len({row["qa_id"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate QA IDs in {split}")
        if len(rows) != expected:
            raise ValueError(f"{split}: expected {expected} Temporal Grounding rows, found {len(rows)}")
    selection_digest = C.split_checksum(rows_by_split["valQ"] + rows_by_split["valT"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        (split, variant): output_dir / f"scores_{split}_{variant}.jsonl"
        for split in ("valQ", "valT") for variant in ("T1", "T2")
    }
    manifests = {}
    records = {}
    for key, path in output_paths.items():
        manifest_path = C.score_manifest_path(path)
        manifest = {
            "schema_version": 1,
            "arm": key[1],
            "selector": SELECTOR_VERSION,
            "model_id": C.MODEL_ID,
            "model_revision": C.MODEL_REV,
            "adapter": "runs/stageA_q3vl8b_r16/final_adapter",
            "dataset_revision": expected_revision,
            "split": key[0],
            "question_set_checksum": selection_digest,
            "offsets": list(OFFSETS),
            "option_rotations": ["ABCD", "BCDA"],
            "global_frames": 8,
            "local_frames": 8,
            "max_frames_per_offset": C.MAX_FRAMES,
            "fallback": "legacy-16-on-low-confidence",
            "explicit_timestamp_context": key[1] == "T2",
            "tranche_limit_per_split": args.limit_per_split,
        }
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous != manifest:
                raise RuntimeError(f"manifest mismatch at {manifest_path}; use a fresh output directory")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[key] = manifest
        records[key] = _jsonl(path)

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 2 scoring")
    C.configure_attention_backend("sdpa", torch)
    model_start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, revision=C.MODEL_REV, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = C.load_peft_for_inference(model, C.work_path("runs/stageA_q3vl8b_r16/final_adapter"))
    device = torch.device("cuda")
    model.to(device).eval()
    model_load_seconds = time.perf_counter() - model_start
    manifest = C.load_manifest()
    start_time = time.perf_counter()
    localization_seconds = 0.0
    localization_calls = 0
    scoring_seconds = 0.0
    totals = {}
    for split in ("valQ", "valT"):
        rows = rows_by_split[split]
        if args.limit_per_split is not None:
            rows = rows[:args.limit_per_split]
        totals[split] = len(rows)
        paths = {variant: output_paths[(split, variant)] for variant in ("T1", "T2")}
        with paths["T1"].open("a") as t1_stream, paths["T2"].open("a") as t2_stream, torch.inference_mode():
            for index, row in enumerate(rows, 1):
                if time.perf_counter() - start_time > MAX_RUNTIME_SECONDS:
                    raise TimeoutError("Phase 2 scoring stopped at the 11-hour safety limit; partial JSONL files are resumable")
                saved = [records[(split, variant)].get(row["qa_id"]) for variant in ("T1", "T2")]
                selection_by_offset = next(
                    (item["temporal_selection_by_offset"] for item in saved if item is not None), None
                )
                if selection_by_offset is None:
                    selection_by_offset = {}
                    for offset in OFFSETS:
                        before = time.perf_counter()
                        selection_by_offset[str(offset)] = _localize(
                            processor, model, device, row, offset, manifest
                        )
                        localization_calls += 1
                        localization_seconds += time.perf_counter() - before

                for variant, stream in (("T1", t1_stream), ("T2", t2_stream)):
                    if row["qa_id"] in records[(split, variant)]:
                        continue
                    before = time.perf_counter()
                    score = _score_variant(
                        processor, model, device, ltr_ids, row, selection_by_offset, variant
                    )
                    scoring_seconds += time.perf_counter() - before
                    stream.write(json.dumps(score, separators=(",", ":")) + "\n")
                    stream.flush()
                    records[(split, variant)][row["qa_id"]] = score
                print(f"{split} {index}/{len(rows)} {row['qa_id']}", flush=True)

    timing = {
        "model_load_seconds": model_load_seconds,
        "localization_seconds": localization_seconds,
        "scoring_seconds": scoring_seconds,
        "elapsed_seconds": time.perf_counter() - start_time,
        "totals": totals,
        "localization_calls": localization_calls,
    }
    (output_dir / "phase2_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
                      
