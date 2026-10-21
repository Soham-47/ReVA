"""Prepare the frozen Phase 5 crop-evidence packet from Phase 4 artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase5_crop_ablation"
INPUT = RUN / "input"
SELECTION = ROOT / "tasks/qcsrr_final_benchmark_selection.json"
PHASE4 = ROOT / "runs/phase4_qcsrr"
RESULTS = PHASE4 / "results_resumed-004"
RANDOM_SEED = 20261018
MAX_CROPS = 4


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_gzip(path: Path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def crop_box(row: dict) -> dict:
    box = row["crop_box_pixels"]
    return {key: int(box[key]) for key in ("left", "top", "right", "bottom")}


def crop_size(box: dict) -> tuple[int, int]:
    return box["right"] - box["left"], box["bottom"] - box["top"]


def bounded_box(cx: float, cy: float, width: int, height: int, frame_width: int, frame_height: int) -> dict:
    width = min(max(1, int(width)), frame_width)
    height = min(max(1, int(height)), frame_height)
    left = min(max(0, round(cx - width / 2)), frame_width - width)
    top = min(max(0, round(cy - height / 2)), frame_height - height)
    return {"left": left, "top": top, "right": left + width, "bottom": top + height}


def draw_sheet(manifest: dict) -> Path:
    out = RUN / "contact_sheets" / "phase5_crop_pairs.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = manifest["questions"]
    thumb_w, thumb_h = 172, 110
    label_h, row_h = 38, 280
    width = 1740
    height = 70 + len(rows) * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), "Phase 5 matched auxiliary crops: random control (E1) vs SigLIP2 retrieved (E2)", fill="black", font=font)
    for qi, question in enumerate(rows):
        y = 45 + qi * row_h
        draw.rectangle((8, y, width - 8, y + row_h - 4), outline="#888", width=1)
        draw.text((18, y + 6), f"{question['qa_id']}  {question['question'][:175]}", fill="black", font=font)
        for ci, slot in enumerate(question["slots"]):
            x1 = 20 + ci * 420
            for arm, xpos in (("random", x1), ("retrieved", x1 + 205)):
                record = slot.get(arm)
                draw.text((xpos, y + 28), f"{arm.upper()}  t={record['timestamp_s']:.3f}s" if record else f"{arm.upper()}  no matched crop", fill="#222", font=font)
                if not record:
                    continue
                image_path = INPUT / record["local_path"]
                with Image.open(image_path) as source:
                    source = source.convert("RGB")
                    source.thumbnail((thumb_w, thumb_h))
                    px = xpos
                    py = y + 48
                    canvas.paste(source, (px, py))
                    draw.rectangle((px, py, px + source.width, py + source.height), outline="#222", width=1)
                    details = f"{record.get('source_frame_index')} | {record['crop_size_pixels'][0]}x{record['crop_size_pixels'][1]}"
                    if arm == "retrieved":
                        details += f" | R0 rank {record['r0_rank']} | {record.get('phrase', '')[:24]}"
                    draw.text((px, py + thumb_h + 4), details, fill="#333", font=font)
        if not question["slots"]:
            draw.text((20, y + 40), "No question-conditioned proposal crop was available.", fill="#a00", font=font)
    canvas.save(out, quality=88)
    return out


def main() -> None:
    if INPUT.exists():
        shutil.rmtree(INPUT)
    INPUT.mkdir(parents=True)
    selection = read_json(SELECTION)
    phase4_manifest = read_json(PHASE4 / "source_manifest.json")
    all_candidates = read_gzip(RESULTS / "candidates.json.gz")
    ranking_rows = read_gzip(RESULTS / "rankings.json.gz")
    with (ROOT / "data/ReVA/train.json").open() as stream:
        train_rows = json.load(stream)["QA"]
    train_by_id = {str(row["qa_id"]): row for row in train_rows}
    manifest_rows = read_json(ROOT / "data/manifest.json")
    candidate_by_id = {str(row["candidate_id"]): row for row in all_candidates}
    selected_ids = [str(item["qa_id"]) for item in selection["questions"]]
    if len(selected_ids) != 12 or len(set(selected_ids)) != 12:
        raise ValueError("Phase 5 must use the frozen 12-question, 12-video Phase 4 cohort")

    candidates_by_qa: dict[str, list[dict]] = {qa_id: [] for qa_id in selected_ids}
    for row in all_candidates:
        if row.get("qa_id") in candidates_by_qa and row.get("arm") == "question-conditioned":
            candidates_by_qa[str(row["qa_id"])].append(row)
    rankings_by_query: dict[tuple[str, str], dict] = {}
    for row in ranking_rows:
        if row.get("arm") == "question-conditioned":
            rankings_by_query[(str(row["qa_id"]), norm(row["phrase"]))] = row

    output_questions = []
    frame_checks = []
    seed_hash = hashlib.sha256(str(RANDOM_SEED).encode()).hexdigest()
    crop_root = INPUT / "crops"
    for source_q in selection["questions"]:
        qa_id = str(source_q["qa_id"])
        if qa_id not in train_by_id:
            raise KeyError(f"selected question missing from train.json: {qa_id}")
        q = train_by_id[qa_id]
        if q["question"] != source_q["question"] or q["video_path"] != source_q["video_path"]:
            raise ValueError(f"selected question/video mismatch for {qa_id}")
        source_info = phase4_manifest[qa_id]
        if source_info["video_path"] != q["video_path"]:
            raise ValueError(f"Phase 4 source mapping mismatch for {qa_id}")
        source_frames = ROOT / "runs/phase4_qcsrr" / source_info["source_frames_dir"]
        width, height = int(source_info["width"]), int(source_info["height"])
        frame_to_timestamp = {
            int(index): float(timestamp)
            for index, timestamp in zip(source_info["source_frame_indices"], source_info["source_timestamps_s"], strict=True)
        }
        ordered_phrases = [str(value) for value in source_q["entity_prompt_phrases"]]

        retrieved_specs = []
        seen_candidate_ids = set()
        phrase_status = []
        for phrase_index, phrase in enumerate(ordered_phrases):
            rank_row = rankings_by_query.get((qa_id, norm(phrase)))
            top = rank_row.get("R0", {}).get("top5", []) if rank_row else []
            prompt_candidate_count = sum(
                1
                for candidate_row in candidates_by_qa[qa_id]
                if any(norm(prov.get("prompt", "")) == norm(phrase) for prov in candidate_row.get("prompt_provenance", []))
            )
            if not top:
                phrase_status.append({
                    "phrase": phrase,
                    "proposal_generated_by_this_prompt": prompt_candidate_count > 0,
                    "prompt_candidate_count": prompt_candidate_count,
                    "ranked_pool_candidates_available": False,
                    "candidate_count": rank_row.get("candidate_count", 0) if rank_row else 0,
                })
                continue
            ranked = top[0]
            candidate_id = str(ranked["candidate_id"])
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                raise KeyError(f"ranked candidate missing from frozen pool: {candidate_id}")
            if candidate_id in seen_candidate_ids:
                phrase_status.append({
                    "phrase": phrase,
                    "proposal_generated_by_this_prompt": prompt_candidate_count > 0,
                    "prompt_candidate_count": prompt_candidate_count,
                    "ranked_pool_candidates_available": True,
                    "candidate_id": candidate_id,
                    "duplicate_selected_for_prior_phrase": True,
                })
                continue
            if candidate.get("qa_id") != qa_id or candidate.get("arm") != "question-conditioned":
                raise ValueError(f"R0 selected a candidate outside the question-conditioned pool: {candidate_id}")
            seen_candidate_ids.add(candidate_id)
            retrieved_specs.append({
                "phrase": phrase,
                "phrase_index": phrase_index,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "r0_rank": 1,
                "similarity": float(ranked["similarity"]),
            })
            phrase_status.append({
                "phrase": phrase,
                "proposal_generated_by_this_prompt": prompt_candidate_count > 0,
                "prompt_candidate_count": prompt_candidate_count,
                "ranked_pool_candidates_available": True,
                "candidate_id": candidate_id,
                "r0_rank": 1,
            })
            if len(retrieved_specs) >= MAX_CROPS:
                break

        slots = []
        used_random = set()
        for slot_index, retrieved in enumerate(retrieved_specs):
            cand = retrieved["candidate"]
            source_index = int(cand["source_frame_index"])
            timestamp_s = float(cand["timestamp"])
            mapped_timestamp = frame_to_timestamp.get(source_index)
            if mapped_timestamp is None or abs(mapped_timestamp - timestamp_s) > 1e-5:
                raise ValueError(f"candidate timestamp/source-index mismatch for {retrieved['candidate_id']}")
            frame_path = source_frames / f"frame-{source_index:06d}.jpg"
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            with Image.open(frame_path) as original:
                original = original.convert("RGB")
                if original.size != (width, height):
                    raise ValueError(f"source frame dimensions mismatch for {frame_path}")
                rb = crop_box(cand)
                if rb["left"] < 0 or rb["top"] < 0 or rb["right"] > width or rb["bottom"] > height:
                    raise ValueError(f"out-of-frame E2 crop box: {retrieved['candidate_id']}")
                e2 = original.crop((rb["left"], rb["top"], rb["right"], rb["bottom"]))
                e2_relative = Path("crops/retrieved") / qa_id / f"crop-{slot_index:02d}.jpg"
                e2_path = INPUT / e2_relative
                e2_path.parent.mkdir(parents=True, exist_ok=True)
                e2.save(e2_path, quality=95, subsampling=0)

            # Phase 4 ranks each question-derived phrase against the complete
            # question-conditioned bank, including proposals from other prompts.
            phrase_pool = candidates_by_qa[qa_id]
            if not phrase_pool:
                raise ValueError(f"no question-conditioned candidate pool for {qa_id}")
            alternatives = [row for row in phrase_pool if row["candidate_id"] != retrieved["candidate_id"]]
            rng_bytes = hashlib.sha256(f"{RANDOM_SEED}|{qa_id}|{slot_index}|{retrieved['phrase']}".encode()).digest()
            rng = random.Random(int.from_bytes(rng_bytes[:8], "big"))
            random_anchor = None
            if alternatives:
                min_frame_gap = min(abs(int(row["source_frame_index"]) - source_index) for row in alternatives)
                nearest = [row for row in alternatives if abs(int(row["source_frame_index"]) - source_index) == min_frame_gap]
                unused = [row for row in nearest if row["candidate_id"] not in used_random]
                random_anchor = rng.choice(unused or nearest)
                used_random.add(str(random_anchor["candidate_id"]))
            else:
                random_anchor = cand
            e2_width, e2_height = crop_size(rb)
            anchor_index = int(random_anchor["source_frame_index"])
            anchor_path = source_frames / f"frame-{anchor_index:06d}.jpg"
            with Image.open(anchor_path) as original:
                original = original.convert("RGB")
                anchor_centroid = random_anchor["centroid_normalized"]
                random_box = bounded_box(
                    float(anchor_centroid["x"]) * width,
                    float(anchor_centroid["y"]) * height,
                    e2_width,
                    e2_height,
                    width,
                    height,
                )
                e1 = original.crop((random_box["left"], random_box["top"], random_box["right"], random_box["bottom"]))
                e1_relative = Path("crops/random") / qa_id / f"crop-{slot_index:02d}.jpg"
                e1_path = INPUT / e1_relative
                e1_path.parent.mkdir(parents=True, exist_ok=True)
                e1.save(e1_path, quality=95, subsampling=0)

            random_timestamp = frame_to_timestamp[anchor_index]
            random_rec = {
                "local_path": str(e1_relative),
                "candidate_id": str(random_anchor["candidate_id"]),
                "anchor_candidate_id": str(random_anchor["candidate_id"]),
                "source_frame_index": anchor_index,
                "timestamp_s": random_timestamp,
                "timestamp_delta_s": round(random_timestamp - timestamp_s, 6),
                "crop_box_pixels": random_box,
                "crop_size_pixels": [e1.width, e1.height],
                "sha256": sha256(e1_path),
                "selection": "seeded_uniform_from_nearest_available_phrase-pool_candidate; size matched to E2",
            }
            retrieved_rec = {
                "local_path": str(e2_relative),
                "candidate_id": retrieved["candidate_id"],
                "phrase": retrieved["phrase"],
                "r0_rank": retrieved["r0_rank"],
                "siglip2_similarity": retrieved["similarity"],
                "source_frame_index": source_index,
                "timestamp_s": timestamp_s,
                "crop_box_pixels": rb,
                "crop_size_pixels": [e2.width, e2.height],
                "sha256": sha256(e2_path),
                "mask_sha256": cand["mask_sha256"],
                "proposal_kinds": cand.get("proposal_kinds", []),
                "prompt_provenance": cand.get("prompt_provenance", []),
                "selection": "R0 rank 1 for question-derived entity phrase",
            }
            slots.append({
                "slot_index": slot_index,
                "phrase_role": "question_entity",
                "phrase": retrieved["phrase"],
                "random": random_rec,
                "retrieved": retrieved_rec,
            })
            frame_checks.append({
                "qa_id": qa_id,
                "candidate_id": retrieved["candidate_id"],
                "source_frame_index": source_index,
                "source_timestamp_s": timestamp_s,
                "mapped_timestamp_s": mapped_timestamp,
                "exact_view_offset_0_match": True,
                "crop_dimensions": [e2.width, e2.height],
            })

        output_questions.append({
            "qa_id": qa_id,
            "video_path": q["video_path"],
            "dataset_name": q["dataset_name"],
            "task": q["task"],
            "category": q["category"],
            "selection_stratum": source_q["selection_stratum"],
            "question": q["question"],
            "entity_prompt_phrases": ordered_phrases,
            "phrase_proposal_status": phrase_status,
            "target_proposal_missing": not bool(phrase_status and phrase_status[0].get("proposal_generated_by_this_prompt")),
            "slots": slots,
            "candidate_pool_size": len(candidates_by_qa[qa_id]),
        })

    # Selection above uses only the frozen questions, Phase 4 candidate rows, and R0 rankings.
    crop_manifest = {
        "schema_version": 1,
        "experiment": "Phase 5 frozen Qwen3-VL crop-evidence ablation",
        "selection_source": "Phase 4 frozen question-conditioned R0 SigLIP2 rankings",
        "selection_does_not_use": ["gold answers", "source-video labels", "manual target coordinates", "predicted answers"],
        "selection_policy": "one R0 rank-1 crop per question-derived entity phrase, target-first, capped at four, deduplicated by candidate ID",
        "random_seed": RANDOM_SEED,
        "random_selection_policy": "For each retrieved slot, choose a different candidate uniformly from the nearest available source frame within the complete question-conditioned candidate pool; center a crop with the exact E2 crop width and height. If no alternative exists, reuse the same candidate and record it.",
        "questions": output_questions,
        "source_timestamp_validation": {"candidate_to_source_manifest_exact_matches": len(frame_checks), "checks": frame_checks},
    }
    (INPUT / "crop_manifest.json").write_text(json.dumps(crop_manifest, indent=2) + "\n")

    # Gold answers are staged separately from retrieval selection and are used only after scoring.
    qas = [{key: value for key, value in train_by_id[qa_id].items() if key != "correct_answer"} for qa_id in selected_ids]
    (INPUT / "qa_rows.json").write_text(json.dumps(qas, indent=2, ensure_ascii=False) + "\n")
    (RUN / "evaluation_key.json").write_text(json.dumps({qa_id: train_by_id[qa_id]["correct_answer"] for qa_id in selected_ids}, indent=2) + "\n")
    source_manifest_subset = {}
    phase4_selected = {}
    video_root = INPUT / "data/ReVA"
    for qa_id in selected_ids:
        q = train_by_id[qa_id]
        video_path = str(q["video_path"])
        info = manifest_rows.get(video_path)
        if info is None:
            raise KeyError(f"selected video missing from dataset manifest: {video_path}")
        source_info = phase4_manifest[qa_id]
        local_video = PHASE4 / "source_videos" / video_path
        if not local_video.is_file() or sha256(local_video) != source_info["video_sha256"]:
            raise ValueError(f"staged Phase 4 source video failed SHA-256 validation: {video_path}")
        destination = video_root / video_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_video, destination)
        source_manifest_subset[video_path] = info
        phase4_selected[qa_id] = source_info
    (INPUT / "data/manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (INPUT / "data/manifest.json").write_text(json.dumps(source_manifest_subset, indent=2) + "\n")
    (INPUT / "phase4_source_manifest.json").write_text(json.dumps(phase4_selected, indent=2) + "\n")

    adapter_src = ROOT / "runs/stageA_q3vl8b_r16/final_adapter"
    adapter_dst = INPUT / "stageA_adapter"
    adapter_dst.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(adapter_src / name, adapter_dst / name)
    (INPUT / "code").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "code/common.py", INPUT / "code/common.py")

    run_manifest = {
        "schema_version": 1,
        "phase": 5,
        "question_ids": selected_ids,
        "distinct_video_count": len({q["video_path"] for q in qas}),
        "base_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "base_model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "adapter": "runs/stageA_q3vl8b_r16/final_adapter",
        "adapter_sha256": sha256(adapter_dst / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(adapter_dst / "adapter_config.json"),
        "frozen_phase4_candidates": "runs/phase4_qcsrr/results_resumed-004/candidates.json.gz",
        "frozen_phase4_rankings": "runs/phase4_qcsrr/results_resumed-004/rankings.json.gz",
        "phase4_rankings_sha256": sha256(RESULTS / "rankings.json.gz"),
        "phase4_candidates_sha256": sha256(RESULTS / "candidates.json.gz"),
        "frame_sampling": "legacy",
        "offsets": [0.0, 0.5],
        "answer_rotations": 2,
        "scoring": "letter logits A-D, inverse-map each option rotation, mean probabilities over all four views",
        "arms": {
            "E0": "unchanged code.common.encode video-only Stage A input",
            "E1": "same video and prompt plus matched seeded random crops",
            "E2": "same video and prompt plus frozen Phase 4 R0 top-ranked per-entity crops",
        },
        "selection_seed": RANDOM_SEED,
        "retrieved_crops": sum(len(q["slots"]) for q in output_questions),
        "random_crops": sum(sum(1 for slot in q["slots"] if slot.get("random")) for q in output_questions),
        "crop_manifest_sha256": sha256(INPUT / "crop_manifest.json"),
        "input_video_sha256": {q["video_path"]: sha256(video_root / q["video_path"]) for q in qas},
        "source_timestamp_mapping_verified": True,
    }
    (INPUT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    (INPUT / "run_manifest.sha256").write_text(sha256(INPUT / "run_manifest.json") + "\n")
    manifest_copy = RUN / "crop_selection_manifest.json"
    manifest_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT / "crop_manifest.json", manifest_copy)
    sheet = draw_sheet(crop_manifest)
    print(json.dumps({
        "status": "prepared",
        "questions": len(output_questions),
        "videos": len({q["video_path"] for q in qas}),
        "retrieved_crops": run_manifest["retrieved_crops"],
        "random_crops": run_manifest["random_crops"],
        "crop_timestamp_checks": len(frame_checks),
        "adapter_sha256": run_manifest["adapter_sha256"],
        "contact_sheet": str(sheet),
        "input_bytes": sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file()),
    }, indent=2))


if __name__ == "__main__":
    main()
