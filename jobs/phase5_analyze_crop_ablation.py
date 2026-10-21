"""Validate and summarize the completed Phase 5 frozen-Qwen ablation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase5_crop_ablation"
OUT = RUN / "results_hpc"
ARMS = ("E0", "E1", "E2")
LETTERS = "ABCD"


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gold_margin(row: dict, gold: str) -> float:
    values = row["p"]
    index = LETTERS.index(gold)
    return float(values[index] - max(values[:index] + values[index + 1 :]))


def make_contact_sheet(qas, score_rows, key, crop_manifest):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 13) if Path(font_path).exists() else ImageFont.load_default()
    width, row_h, left = 1810, 254, 640
    canvas = Image.new("RGB", (width, row_h * len(qas) + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "Phase 5: answer outcomes with matched random (E1) and retrieved (E2) crops", fill="black", font=font)
    crops_by_id = {str(q["qa_id"]): q for q in crop_manifest["questions"]}
    for row_index, q in enumerate(qas):
        qa_id = str(q["qa_id"])
        y = 24 + row_index * row_h
        draw.rectangle((6, y, width - 6, y + row_h - 4), outline="#888")
        draw.text((14, y + 8), qa_id, fill="#111", font=font)
        question = "\n".join(textwrap.wrap(q["question"], width=43)[:4])
        draw.multiline_text((14, y + 34), question, fill="#222", font=small, spacing=3)
        draw.text((14, y + 136), f"Gold: {key[qa_id]}", fill="#111", font=small)
        for index, arm in enumerate(ARMS):
            row = score_rows[arm][qa_id]
            pred = row["pred"]
            correct = pred == key[qa_id]
            p_gold = row["p"][LETTERS.index(key[qa_id])]
            margin = gold_margin(row, key[qa_id])
            color = "#176b2c" if correct else "#a32323"
            draw.text((14 + index * 192, y + 188), f"{arm}: {pred} {'✓' if correct else '✗'}", fill=color, font=small)
            draw.text((14 + index * 192, y + 207), f"p(gold)={p_gold:.3f} Δ={margin:+.3f}", fill="#333", font=small)
        crop_question = crops_by_id[qa_id]
        for slot_index, slot in enumerate(crop_question["slots"][:4]):
            x = left + slot_index * 292
            phrase = str(slot.get("retrieved", {}).get("phrase", ""))[:36]
            draw.text((x, y + 7), f"slot {slot_index + 1}: {phrase}", fill="#111", font=small)
            for arm_key, dx, label in (("random", 0, "E1 random"), ("retrieved", 144, "E2 retrieved")):
                record = slot.get(arm_key)
                draw.text((x + dx, y + 28), label, fill="#444", font=small)
                if not record:
                    draw.text((x + dx, y + 48), "no crop", fill="#a00", font=small)
                    continue
                path = RUN / "input" / record["local_path"]
                with Image.open(path) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((136, 136))
                    canvas.paste(thumb, (x + dx, y + 48))
                    draw.rectangle((x + dx, y + 48, x + dx + thumb.width, y + 48 + thumb.height), outline="#444")
                detail = f"t={record['timestamp_s']:.2f}s {record['crop_size_pixels'][0]}x{record['crop_size_pixels'][1]}"
                draw.text((x + dx, y + 190), detail, fill="#444", font=small)
    path = RUN / "contact_sheets" / "phase5_answer_outcomes.jpg"
    canvas.save(path, quality=90)
    return path


def main():
    qas = read_json(RUN / "input/qa_rows.json")
    manifest = read_json(RUN / "input/run_manifest.json")
    key = read_json(RUN / "evaluation_key.json")
    crop_manifest = read_json(RUN / "input/crop_manifest.json")
    phase4_sources = read_json(RUN / "input/phase4_source_manifest.json")
    reviews = read_json(ROOT / "runs/phase4_qcsrr/annotations/source_review_primary.json")["questions"]
    crop_audit = {
        "train_014154": "Both retrieved regions show the commercial-looking building and apartment-complex scene at coarse scale; this supports category/context, not exact boundaries.",
        "train_005918": "The retrieved green-space crop is narrow but relevant; the roundabout and road crops are broad contextual views. The source review says the island has no stable orientation, so these do not make the answer geometry assessable.",
        "train_011679": "Retrieved crops show the mall/theater block and parking context, but signage and building boundaries are not consistently legible at crop scale; the comparison still relies on the original video and broad scene context.",
        "train_012957": "The vehicle crop is only 24×21 pixels and cannot independently identify the target set. Intersection and green-roof-building crops provide context, but the source review could not uniquely verify the named building.",
        "train_008503": "The retrieved white-roof building crop is relevant and large enough to show the building; its first appearance is nevertheless left-censored at clip start.",
        "train_008163": "The construction/cleared parcel is represented by a useful contextual crop. Random and retrieved crops are nearly the same view, so this pair is a weak semantic-selection contrast.",
        "train_001478": "The bus crop is 49×35 pixels: bus-like content is present, but the crop cannot establish the question's white/red identity, consistent with the source-review ambiguity.",
        "train_010200": "The baseball-field crop is scene-relevant and readable, but the video contains multiple diamonds; the singular field identity remains ambiguous.",
        "train_003224": "The crops show a field-side fence/dugout-like structure, not enough detail to certify the structure type or unique left-of relation; the source review marks that instance ambiguous.",
        "train_002092": "The retrieved contextual regions show the vegetated central median and adjacent road. They are scene-relevant boxes, not precise land-cover masks.",
        "train_012936": "The motorcycle crop is only 12×19 pixels and does not resolve a unique motorcycle; neither crop representation fixes the source-level identity failure.",
        "train_014264": "The vehicle crop is only 20×22 pixels while the highway crop is broad context. This supports group-level scene context, not individual vehicle identity or motion tracking.",
    }
    question_ids = [str(row["qa_id"]) for row in qas]
    if len(question_ids) != 12 or len(set(question_ids)) != 12:
        raise ValueError("expected the fixed 12-question cohort")
    score_rows = {}
    mapping_counts = {arm: 0 for arm in ("E1", "E2")}
    for arm in ARMS:
        rows = read_jsonl(OUT / f"scores_{arm}.jsonl")
        if len(rows) != 12 or {str(row["qa_id"]) for row in rows} != set(question_ids):
            raise ValueError(f"incomplete/mismatched rows for {arm}")
        score_rows[arm] = {str(row["qa_id"]): row for row in rows}
        for row in rows:
            if len(row["views"]) != 4 or abs(sum(row["p"]) - 1.0) > 1e-5:
                raise ValueError(f"bad view count or probability mass: {arm}/{row['qa_id']}")
            if arm != "E0":
                qa_source = phase4_sources[row["qa_id"]]
                source_times = {
                    int(index): float(timestamp)
                    for index, timestamp in zip(
                        qa_source["source_frame_indices"], qa_source["source_timestamps_s"], strict=True
                    )
                }
                for view in row["views"]:
                    sampled = set(int(value) for value in view["sampled_source_frame_indices"])
                    for mapping in view["crop_source_mapping"]:
                        source_index = int(mapping["source_frame_index"])
                        if source_index not in source_times or abs(source_times[source_index] - float(mapping["timestamp_s"])) > 1e-5:
                            raise ValueError(f"crop timestamp mismatch in {arm}/{row['qa_id']}")
                        if bool(mapping["also_in_this_view"]) != (source_index in sampled):
                            raise ValueError(f"crop/source-frame membership mismatch in {arm}/{row['qa_id']}")
                        mapping_counts[arm] += 1
    if any(score_rows[arm][qid]["manifest_digest"] != (RUN / "input/run_manifest.sha256").read_text().strip() for arm in ARMS for qid in question_ids):
        raise ValueError("score rows do not match the frozen run manifest")
    if set(key) != set(question_ids):
        raise ValueError("answer key does not match the frozen cohort")

    arm_metrics = {}
    for arm in ARMS:
        correct = [qid for qid in question_ids if score_rows[arm][qid]["pred"] == key[qid]]
        margins = [gold_margin(score_rows[arm][qid], key[qid]) for qid in question_ids]
        gold_probs = [score_rows[arm][qid]["p"][LETTERS.index(key[qid])] for qid in question_ids]
        arm_metrics[arm] = {
            "correct": len(correct), "total": len(question_ids), "accuracy": len(correct) / len(question_ids),
            "correct_qa_ids": correct, "mean_gold_probability": sum(gold_probs) / len(gold_probs),
            "mean_gold_margin": sum(margins) / len(margins),
        }

    pairs = {}
    for base, other in (("E0", "E1"), ("E0", "E2"), ("E1", "E2")):
        fixes = [qid for qid in question_ids if score_rows[base][qid]["pred"] != key[qid] and score_rows[other][qid]["pred"] == key[qid]]
        regressions = [qid for qid in question_ids if score_rows[base][qid]["pred"] == key[qid] and score_rows[other][qid]["pred"] != key[qid]]
        pairs[f"{base}_to_{other}"] = {"fixes": fixes, "regressions": regressions, "unchanged_correct": [qid for qid in question_ids if score_rows[base][qid]["pred"] == score_rows[other][qid]["pred"] == key[qid]], "unchanged_wrong": [qid for qid in question_ids if score_rows[base][qid]["pred"] == score_rows[other][qid]["pred"] != key[qid]]}

    task_groups = {}
    per_question = []
    for q in qas:
        qid = str(q["qa_id"])
        task = str(q["task"])
        group = task_groups.setdefault(task, {arm: {"correct": 0, "total": 0} for arm in ARMS})
        item = {
            "qa_id": qid, "task": task, "question": q["question"], "gold": key[qid],
            "assessability": reviews[qid].get("assessability", {}),
            "source_only_notes": reviews[qid].get("source_only_notes", ""),
            "manual_crop_audit_posthoc": crop_audit[qid],
            "arms": {},
        }
        for arm in ARMS:
            row = score_rows[arm][qid]
            correct = row["pred"] == key[qid]
            group[arm]["total"] += 1
            group[arm]["correct"] += int(correct)
            item["arms"][arm] = {
                "prediction": row["pred"], "correct": correct,
                "gold_probability": row["p"][LETTERS.index(key[qid])],
                "gold_margin": gold_margin(row, key[qid]),
            }
        per_question.append(item)

    for task, group in task_groups.items():
        for arm in ARMS:
            group[arm]["accuracy"] = group[arm]["correct"] / group[arm]["total"]

    summary = read_json(OUT / "hpc_run_summary.json")
    crop_pairs = []
    for q in crop_manifest["questions"]:
        for slot in q["slots"]:
            random_row, retrieved_row = slot.get("random"), slot.get("retrieved")
            if not random_row or not retrieved_row:
                continue
            random_hash = hashlib.sha256((RUN / "input" / random_row["local_path"]).read_bytes()).hexdigest()
            retrieved_hash = hashlib.sha256((RUN / "input" / retrieved_row["local_path"]).read_bytes()).hexdigest()
            crop_pairs.append({
                "qa_id": q["qa_id"],
                "exact_timestamp_match": abs(float(random_row["timestamp_s"]) - float(retrieved_row["timestamp_s"])) < 1e-6,
                "timestamp_delta_s": abs(float(random_row["timestamp_s"]) - float(retrieved_row["timestamp_s"])),
                "same_candidate_id": random_row.get("candidate_id") == retrieved_row.get("candidate_id"),
                "byte_identical_crop": random_hash == retrieved_hash,
            })
    crop_pair_summary = {
        "matched_slots": len(crop_pairs),
        "exact_timestamp_matches": sum(row["exact_timestamp_match"] for row in crop_pairs),
        "same_candidate_id": sum(row["same_candidate_id"] for row in crop_pairs),
        "byte_identical_crop_pairs": sum(row["byte_identical_crop"] for row in crop_pairs),
        "mean_timestamp_delta_s": sum(row["timestamp_delta_s"] for row in crop_pairs) / len(crop_pairs),
        "source_mapping_records_checked_per_arm": mapping_counts,
        "question_conditioned_candidate_pool_sizes": {
            row["qa_id"]: row["candidate_pool_size"] for row in crop_manifest["questions"]
        },
        "entity_prompts_without_prompt_specific_proposals": [
            {"qa_id": row["qa_id"], "phrase": status["phrase"]}
            for row in crop_manifest["questions"]
            for status in row["phrase_proposal_status"]
            if not status["proposal_generated_by_this_prompt"]
        ],
    }
    metrics = {
        "run_manifest_digest": (RUN / "input/run_manifest.sha256").read_text().strip(),
        "question_ids": question_ids, "arm_metrics": arm_metrics, "paired_transitions": pairs,
        "task_metrics": task_groups, "per_question": per_question,
        "crop_pool_and_mapping": crop_pair_summary,
        "manual_crop_audit_posthoc": crop_audit,
        "resources": {key: summary[key] for key in (
            "gpu_name", "gpu_count", "slurm_job_id", "model_load_seconds",
            "model_load_peak_cuda_allocated_bytes", "model_load_peak_cuda_reserved_bytes",
            "peak_cuda_memory_by_arm", "inference_plus_preprocessing_seconds_by_arm",
            "forward_seconds_by_arm", "token_totals_by_arm", "gpu_scoring_elapsed_seconds",
            "processor_preflight", "adapter_sha256", "model_revision",
        )},
        "limitations": [
            "12 selected training questions are diagnostic examples, not a held-out estimate.",
            "Post-hoc crop-content judgments were not blinded to selected regions or Qwen outcomes; they are qualitative only.",
            "Source-review assessability is used to separate objective subsets; official answer accuracy is shown for all questions.",
        ],
    }
    (OUT / "phase5_evaluation.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    contact = make_contact_sheet(qas, score_rows, key, crop_manifest)

    def fmt(arm, qid):
        row = score_rows[arm][qid]
        return f"{row['pred']} {'✓' if row['pred'] == key[qid] else '✗'}; p(gold) {row['p'][LETTERS.index(key[qid])]:.3f}; margin {gold_margin(row,key[qid]):+.3f}"

    report = [
        "# Phase 5 — Frozen Qwen3-VL crop-evidence ablation",
        "",
        "## Result",
        "",
        f"On the 12-question training diagnostic cohort, E0 scored **{arm_metrics['E0']['correct']}/12**, E1 **{arm_metrics['E1']['correct']}/12**, and E2 **{arm_metrics['E2']['correct']}/12**. E2 matched E0; retrieved crops did not improve accuracy over the video-only baseline. E1 was one answer lower.",
        "",
        "| Arm | Correct | Accuracy | Mean p(gold) | Mean gold margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = arm_metrics[arm]
        report.append(f"| {arm} | {m['correct']}/12 | {m['accuracy']:.1%} | {m['mean_gold_probability']:.3f} | {m['mean_gold_margin']:+.3f} |")
    report.extend(["", "## Paired changes", "", "| Comparison | Wrong→correct | Correct→wrong | Interpretation |", "|---|---|---|---|"])
    for base, other in (("E0", "E1"), ("E0", "E2"), ("E1", "E2")):
        pair = pairs[f"{base}_to_{other}"]
        fixes = ", ".join(pair["fixes"]) or "none"
        regs = ", ".join(pair["regressions"]) or "none"
        report.append(f"| {base} → {other} | {fixes} | {regs} | fixes {len(pair['fixes'])}; regressions {len(pair['regressions'])} |")
    report.extend(["", "## Per-question scores", "", "| QA ID | Task | Gold | E0 | E1 | E2 |", "|---|---|---:|---|---|---|"])
    for q in qas:
        qid = str(q["qa_id"])
        report.append(f"| {qid} | {q['task']} | {key[qid]} | {fmt('E0',qid)} | {fmt('E1',qid)} | {fmt('E2',qid)} |")
    report.extend(["", "## Task strata", "", "| Task | Questions | E0 | E1 | E2 |", "|---|---:|---:|---:|---:|"])
    for task, group in task_groups.items():
        report.append(f"| {task} | {group['E0']['total']} | {group['E0']['correct']}/{group['E0']['total']} | {group['E1']['correct']}/{group['E1']['total']} | {group['E2']['correct']}/{group['E2']['total']} |")
    resources = metrics["resources"]
    report.extend([
        "", "## Source-video assessability", "",
        "Independent Phase 4 source reviews were completed before model calls. Official QA correctness is still reported for all 12; these notes identify where visual ground truth is assessable:", "",
        "| QA ID | Category | Instance | Geometry | Temporal | Source review note |", "|---|---|---|---|---|---|",
    ])
    for item in per_question:
        a = item["assessability"]
        note = item["source_only_notes"].replace("|", "/")
        report.append(f"| {item['qa_id']} | {a.get('category')} | {a.get('instance')} | {a.get('geometry')} | {a.get('temporal')} | {note} |")
    report.extend([
        "",
        "At the operation level, source review marked 2/4 geometric questions assessable (`train_014154`, `train_011679`): E0/E2 answer both correctly; E1 gets one of two because of the `train_011679` regression. Only `train_008503` has assessable sampled temporal presence, and it is left-censored at clip start; all arms choose the sampled answer. The median scene-region question `train_002092` is assessable and all arms are correct. Other temporal references and the structure/field or group-motion cases retain the source-review ambiguity shown above.",
        "", "## Crop evidence and interpretation", "",
        f"The paired crop sheet shows all supplied random/retrieved regions and their source timestamps alongside answer outcomes. There were {crop_pair_summary['matched_slots']} crop slots across 12 questions; {crop_pair_summary['exact_timestamp_matches']} E1/E2 pairs use the same source timestamp and the remaining pairs use the nearest available frame. The paired crops match dimensions; no random/retrieved pair reused the same candidate ID, while {crop_pair_summary['byte_identical_crop_pairs']} pairs were pixel-identical due to different proposals yielding the same crop. All {mapping_counts['E1']} E1 and {mapping_counts['E2']} E2 crop-to-source-frame timestamp records matched the Phase 4 source map, and the offset-view membership flags matched the Stage A sample indices. The source timestamps remain explicitly labeled even when they do not occur in the offset-specific 16-frame sample.",
        "",
        "The 12 per-question question-conditioned pools contained 1,497 candidate rows total (4–552 per question). Three entity prompts produced no prompt-specific proposals: `intersection` in `train_012957`, `baseball field` in `train_003224`, and `land-cover region` in `train_002092`. The shared pool still supplied fallback crops for all three; the target-specific land-cover proposal was absent for `train_002092`.",
        "",
        "Manual inspection of the crop sheet is post hoc and not blinded: some retrieved regions are useful scene context (roundabout, mall, median, baseball field, highway), while several small-object crops (bus, motorcycle, some vehicles) are tiny and are not independently identifiable from the crop alone. The original 16 sampled video frames remain present in E1/E2. The cohort therefore tests whether crops help the frozen model, not whether each crop is a valid target label.",
        "",
        "E2 fixed the E1 regression on `train_011679` (AMC/Cinemark/parking distance), but did not fix any E0 error. E0 and E2 were both wrong only on `train_012936`, whose source review says no unique motorcycle is identifiable at 640×360. E1 additionally regressed `train_011679`. Across all questions, mean gold probability and mean gold margin were lower in both crop arms than E0, indicating no evidence of a general confidence benefit.",
        "",
        "## Runtime, memory and tokens", "",
        f"- Model: `Qwen/Qwen3-VL-8B-Instruct` revision `{resources['model_revision']}` with the frozen Stage A adapter SHA-256 `{resources['adapter_sha256']}`.",
        f"- Hardware: {resources['gpu_name']} × {resources['gpu_count']} on Slurm job `{resources['slurm_job_id']}`.",
        f"- Model/adapter load and placement: {resources['model_load_seconds']:.1f} s; scoring plus preprocessing: {resources['gpu_scoring_elapsed_seconds']:.1f} s for 144 forward calls.",
        f"- Peak CUDA allocation: E0 {resources['peak_cuda_memory_by_arm']['E0']['allocated_bytes']/2**30:.2f} GiB; E1/E2 {resources['peak_cuda_memory_by_arm']['E1']['allocated_bytes']/2**30:.2f} GiB. Peak reserved was {resources['peak_cuda_memory_by_arm']['E1']['reserved_bytes']/2**30:.2f} GiB for crop arms.",
        f"- Per arm: E0 {resources['token_totals_by_arm']['E0']['input_tokens']:,} input tokens and 0 auxiliary-image tokens; E1/E2 each {resources['token_totals_by_arm']['E1']['input_tokens']:,} input tokens, including {resources['token_totals_by_arm']['E1']['auxiliary_image_tokens']:,} auxiliary-image vision tokens. Each arm had 48 forward calls and 96,000 video vision tokens.",
        "- Slurm allocation: 27:33 (27.55 H200 GPU-minutes), 12 CPUs and 96 GB requested; batch MaxRSS 30,866,748 KB (29.4 GiB), total CPU time 5:25. Slurm records `billing=12` as an internal resource TRES; the account exposes no monetary charge, so no dollar cost is claimed.",
        "",
        "## Post-hoc crop-content audit",
        "",
        "I reviewed the displayed E1/E2 crops after the run. These are qualitative, non-blinded judgments and do not change the frozen rankings or answer scores:",
        "",
        "| QA ID | Retrieved-crop assessment |",
        "|---|---|",
    ])
    for q in qas:
        qid = str(q["qa_id"])
        report.append(f"| {qid} | {crop_audit[qid]} |")
    report.extend([
        "",
        "## Conclusion and limits", "",
        "This diagnostic does not justify held-out Qwen validation yet. The baseline is already 11/12 on a small, selected training set, leaving little headroom; E2 ties it, E1 is worse, and crop arms lower average gold probability/margin. The unchanged baseline plus image-plus-video framing is safe enough to retain as a hypothesis, but the current R0 crop selection has no demonstrated downstream utility. Because several questions are visually ambiguous or censored, these counts are not a generalization estimate. A next run should be considered only after a predeclared, independently selected held-out `valT` set and a refined evidence format for small objects; do not infer improvement from this cohort.",
        "",
        f"## Artifacts\n\n- Result contact sheet: `{contact.relative_to(ROOT)}`\n- Raw score rows and resource summary: `{(OUT / 'scores_E0.jsonl').relative_to(ROOT)}`, `{(OUT / 'scores_E1.jsonl').relative_to(ROOT)}`, `{(OUT / 'scores_E2.jsonl').relative_to(ROOT)}`, `{(OUT / 'hpc_run_summary.json').relative_to(ROOT)}`\n- Evaluation JSON and Slurm accounting: `{(OUT / 'phase5_evaluation.json').relative_to(ROOT)}`, `runs/phase5_crop_ablation/hpc_accounting.txt`\n- Original E1/E2 crop sheet: `runs/phase5_crop_ablation/contact_sheets/phase5_crop_pairs.jpg`",
    ])
    (RUN / "PHASE5_FINAL_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"status": "complete", "metrics": arm_metrics, "paired": pairs, "contact_sheet": str(contact), "report": str(RUN / 'PHASE5_FINAL_REPORT.md')}, indent=2))


if __name__ == "__main__":
    main()
