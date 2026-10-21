"""Score the isolated Replicate SAM 3 overlays with the Stage A adapter on Modal.

Replicate inference and the SAM 3 mask cache are produced by
``jobs/replicate_sam3_pilot.py``. This job stages that cache, builds only the
Grounding DINO context needed to preserve the detector control, and runs the
existing VLM scorer on the supplemental subset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN = os.environ.get("SAM3_RUN", "runs/sam3_remaining17_stageA_replicate")
SUBSET = f"{RUN}/subset_ids.json"
API_MANIFEST = f"{RUN}/manifest.json"
PRIOR_RUN = "runs/sam3_pilot_stageA_replicate"
SPLITS = ("valQ", "valT")
ADAPTER = "runs/stageA_q3vl8b_r16/final_adapter"
BASELINE = {
    "valQ": "runs/stageA_q3vl8b_r16/legacy/scores_valQ.jsonl",
    "valT": "runs/stageA_q3vl8b_r16/legacy/scores_valT.jsonl",
}
DETECTOR = {
    "valQ": "runs/grounding_pilot_stageA_modal/scores_valQ_grounding_pilot32.grounding.detector-v1.jsonl",
    "valT": "runs/grounding_pilot_stageA_modal/scores_valT_grounding_pilot32.grounding.detector-v1.jsonl",
}
SAM2 = {
    "valQ": "runs/segmentation_pilot_stageA_modal/scores_valQ_segmentation_pilot32.segmentation.sam2-box-overlay-v1.jsonl",
    "valT": "runs/segmentation_pilot_stageA_modal/scores_valT_segmentation_pilot32.segmentation.sam2-box-overlay-v1.jsonl",
}
PRIOR_SAM3 = {
    "valQ": f"{PRIOR_RUN}/scores_valQ_sam3.segmentation.sam3-direct-overlay-v1.jsonl",
    "valT": f"{PRIOR_RUN}/scores_valT_sam3.segmentation.sam3-direct-overlay-v1.jsonl",
}

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
        "pillow",
        "pyyaml",
        "tqdm",
    )
    .env({"WORK": WORK, "HF_HOME": f"{WORK}/hf_cache"})
    .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
    .add_local_dir(str(ROOT / "data" / "splits"), f"{WORK}/data/splits", copy=True)
    .add_local_file(str(ROOT / "data" / "manifest.json"), f"{WORK}/data/manifest.json", copy=True)
    .add_local_file(str(ROOT / "data" / "dataset_revision.json"), f"{WORK}/data/dataset_revision.json", copy=True)
    .add_local_dir(str(ROOT / "runs" / "stageA_q3vl8b_r16" / "final_adapter"), f"{WORK}/{ADAPTER}", copy=True)
    .add_local_dir(str(ROOT / RUN / "overlay_cache"), f"{WORK}/{RUN}/overlay_cache", copy=True)
    .add_local_file(str(ROOT / SUBSET), f"{WORK}/{SUBSET}", copy=True)
    .add_local_file(str(ROOT / API_MANIFEST), f"{WORK}/{API_MANIFEST}", copy=True)
)
for reference in (*BASELINE.values(), *DETECTOR.values(), *SAM2.values()):
    image = image.add_local_file(str(ROOT / reference), f"{WORK}/{reference}", copy=True)
for reference in PRIOR_SAM3.values():
    image = image.add_local_file(str(ROOT / reference), f"{WORK}/{reference}", copy=True)

app = modal.App("reva-sam3-score-pilot", image=image)


@app.function(gpu="A100", cpu=8, memory=65536, timeout=12 * 60 * 60)
def sam3_score_pilot() -> dict[str, bytes]:
    work = Path(WORK)
    logs: list[str] = []

    def run(*args: str) -> str:
        command = " ".join(shlex.quote(arg) for arg in args)
        print(f"$ {command}", flush=True)
        result = subprocess.run(
            [sys.executable, *args],
            cwd=WORK,
            env=os.environ.copy(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        logs.append(f"$ {command}\n{result.stdout}")
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {command}")
        return result.stdout

    subset = json.loads((work / SUBSET).read_text())
    api_manifest = json.loads((work / API_MANIFEST).read_text())
    split_rows: dict[str, list[dict]] = {}
    selected_rows: list[dict] = []
    for split in SPLITS:
        rows = json.loads((work / "data" / "splits" / f"{split}.json").read_text())["QA"]
        split_rows[split] = rows
        ids = subset["splits"][split]["ids"]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"subset contains duplicate IDs in {split}")
        by_id = {row["qa_id"]: row for row in rows}
        if set(ids) - set(by_id):
            raise RuntimeError(f"subset IDs missing from {split}")
        selected_rows.extend(by_id[qa_id] for qa_id in ids)
    from huggingface_hub import hf_hub_download, snapshot_download

    dataset_root = work / "data" / "ReVA"
    dataset_root.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="ReVA-Benchmark/ReVA",
        repo_type="dataset",
        filename="train.json",
        revision="06224fa1b42e5e35677a8b7f2145e06307a9a4ee",
        local_dir=str(dataset_root),
    )
    train_rows = json.loads((dataset_root / "train.json").read_text())["QA"]
    train_by_id = {row["qa_id"]: row for row in train_rows}
    for row in selected_rows:
        if train_by_id.get(row["qa_id"]) != row:
            raise RuntimeError(f"selected split row differs from dataset train row: {row['qa_id']}")
    selected_paths = sorted({row["video_path"] for row in selected_rows})
    snapshot_download(
        repo_id="ReVA-Benchmark/ReVA",
        repo_type="dataset",
        revision="06224fa1b42e5e35677a8b7f2145e06307a9a4ee",
        local_dir=str(dataset_root),
        allow_patterns=selected_paths,
        max_workers=8,
    )
    missing = [path for path in selected_paths if not (dataset_root / path).is_file()]
    if missing:
        raise RuntimeError(f"selected source videos are missing: {missing[:3]}")

    code_dir = str(work / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import replicate_sam3 as R

    expected_id_count, expected_call_count = R.validate_sam3_manifest(api_manifest, subset)
    if len(selected_rows) != expected_id_count or len({row["video_path"] for row in selected_rows}) != expected_id_count:
        raise RuntimeError("SAM 3 score subset must contain one unique video per selected ID")
    import grounding as G

    detector_dir = work / "hf_cache" / "grounding-dino-tiny-a2bb814"
    snapshot_download(
        repo_id=G.GROUNDING_DETECTOR_MODEL_ID,
        revision=G.GROUNDING_DETECTOR_MODEL_REVISION,
        local_dir=str(detector_dir),
        allow_patterns=["*.json", "vocab.txt", "model.safetensors"],
    )
    (detector_dir / G.GROUNDING_CACHE_MARKER).write_text(f"{G.GROUNDING_DETECTOR_MODEL_REVISION}\n")
    grounding_cache = f"{RUN}/grounding_cache"
    for split in SPLITS:
        run(
            "code/build_grounding_cache.py",
            "--split", split,
            "--model-dir", str(detector_dir),
            "--grounding-cache-dir", grounding_cache,
            "--qa-ids-file", SUBSET,
            "--offsets", "0,0.5",
            "--frame-sampling", "legacy",
            "--device", "cuda",
            "--batch-size", "4",
        )

    score_paths: dict[str, Path] = {}
    outputs: dict[str, bytes] = {}
    for split in SPLITS:
        score_stem = f"{RUN}/scores_{split}_sam3.jsonl"
        run(
            "code/score.py",
            "--adapter", ADAPTER,
            "--split", split,
            "--grounding-context",
            "--grounding-cache-dir", grounding_cache,
            "--sam3-overlay-dir", f"{RUN}/overlay_cache",
            "--qa-ids-file", SUBSET,
            "--offsets", "0,0.5",
            "--rots", "2",
            "--frame-sampling", "legacy",
            "--out", score_stem,
        )
        path = work / f"{RUN}/scores_{split}_sam3.segmentation.sam3-direct-overlay-v1.jsonl"
        manifest = Path(str(path) + ".manifest.json")
        if not path.is_file() or not manifest.is_file():
            raise RuntimeError(f"missing score output: {path}")
        score_paths[split] = path
        outputs[path.name] = path.read_bytes()
        outputs[manifest.name] = manifest.read_bytes()

    def read_scores(path: str | Path) -> dict[str, dict]:
        return {json.loads(line)["qa_id"]: json.loads(line) for line in Path(work / path).read_text().splitlines()}

    base_arms = {
        "baseline": {split: read_scores(path) for split, path in BASELINE.items()},
        "detector": {split: read_scores(path) for split, path in DETECTOR.items()},
        "sam2": {split: read_scores(path) for split, path in SAM2.items()},
    }
    new_sam3 = {split: read_scores(path) for split, path in score_paths.items()}
    prior_sam3 = {split: read_scores(path) for split, path in PRIOR_SAM3.items()}
    subset_ids = {split: list(subset["splits"][split]["ids"]) for split in SPLITS}
    prior_ids = {split: set(prior_sam3[split]) for split in SPLITS}
    new_ids = {split: set(new_sam3[split]) for split in SPLITS}
    if any(prior_ids[split] & new_ids[split] for split in SPLITS):
        raise RuntimeError("prior and supplemental SAM 3 score IDs overlap")
    if any(new_ids[split] != set(subset_ids[split]) for split in SPLITS):
        raise RuntimeError("supplemental SAM 3 score IDs do not match the selected subset")
    combined_ids = {
        split: sorted(prior_ids[split] | new_ids[split])
        for split in SPLITS
    }
    for split in SPLITS:
        for arm in ("detector", "sam2"):
            if set(base_arms[arm][split]) != set(combined_ids[split]):
                raise RuntimeError(f"{arm} scores do not cover the combined SAM 3 IDs for {split}")
    merged_sam3 = {
        split: {**prior_sam3[split], **new_sam3[split]}
        for split in SPLITS
    }

    def compare(ids_by_split: dict[str, list[str]], sam3_scores: dict[str, dict]) -> tuple[dict, dict]:
        arms = {**base_arms, "sam3": sam3_scores}
        rows_by_arm: dict[str, list[dict]] = {}
        for arm, by_split in arms.items():
            rows = []
            for split in SPLITS:
                truth = {row["qa_id"]: row["correct_answer"] for row in split_rows[split]}
                tasks = {row["qa_id"]: row["task"] for row in split_rows[split]}
                for qa_id in ids_by_split[split]:
                    rows.append({"qa_id": qa_id, "split": split, "task": tasks[qa_id], "correct": by_split[split][qa_id]["pred"] == truth[qa_id]})
            rows_by_arm[arm] = rows
        summaries = {}
        for arm, rows in rows_by_arm.items():
            summaries[arm] = {"correct": sum(row["correct"] for row in rows), "total": len(rows), "accuracy": sum(row["correct"] for row in rows) / len(rows), "by_task": {}}
            for task in sorted({row["task"] for row in rows}):
                group = [row for row in rows if row["task"] == task]
                summaries[arm]["by_task"][task] = {"correct": sum(row["correct"] for row in group), "total": len(group)}
        predictions = {
            arm: {
                qa_id: arms[arm][split][qa_id]["pred"]
                for split in SPLITS
                for qa_id in ids_by_split[split]
            }
            for arm in arms
        }
        truth = {row["qa_id"]: row["correct_answer"] for split in SPLITS for row in split_rows[split] if row["qa_id"] in predictions["sam3"]}
        pairwise = {}
        for left, right in (("baseline", "sam3"), ("detector", "sam3"), ("sam2", "sam3")):
            changed = [qa_id for qa_id in predictions[left] if predictions[left][qa_id] != predictions[right][qa_id]]
            gained = [qa_id for qa_id in changed if predictions[left][qa_id] != truth[qa_id] and predictions[right][qa_id] == truth[qa_id]]
            lost = [qa_id for qa_id in changed if predictions[left][qa_id] == truth[qa_id] and predictions[right][qa_id] != truth[qa_id]]
            pairwise[f"{left}_to_{right}"] = {"changed": len(changed), "gained": len(gained), "lost": len(lost), "question_ids": changed}
        return summaries, pairwise

    def write_comparison(
        name: str,
        title: str,
        ids_by_split: dict[str, list[str]],
        sam3_scores: dict[str, dict],
    ) -> tuple[Path, Path]:
        summaries, pairwise = compare(ids_by_split, sam3_scores)
        comparison = work / f"{RUN}/{name}.json"
        payload = {
            "scope": title,
            "comparison_question_count": sum(len(ids) for ids in ids_by_split.values()),
            "ids": ids_by_split,
            "api": {
                "model": api_manifest.get("model"),
                "version": api_manifest.get("version"),
                "call_count": api_manifest.get("call_count"),
                "prediction_attempt_count": api_manifest.get("prediction_attempt_count"),
                "approx_cost_usd": api_manifest.get("approx_cost_usd"),
                "expected_id_count": expected_id_count,
                "expected_call_count": expected_call_count,
            },
            "arms": summaries,
            "pairwise_prediction_changes": pairwise,
        }
        payload["subset" if title == "supplemental-17" else "supplemental_subset"] = subset
        comparison.write_text(json.dumps(payload, indent=2) + "\n")
        report_lines = [f"# SAM 3 direct-text overlay comparison ({title})", "", "| Arm | Correct | Accuracy |", "|---|---:|---:|"]
        for arm in ("baseline", "detector", "sam2", "sam3"):
            value = summaries[arm]
            report_lines.append(f"| {arm} | {value['correct']}/{value['total']} | {value['accuracy']:.1%} |")
        report_lines += ["", "| Task | n | Baseline | Detector | SAM 2 | SAM 3 |", "|---|---:|---:|---:|---:|---:|"]
        for task in sorted(summaries["sam3"]["by_task"]):
            total = summaries["sam3"]["by_task"][task]["total"]
            cells = [f"{summaries[arm]['by_task'][task]['correct']}/{summaries[arm]['by_task'][task]['total']}" for arm in ("baseline", "detector", "sam2", "sam3")]
            report_lines.append(f"| {task} | {total} | {' | '.join(cells)} |")
        report_lines += ["", f"Supplemental SAM 3 API outputs: {api_manifest.get('call_count')}; prediction attempts: {api_manifest.get('prediction_attempt_count', api_manifest.get('call_count'))} (approx. ${api_manifest.get('approx_cost_usd'):.3f}). Comparison questions: {sum(len(ids) for ids in ids_by_split.values())}.", "", "| Pair | Prediction changes |", "|---|---:|"]
        for pair, value in pairwise.items():
            report_lines.append(f"| {pair} | {value['changed']} (gained {value['gained']}, lost {value['lost']}) |")
        report_lines += ["", "SAM 3 uses all direct prompt masks for each question and the same detector context, Stage A adapter, legacy sampling, offsets, and rotations. The SAM 3 arm never loads the SAM 2 cache or filters direct masks through DINO entities.", ""]
        report = work / f"{RUN}/{name.replace('summary', 'report')}.md"
        report.write_text("\n".join(report_lines))
        outputs[comparison.name] = comparison.read_bytes()
        outputs[report.name] = report.read_bytes()
        return comparison, report

    write_comparison("comparison_summary", "supplemental-17", subset_ids, new_sam3)
    write_comparison("comparison_summary_32", "combined-32", combined_ids, merged_sam3)
    outputs["sam3_score_pilot.log"] = "\n".join(logs).encode()
    return outputs


@app.local_entrypoint()
def main() -> None:
    returned = sam3_score_pilot.remote()
    destination = ROOT / RUN
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in returned.items():
        path = destination / name
        path.write_bytes(contents)
        print(f"saved {path} ({len(contents):,} bytes)")
