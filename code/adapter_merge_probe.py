"""Compare historical merged and exact unmerged adapter inference on smoke rows."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def _move_inputs(encoded: dict, device: torch.device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def _rows() -> list[dict]:
    with C.work_path("data/splits/fit_A.json").open() as stream:
        rows = json.load(stream)["QA"]
    picks = []
    for source in ("VisDrone", "Hawk_UAV", "ERA", "UAVDT"):
        picks.append(next(row for row in rows if row["dataset_name"].startswith(source)))
    manifest = C.load_manifest()
    worst = max(
        rows,
        key=lambda row: (
            manifest[row["video_path"]]["width"] * manifest[row["video_path"]]["height"],
            manifest[row["video_path"]]["n_frames"],
        ),
    )
    if worst["qa_id"] not in {row["qa_id"] for row in picks}:
        picks.append(worst)
    return picks


def _score_model(
    model,
    encoded_rows: list[tuple[dict, dict, list[str]]],
    ltr_ids: list[int],
    mode: str,
) -> list[dict]:
    device = torch.device("cuda")
    output = []
    with torch.inference_mode():
        for row, encoded, permutation in encoded_rows:
            shown_logits = model(**_move_inputs(encoded, device)).logits[0, -1, ltr_ids].float().cpu()
            shown_probabilities = torch.softmax(shown_logits, dim=-1)
            gold_shown_index = permutation.index(row["correct_answer"])
            wrong = torch.cat(
                (shown_probabilities[:gold_shown_index], shown_probabilities[gold_shown_index + 1 :])
            )
            original_logits = torch.zeros_like(shown_logits)
            original_probabilities = torch.zeros_like(shown_probabilities)
            for shown_index, original_letter in enumerate(permutation):
                original_index = C.LETTERS.index(original_letter)
                original_logits[original_index] = shown_logits[shown_index]
                original_probabilities[original_index] = shown_probabilities[shown_index]
            output.append(
                {
                    "arm": None,
                    "mode": mode,
                    "qa_id": row["qa_id"],
                    "dataset_name": row["dataset_name"],
                    "gold": row["correct_answer"],
                    "permutation": permutation,
                    "logits": [float(value) for value in original_logits],
                    "probabilities": [float(value) for value in original_probabilities],
                    "shown_logits": [float(value) for value in shown_logits],
                    "shown_probabilities": [float(value) for value in shown_probabilities],
                    "prediction": permutation[int(torch.argmax(shown_logits))],
                    "gold_margin": float(shown_probabilities[gold_shown_index] - wrong.max()),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="adapter arm name and runtime adapter directory; repeat for A0/A1/A2",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("adapter merge probe requires CUDA")

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            raise ValueError(f"--arm must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        arms[name] = Path(path)
    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    encoded_rows: list[tuple[dict, dict, list[str]]] = []
    for row in _rows():
        frames, meta = C.get_video(row["video_path"], 0.0)
        encoded, permutation = C.encode(processor, row, frames, meta, ["B", "D", "A", "C"])
        encoded_rows.append((row, encoded, permutation))

    records = []
    for arm, adapter_path in arms.items():
        if not adapter_path.is_dir():
            raise FileNotFoundError(adapter_path)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            C.MODEL_ID,
            revision=C.MODEL_REV,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        loaded = C.load_peft_for_inference(base, adapter_path, merge=False).cuda().eval()
        adapter_names = list(loaded.peft_config)
        unmerged = _score_model(loaded, encoded_rows, ltr_ids, "unmerged")
        for record in unmerged:
            record["arm"] = arm
            record["adapter_names"] = adapter_names
        merged = loaded.merge_and_unload(adapter_names=adapter_names).cuda().eval()
        merged_rows = _score_model(merged, encoded_rows, ltr_ids, "merged")
        for record in merged_rows:
            record["arm"] = arm
            record["adapter_names"] = adapter_names
        records.extend(unmerged)
        records.extend(merged_rows)
        unmerged_by_id = {record["qa_id"]: record for record in unmerged}
        merged_by_id = {record["qa_id"]: record for record in merged_rows}
        print(
            f"arm={arm} adapters={adapter_names} "
            f"peak_gb={torch.cuda.max_memory_allocated() / 2**30:.1f}"
        )
        for qa_id in unmerged_by_id:
            left = unmerged_by_id[qa_id]
            right = merged_by_id[qa_id]
            max_answer_diff = max(
                abs(a - b) for a, b in zip(left["logits"], right["logits"])
            )
            print(
                f"arm={arm} qa_id={qa_id} "
                f"unmerged_pred={left['prediction']} merged_pred={right['prediction']} "
                f"unmerged_margin={left['gold_margin']:.6f} merged_margin={right['gold_margin']:.6f} "
                f"answer_logit_max_diff={max_answer_diff:.6f}"
            )
        del merged, loaded, base
        gc.collect()
        torch.cuda.empty_cache()

    output = C.work_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
