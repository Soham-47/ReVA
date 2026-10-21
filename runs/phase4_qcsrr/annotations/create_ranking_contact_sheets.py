from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/phase4_qcsrr"
RESULTS = RUN / "results_resumed-004"
OUT = RUN / "annotations/ranking_contact_sheets"


def digest(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=np.uint8)
    return hashlib.sha256(
        str(tuple(values.shape)).encode("ascii") + np.packbits(values, axis=None).tobytes()
    ).hexdigest()


def main() -> None:
    sys.path.insert(0, str(ROOT / "code"))
    import replicate_sam3

    selection = json.loads((ROOT / "tasks/qcsrr_final_benchmark_selection.json").read_text())
    source = json.loads((RUN / "source_manifest.json").read_text())
    manifest = json.loads((RESULTS / "replicate_manifest.json").read_text())
    candidates = json.loads(gzip.open(RESULTS / "candidates.json.gz", "rt").read())
    rankings = json.loads(gzip.open(RESULTS / "rankings.json.gz", "rt").read())
    by_id = {row["candidate_id"]: row for row in candidates}
    wanted = {(row["qa_id"], row["arm"], row["source_frame_index"], row["mask_sha256"])
              for row in candidates}
    alias_arms: dict[tuple[str, int], set[str]] = {}
    for candidate in candidates:
        for provenance in candidate["prompt_provenance"]:
            alias_arms.setdefault((candidate["qa_id"], int(provenance["call_index"])), set()).add(candidate["arm"])
    masks: dict[tuple[str, str, int, str], np.ndarray] = {}

    with tarfile.open(RESULTS / "replicate_outputs.tar.gz", "r:gz") as archive:
        for record in manifest["records"]:
            member = archive.extractfile(record["output_name"])
            if member is None:
                continue
            entries, _ = replicate_sam3._read_mask_archive(member.read())
            qid, arm = record["qa_id"], record["arm"]
            output_arms = {arm}
            for duplicate_index in record.get("duplicate_call_indices", []):
                output_arms.update(alias_arms.get((qid, int(duplicate_index)), set()))
            indices = source[qid]["source_frame_indices"]
            for entry in entries:
                packed = int(entry["frame_index"])
                source_index = int(indices[packed])
                with Image.open(io.BytesIO(entry["payload"])) as image:
                    union = np.asarray(image.convert("L")) > 0
                variants = [union]
                labels, count = ndimage.label(union, structure=np.ones((3, 3), dtype=np.uint8))
                variants.extend(labels == label for label in range(1, int(count) + 1))
                for mask in variants:
                    for output_arm in output_arms:
                        key = (qid, output_arm, source_index, digest(mask))
                        if key in wanted:
                            masks[key] = mask

    missing = wanted - set(masks)
    assert not missing, f"could not reconstruct {len(missing)} candidate masks"

    grouped: dict[str, dict[str, dict]] = {}
    for row in rankings:
        grouped.setdefault(row["qa_id"], {})[(row["arm"] + ":" + row["phrase_role"])] = row

    font = ImageFont.load_default()
    card_w, card_h, image_h = 220, 204, 150
    OUT.mkdir(parents=True, exist_ok=True)
    for question in selection["questions"]:
        qid = question["qa_id"]
        rows = [
            ("R0 question-conditioned", grouped[qid].get("question-conditioned:target"), "R0"),
            ("R1 spatial/relational", grouped[qid].get("question-conditioned:target"), "R1"),
            ("R2 temporal association", grouped[qid].get("question-conditioned:target"), "R2"),
            ("Fixed-vocabulary control", grouped[qid].get("fixed-vocabulary:fixed_control"), "R0"),
        ]
        sheet = Image.new("RGB", (card_w * 5, 78 + card_h * len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        title = f"{qid} — {question['question']}"
        draw.multiline_text((8, 6), "\n".join(textwrap.wrap(title, width=135)), fill="black", font=font, spacing=2)
        y0 = 78
        for row_index, (label, record, ranking_key) in enumerate(rows):
            y = y0 + row_index * card_h
            draw.text((5, y + 2), label, fill="black", font=font)
            if record is None:
                draw.text((8, y + 32), "No ranking row", fill="gray", font=font)
                continue
            predictions = record[ranking_key]["top5"][:5]
            for rank in range(5):
                x = rank * card_w
                draw.rectangle((x, y + 20, x + card_w - 3, y + card_h - 3), outline="#888", width=1)
                if rank >= len(predictions):
                    draw.text((x + 6, y + 28), "No candidate", fill="#666", font=font)
                    continue
                pred = predictions[rank]
                cid = pred["candidate_id"]
                candidate = by_id[cid]
                key = (qid, candidate["arm"], candidate["source_frame_index"], candidate["mask_sha256"])
                mask = masks[key]
                meta = source[qid]
                frame_path = RUN / "source_frames" / qid / f"frame-{candidate['source_frame_index']:06d}.jpg"
                frame = Image.open(frame_path).convert("RGB")
                box = candidate["crop_box_pixels"]
                crop = frame.crop((box["left"], box["top"], box["right"], box["bottom"]))
                mask_crop = Image.fromarray((mask * 255).astype(np.uint8)).crop(
                    (box["left"], box["top"], box["right"], box["bottom"])
                )
                overlay = Image.new("RGBA", crop.size, (255, 35, 30, 0))
                overlay.putalpha(mask_crop.point(lambda value: int(value * 0.38)))
                crop = Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")
                crop.thumbnail((card_w - 10, image_h - 4), Image.Resampling.LANCZOS)
                px = x + (card_w - crop.width) // 2
                py = y + 21 + (image_h - crop.height) // 2
                sheet.paste(crop, (px, py))
                sim = pred.get("similarity", candidate.get("similarity"))
                track = f" track={pred.get('track_length')}" if ranking_key == "R2" else ""
                detail = f"#{rank + 1} sim={sim:.4f} f={candidate['source_frame_index']} t={candidate['timestamp']:.2f}s{track}"
                draw.text((x + 5, y + 21 + image_h), detail, fill="black", font=font)
                draw.text((x + 5, y + 21 + image_h + 13), f"{candidate['proposal_kinds']} | {cid[-10:]}", fill="#444", font=font)
        sheet.save(OUT / f"{qid}.jpg", quality=90)

    assert len(list(OUT.glob("train_*.jpg"))) == 12
    print(json.dumps({"status": "complete", "question_sheets": 12, "mask_candidates": len(masks), "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
