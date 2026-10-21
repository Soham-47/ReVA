"""Render rank-free Gate 1 candidate sheets for human relevance labeling."""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "annotations" / "blinded_review"
SEED = 20261018
COLS, ROWS = 8, 8
CARD_W, CARD_H = 164, 110


def load_mask(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(member))) as image:
        return np.asarray(image.convert("L")) > 0


def make_card(candidate: dict, image: Image.Image, mask: np.ndarray, alias: str) -> Image.Image:
    card = Image.new("RGB", (CARD_W, CARD_H), "white")
    draw = ImageDraw.Draw(card)
    draw.text((3, 2), f"{alias}  |  {candidate['timestamp_s']:.2f}s", fill="black")

    source = image.copy().convert("RGB")
    box = candidate["proposal_box_pixels"]
    source_draw = ImageDraw.Draw(source)
    source_draw.rectangle((box["left"], box["top"], box["right"] - 1, box["bottom"] - 1), outline=(255, 30, 30), width=2)
    source.thumbnail((54, 76), Image.Resampling.LANCZOS)
    card.paste(source, (3, 24))

    crop_box = candidate["crop_box_pixels"]
    crop = image.crop((crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])).convert("RGB")
    crop_mask = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").crop(
        (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])
    )
    overlay = Image.new("RGB", crop.size, (0, 220, 80))
    crop = Image.composite(overlay, crop, crop_mask.point(lambda value: int(value * 0.35)))
    crop.thumbnail((101, 76), Image.Resampling.LANCZOS)
    card.paste(crop, (60, 24))
    return card


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = [json.loads(line) for line in (ROOT / "candidates.jsonl").read_text().splitlines()]
    by_video: dict[str, list[dict]] = {}
    for item in candidates:
        by_video.setdefault(item["qa_id"], []).append(item)

    mapping: list[dict] = []
    csv_path = OUT / "annotation_template.csv"
    template_fields = [
        "blind_id", "qa_id", "category_correct", "instance_correct", "region_relevant",
        "target_visible", "mask_complete", "oversegmented", "wrong_instance",
        "cross_frame_identity", "notes",
    ]
    with zipfile.ZipFile(ROOT / "candidate_masks.zip") as archive, csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=template_fields)
        writer.writeheader()
        rng = random.Random(SEED)
        for qa_id, rows in sorted(by_video.items()):
            rng.shuffle(rows)
            frame_cache: dict[str, Image.Image] = {}
            for idx, candidate in enumerate(rows, 1):
                blind_id = f"{qa_id[-2:]}-{idx:04d}"
                mapping.append({"blind_id": blind_id, "candidate_id": candidate["candidate_id"], "qa_id": qa_id})
                writer.writerow({"blind_id": blind_id, "qa_id": qa_id})
                frame_path = ROOT / candidate["source_frame"]
                key = str(frame_path)
                if key not in frame_cache:
                    frame_cache[key] = Image.open(frame_path).convert("RGB")
                mask = load_mask(archive, candidate["mask_member"])
                card = make_card(candidate, frame_cache[key], mask, blind_id)
                candidate["_review_card"] = card

            for page, start in enumerate(range(0, len(rows), COLS * ROWS), 1):
                subset = rows[start : start + COLS * ROWS]
                sheet = Image.new("RGB", (COLS * CARD_W, ROWS * CARD_H), (238, 238, 238))
                for offset, candidate in enumerate(subset):
                    alias = f"{qa_id[-2:]}-{start + offset + 1:04d}"
                    x = (offset % COLS) * CARD_W
                    y = (offset // COLS) * CARD_H
                    sheet.paste(candidate.pop("_review_card"), (x, y))
                sheet.save(OUT / f"{qa_id}_page-{page:02d}.jpg", quality=92)

    (OUT / "blind_id_key.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Rank-free Gate 1 relevance review\n\n"
        "Pages are randomly ordered by a fixed seed and hide proposal arm, proposal prompt, candidate type, score, and rank. Each card shows the source frame with the proposal box and an enlarged source-resolution crop with a translucent mask overlay. The CSV is an annotation template; labels are evaluation-only. The key linking blind IDs to candidate IDs is kept separate from the pages.\n\n"
        "The prior audit exposed some top-ranked examples, so full blinding is impossible. Do not infer or enter labels from rank. Candidate IDs were assigned after shuffling and are not rank-derived.\n"
    )


if __name__ == "__main__":
    main()
