"""Render saved per-frame top-five crops with rank and score labels."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "annotations" / "ranked_top5"
CARD_W, CARD_H = 170, 112


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = {
        row["candidate_id"]: row
        for row in (json.loads(line) for line in (ROOT / "candidates.jsonl").read_text().splitlines())
    }
    rankings = json.loads((ROOT / "rankings.json").read_text())
    by_key = {
        (row["qa_id"], row["phrase"], row["arm"], row["crop_style"], row["packed_frame_index"]): row
        for row in rankings
        if row["rank_scope"] == "per_frame"
    }
    with zipfile.ZipFile(ROOT / "candidate_masks.zip") as archive:
        for qa_id, phrases in (("train_010397", ["black car", "parking lot"]), ("train_009869", ["baseball field"])):
            for phrase in phrases:
                for first_frame in range(0, 16, 2):
                    page = Image.new("RGB", (5 * CARD_W, 8 * CARD_H + 35), "white")
                    draw = ImageDraw.Draw(page)
                    draw.text((5, 5), f"{qa_id} | {phrase} | per-frame top five | frames {first_frame}-{first_frame+1}", fill="black")
                    for frame_offset in range(2):
                        frame_idx = first_frame + frame_offset
                        group_y = 25 + frame_offset * 4 * CARD_H
                        for group_idx, (arm, style) in enumerate(
                            [(a, s) for a in ("question-conditioned", "fixed-vocabulary") for s in ("padded_rgb_box", "masked_rgb")]
                        ):
                            row = by_key.get((qa_id, phrase, arm, style, frame_idx))
                            row_y = group_y + group_idx * CARD_H
                            short_arm = "Q" if arm == "question-conditioned" else "F"
                            short_style = "pad" if style == "padded_rgb_box" else "mask"
                            draw.text((3, row_y + 2), f"f{frame_idx:02d} {short_arm}/{short_style}", fill="black")
                            ranked = [] if row is None else row["siglip_top5"]
                            for rank in range(1, 6):
                                x = (rank - 1) * CARD_W
                                if rank > len(ranked):
                                    draw.text((x + 3, row_y + 24), f"R{rank}: no candidate", fill=(90, 90, 90))
                                    continue
                                candidate = ranked[rank - 1]
                                meta = candidates[candidate["candidate_id"]]
                                score = candidate["similarity"]
                                draw.text((x + 38, row_y + 2), f"R{rank} {score:.4f}", fill="black")
                                frame_path = ROOT / meta["source_frame"]
                                with Image.open(frame_path) as im:
                                    image = im.convert("RGB")
                                    box = meta["proposal_box_pixels"]
                                    ImageDraw.Draw(image).rectangle(
                                        (box["left"], box["top"], box["right"] - 1, box["bottom"] - 1), outline=(255, 0, 0), width=2
                                    )
                                    image.thumbnail((34, 70), Image.Resampling.LANCZOS)
                                    page.paste(image, (x + 2, row_y + 23))
                                    crop_box = meta["crop_box_pixels"]
                                    crop = im.convert("RGB").crop((crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"]))
                                with Image.open(io.BytesIO(archive.read(meta["mask_member"]))) as mask_image:
                                    mask = np.asarray(mask_image.convert("L")) > 0
                                crop_mask = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").crop(
                                    (crop_box["left"], crop_box["top"], crop_box["right"], crop_box["bottom"])
                                )
                                overlay = Image.new("RGB", crop.size, (20, 230, 80))
                                crop = Image.composite(overlay, crop, crop_mask.point(lambda value: int(value * 0.28)))
                                crop.thumbnail((128, 70), Image.Resampling.LANCZOS)
                                page.paste(crop, (x + 38, row_y + 23))
                    page.save(OUT / f"{qa_id}_{phrase.replace(' ', '-')}_frames-{first_frame:02d}-{first_frame+1:02d}.jpg", quality=92)
    (OUT / "README.md").write_text(
        "These sheets visualize only the frozen SigLIP2 per-frame top-five rankings. Candidate rank, cosine similarity, crop style, proposal arm, and timestamp are shown. They are unblinded retrieval examples; rank exposure makes them unsuitable as annotation inputs. The rank-free review sheets are in `../blinded_review/`.\n"
    )


if __name__ == "__main__":
    main()
