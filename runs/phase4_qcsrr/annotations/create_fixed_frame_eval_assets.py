from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/phase4_qcsrr"
RESULTS = RUN / "results_resumed-004"
OUT = RUN / "annotations/fixed_frame_eval"
SHEETS = OUT / "contact_sheets"
PACKED_FRAME_INDICES = (0, 13)
PAGE_SIZE = 20
CARD_W, CARD_H = 360, 232
PANEL_SIZE = (170, 150)


def mask_digest(mask: np.ndarray) -> str:
    values = np.asarray(mask, dtype=np.uint8)
    return hashlib.sha256(
        str(tuple(values.shape)).encode("ascii") + np.packbits(values, axis=None).tobytes()
    ).hexdigest()


def candidate_alias(candidate_id: str) -> str:
    token = hashlib.sha256(
        f"ReVA-fixed-frame-eval-v1|{candidate_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"FVE-{token}"


def candidate_record_digest(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#f2f2f2")
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def render_card(
    row: dict,
    alias: str,
    mask: np.ndarray,
    frame: Image.Image,
    font: ImageFont.ImageFont,
) -> Image.Image:
    width, height = frame.size
    if tuple(mask.shape) != (height, width):
        raise AssertionError(f"mask/frame shape mismatch for {row['candidate_id']}")
    box = row["crop_box_pixels"]
    left, top, right, bottom = (int(box[key]) for key in ("left", "top", "right", "bottom"))
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise AssertionError(f"invalid crop box for {row['candidate_id']}")
    mask_crop = mask[top:bottom, left:right]
    if not bool(mask_crop.any()):
        raise AssertionError(f"empty crop mask for {row['candidate_id']}")

    context = frame.copy()
    ImageDraw.Draw(context).rectangle((left, top, right - 1, bottom - 1), outline="#e53935", width=3)
    crop = frame.crop((left, top, right, bottom)).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (229, 57, 53, 0))
    overlay.putalpha(Image.fromarray((mask_crop.astype(np.uint8) * 110), mode="L"))
    crop = Image.alpha_composite(crop, overlay).convert("RGB")

    card = Image.new("RGB", (CARD_W, CARD_H), "white")
    draw = ImageDraw.Draw(card)
    draw.text((8, 6), alias, fill="black", font=font)
    draw.text(
        (8, 22),
        f"{row['qa_id']}  frame {row['source_frame_index']}  t={row['timestamp']:.3f}s",
        fill="#444",
        font=font,
    )
    context_panel = fit_panel(context, PANEL_SIZE)
    crop_panel = fit_panel(crop, PANEL_SIZE)
    card.paste(context_panel, (8, 42))
    card.paste(crop_panel, (182, 42))
    draw.rectangle((8, 42, 8 + PANEL_SIZE[0] - 1, 42 + PANEL_SIZE[1] - 1), outline="#999")
    draw.rectangle((182, 42, 182 + PANEL_SIZE[0] - 1, 42 + PANEL_SIZE[1] - 1), outline="#999")
    draw.text((8, 198), "source context", fill="#666", font=font)
    draw.text((182, 198), "mask overlay crop", fill="#666", font=font)
    return card


def reconstruct_masks(source: dict, candidates: list[dict]) -> dict[tuple[str, str, int, str], np.ndarray]:
    import replicate_sam3

    selected = [row for row in candidates if int(row["packed_frame_index"]) in PACKED_FRAME_INDICES]
    wanted = {
        (row["qa_id"], row["arm"], int(row["source_frame_index"]), row["mask_sha256"])
        for row in selected
    }
    alias_arms: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in candidates:
        for provenance in row["prompt_provenance"]:
            alias_arms[(row["qa_id"], int(provenance["call_index"]))].add(row["arm"])

    masks: dict[tuple[str, str, int, str], np.ndarray] = {}
    manifest = json.loads((RESULTS / "replicate_manifest.json").read_text())
    with tarfile.open(RESULTS / "replicate_outputs.tar.gz", "r:gz") as archive:
        for record in manifest["records"]:
            member = archive.extractfile(record["output_name"])
            if member is None:
                raise AssertionError(f"missing archive member {record['output_name']}")
            entries, _ = replicate_sam3._read_mask_archive(member.read())
            qid, arm = record["qa_id"], record["arm"]
            output_arms = {arm}
            for duplicate_index in record.get("duplicate_call_indices", []):
                output_arms.update(alias_arms.get((qid, int(duplicate_index)), set()))
            source_indices = source[qid]["source_frame_indices"]
            for entry in entries:
                packed_index = int(entry["frame_index"])
                if packed_index not in PACKED_FRAME_INDICES:
                    continue
                source_index = int(source_indices[packed_index])
                with Image.open(io.BytesIO(entry["payload"])) as image:
                    union = np.asarray(image.convert("L")) > 0
                variants = [union]
                labels, component_count = ndimage.label(
                    union, structure=np.ones((3, 3), dtype=np.uint8)
                )
                variants.extend(labels == label for label in range(1, int(component_count) + 1))
                for mask in variants:
                    digest = mask_digest(mask)
                    for output_arm in output_arms:
                        key = (qid, output_arm, source_index, digest)
                        if key not in wanted:
                            continue
                        if key in masks:
                            assert np.array_equal(masks[key], mask), f"conflicting reconstructed mask for {key}"
                        else:
                            masks[key] = mask
    missing = wanted - set(masks)
    assert not missing, f"could not reconstruct {len(missing)} selected candidate masks"
    return masks


def main() -> None:
    sys.path.insert(0, str(ROOT / "code"))
    source = json.loads((RUN / "source_manifest.json").read_text())
    candidates = json.loads(gzip.open(RESULTS / "candidates.json.gz", "rt").read())
    selected = [row for row in candidates if int(row["packed_frame_index"]) in PACKED_FRAME_INDICES]
    assert len(selected) == 470, len(selected)
    assert {row["qa_id"] for row in selected} == set(source)
    assert len({row["candidate_id"] for row in selected}) == len(selected)

    aliases = {row["candidate_id"]: candidate_alias(row["candidate_id"]) for row in selected}
    assert len(set(aliases.values())) == len(aliases)
    masks = reconstruct_masks(source, candidates)

    rows = sorted(
        selected,
        key=lambda row: (row["qa_id"], int(row["packed_frame_index"]), aliases[row["candidate_id"]]),
    )
    counts_by_video: dict[str, dict] = {}
    counts_by_arm = Counter(row["arm"] for row in rows)
    counts_by_frame = Counter(int(row["packed_frame_index"]) for row in rows)
    for qid in sorted(source):
        qrows = [row for row in rows if row["qa_id"] == qid]
        counts_by_video[qid] = {
            "total": len(qrows),
            "by_arm": dict(sorted(Counter(row["arm"] for row in qrows).items())),
            "by_packed_frame": dict(sorted(Counter(str(row["packed_frame_index"]) for row in qrows).items())),
            "by_arm_and_packed_frame": {
                arm: dict(sorted(Counter(str(row["packed_frame_index"]) for row in qrows if row["arm"] == arm).items()))
                for arm in sorted({row["arm"] for row in qrows})
            },
            "source_frames": {
                str(packed): {
                    "source_frame_index": int(source[qid]["source_frame_indices"][packed]),
                    "timestamp_s": float(source[qid]["source_timestamps_s"][packed]),
                }
                for packed in PACKED_FRAME_INDICES
            },
        }

    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {
                "alias": aliases[row["candidate_id"]],
                "qa_id": row["qa_id"],
                "arm": row["arm"],
                "packed_frame_index": int(row["packed_frame_index"]),
                "source_frame_index": int(row["source_frame_index"]),
                "timestamp_s": float(row["timestamp"]),
                "candidate_id": row["candidate_id"],
                "candidate_record_sha256": candidate_record_digest(row),
                "mask_sha256": row["mask_sha256"],
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    for old in SHEETS.glob("page-*.jpg"):
        old.unlink()

    alias_map = {
        "schema_version": 1,
        "status": "evaluator_only",
        "alias_policy": "FVE- plus the first 12 hex digits of SHA-256(ReVA-fixed-frame-eval-v1|candidate_id)",
        "aliases": {aliases[row["candidate_id"]]: row["candidate_id"] for row in rows},
    }
    (OUT / "evaluator_only_alias_to_candidate_id.json").write_text(
        json.dumps(alias_map, indent=2, sort_keys=True) + "\n"
    )

    selection_manifest = {
        "schema_version": 1,
        "status": "frozen",
        "evaluator_only": True,
        "source_results": "runs/phase4_qcsrr/results_resumed-004",
        "selection_rule": {
            "candidate_source": "results_resumed-004/candidates.json.gz",
            "packed_frame_indices": list(PACKED_FRAME_INDICES),
            "include": "every retained candidate row from both proposal arms whose packed_frame_index is 0 or 13",
            "rank_independent": True,
            "labels_created": False,
        },
        "counts": {
            "total": len(rows),
            "by_arm": dict(sorted(counts_by_arm.items())),
            "by_packed_frame": dict(sorted((str(k), v) for k, v in counts_by_frame.items())),
            "per_video": counts_by_video,
        },
        "selected_candidates": manifest_rows,
    }
    (OUT / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n"
    )

    font = ImageFont.load_default()
    rendered_aliases: Counter[str] = Counter()
    rendered_masks: Counter[str] = Counter()
    for page_number, start in enumerate(range(0, len(rows), PAGE_SIZE), start=1):
        page_rows = rows[start : start + PAGE_SIZE]
        page = Image.new("RGB", (CARD_W * 4, CARD_H * 5 + 38), "#e9e9e9")
        draw = ImageDraw.Draw(page)
        draw.text(
            (8, 8),
            f"Fixed-frame candidate review  |  page {page_number}/{(len(rows) + PAGE_SIZE - 1) // PAGE_SIZE}",
            fill="black",
            font=font,
        )
        for index, row in enumerate(page_rows):
            alias = aliases[row["candidate_id"]]
            key = (row["qa_id"], row["arm"], int(row["source_frame_index"]), row["mask_sha256"])
            frame_path = RUN / "source_frames" / row["qa_id"] / f"frame-{int(row['source_frame_index']):06d}.jpg"
            if not frame_path.is_file():
                raise AssertionError(f"missing source frame {frame_path}")
            frame = Image.open(frame_path).convert("RGB")
            card = render_card(row, alias, masks[key], frame, font)
            x = (index % 4) * CARD_W
            y = 38 + (index // 4) * CARD_H
            page.paste(card, (x, y))
            rendered_aliases[alias] += 1
            rendered_masks[alias] += 1
        page.save(SHEETS / f"page-{page_number:03d}.jpg", quality=92, optimize=True)

    assert set(rendered_aliases) == set(aliases.values())
    assert all(count == 1 for count in rendered_aliases.values())
    assert rendered_masks == rendered_aliases
    assert sum(counts_by_video[qid]["total"] for qid in counts_by_video) == len(rows)
    assert sum(counts_by_arm.values()) == len(rows)
    assert sum(counts_by_frame.values()) == len(rows)
    assert all(
        row["source_frame_index"] == source[row["qa_id"]]["source_frame_indices"][row["packed_frame_index"]]
        and row["timestamp"] == source[row["qa_id"]]["source_timestamps_s"][row["packed_frame_index"]]
        for row in rows
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_candidates": len(rows),
                "masks_reconstructed": len(masks),
                "pages": len(list(SHEETS.glob("page-*.jpg"))),
                "output_dir": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
