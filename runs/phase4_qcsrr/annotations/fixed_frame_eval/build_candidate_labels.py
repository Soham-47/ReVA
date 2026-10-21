"""Build the blinded, source-only labels for the fixed-frame review packet.

This script intentionally reads only the frozen selection manifest, question
wording, source-review files, and anonymous contact-sheet metadata.  It does
not read rankings, scores, result archives, or candidate provenance.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/phase4_qcsrr"
OUT = RUN / "annotations/fixed_frame_eval"
MANIFEST = OUT / "selection_manifest.json"
QUESTIONS = ROOT / "tasks/qcsrr_final_benchmark_selection.json"
PRIMARY = RUN / "annotations/source_review_primary.json"
ADJUDICATION = RUN / "annotations/source_review_adjudication.json"
SECOND = RUN / "annotations/source_review_second.json"
OUTPUT = OUT / "candidate_labels.json"

CAT = {"positive", "negative", "ambiguous", "not_assessable"}
INST = CAT
ROLES = {"target", "reference", "context"}
VIS = {"visible", "absent", "ambiguous", "uncertain"}
QUALITY = {"useful", "too_broad", "irrelevant", "ambiguous"}
COMPLETE = {"complete", "partial", "fragment", "overbroad", "not_applicable"}


def load(path: Path):
    return json.loads(path.read_text())


def question_map() -> dict[str, dict]:
    return {row["qa_id"]: row for row in load(QUESTIONS)["questions"]}


def source_map() -> dict[str, dict]:
    return load(PRIMARY)["questions"]


def adjudication_map() -> dict[str, dict]:
    return load(ADJUDICATION).get("source_relations", {})


def page_for(index: int) -> str:
    return f"contact_sheets/page-{index // 20 + 1:03d}.jpg"


def target_visibility_for(qa: str, packed_frame_index: int) -> str:
    """Visibility of the singular question target, independent of this crop.

    This is intentionally separate from category_visibility: a scene can show
    a category while leaving the named instance unresolved (for example, the
    multiple baseball diamonds in train_010200).
    """
    if qa == "train_001478":
        return "absent" if packed_frame_index == 0 else "visible"
    if qa == "train_008163":
        return "visible" if packed_frame_index == 0 else "uncertain"
    if qa == "train_008503":
        # The primary review says visible through the clip, while the second
        # review marks later selected samples as not visible.  Preserve that
        # source-review disagreement at the later frozen sample.
        return "visible" if packed_frame_index == 0 else "uncertain"
    if qa == "train_010200":
        # The baseball-field category is visible in the source reviews, but
        # no single diamond is uniquely the question target.
        return "ambiguous"
    if qa == "train_012936":
        return "ambiguous"
    return "visible"


def category_visibility_for(qa: str, packed_frame_index: int) -> str:
    """Visibility of the requested category at the source timestamp."""
    if qa == "train_001478":
        return "absent" if packed_frame_index == 0 else "visible"
    if qa == "train_008163":
        return "visible" if packed_frame_index == 0 else "uncertain"
    if qa == "train_008503":
        # Primary review says visible; second review disagrees after clip
        # start, so the packed-13 result is not collapsed to visible/absent.
        return "visible" if packed_frame_index == 0 else "uncertain"
    if qa == "train_010200":
        # Both source reviews describe the baseball-field category as visible
        # across samples, even though singular instance identity is ambiguous.
        return "visible"
    if qa == "train_012936":
        return "ambiguous"
    return "visible"


def make_row(row: dict, index: int, questions: dict[str, dict], sources: dict[str, dict], relations: dict[str, dict]) -> dict:
    qa = row["qa_id"]
    question = questions[qa]["question"]
    source_note = sources[qa].get("source_only_notes", "")
    relation = relations.get(qa, {})
    if qa == "train_001478":
        source_note = (
            "Adjudicated source-only correction: the preliminary bright-green-bus note was erroneous; "
            "the white/red bus is absent at packed frames 0-11, ambiguous at packed frame 12, "
            "visible from packed frame 13, and remains visible at the final sample."
        )
    result = {
        "alias": row["alias"],
        "qa_id": qa,
        "question": question,
        "packed_frame_index": int(row["packed_frame_index"]),
        "source_frame_index": int(row["source_frame_index"]),
        "timestamp_s": float(row["timestamp_s"]),
        "category_relevance": "ambiguous",
        "correct_instance_relevance": "ambiguous",
        "relevant_roles": [],
        "visibility": "visible",  # legacy alias; replaced with target visibility below
        "target_visibility": "ambiguous",
        "category_visibility": "ambiguous",
        "crop_visibility": "visible",
        "contextual_crop_quality": "useful",
        "mask_completeness": "complete",
        "source_only_reason": "",
        "source_evidence": {
            "source_frame": f"source_frames/{qa}/frame-{int(row['source_frame_index']):06d}.jpg",
            "anonymous_contact_sheet": page_for(index),
            "source_review_note": source_note,
            "adjudicated_relation": relation.get("observed_relation"),
        },
        "certainty": "medium",
    }

    def set_label(category, instance, roles, reason, *, certainty="medium", visibility="visible", quality="useful", completeness="complete"):
        result.update(
            category_relevance=category,
            correct_instance_relevance=instance,
            relevant_roles=sorted(set(roles)),
            crop_visibility=visibility,
            contextual_crop_quality=quality,
            mask_completeness=completeness,
            source_only_reason=reason,
            certainty=certainty,
        )

    if qa == "train_001478":
        if index in {6, 8, 10, 11, 14}:
            set_label("positive", "positive", ["target"], "The crop contains the white/red bus visible in this source frame.", certainty="high")
        elif index in {9, 12}:
            set_label("ambiguous", "ambiguous", ["target"], "A tiny red/white road-user crop may be the question-referenced bus, but source resolution does not make the identity secure.", certainty="low", quality="ambiguous", completeness="fragment")
        elif index == 13:
            set_label("positive", "ambiguous", ["target"], "The broad crop includes the question-referenced bus together with another road user; the bus is visible but the mask is not isolated.", certainty="medium", quality="too_broad", completeness="overbroad")
        elif index in {2, 4, 5}:
            set_label("ambiguous", "ambiguous", [], "The clip-start frame has no defensible white/red bus identity; this small road-user crop cannot be matched uniquely from source pixels.", certainty="low", quality="ambiguous", completeness="fragment")
        elif index == 1:
            set_label("negative", "negative", [], "The crop is a roadway and dark car, while the source review places the white/red bus later in the clip.", certainty="high", quality="too_broad", completeness="overbroad")
        else:
            set_label("negative", "negative", [], "The crop is a non-bus road user; the white/red bus is not present in this clip-start source frame.", certainty="high")

    elif qa == "train_002092":
        # Source pixels distinguish planted vegetation from the central median
        # only when the crop is large enough.  Side trees match vegetation but
        # are wrong for the requested median instance.
        side = {15, 17, 19, 22, 25, 26, 30, 36, 42}
        central = {28, 31, 37, 39, 43, 47, 49}
        context = {33, 35, 40, 41, 45, 46, 48, 50}
        tiny = {16, 18, 20, 21, 23, 24, 27, 29, 32, 34, 38, 44}
        if index in side:
            set_label("positive", "negative", ["target"], "Vegetation is visible, but the crop is on a roadside verge or tree row rather than the central median named by the question.", certainty="high")
        elif index in central:
            set_label("positive", "positive", ["target"], "The crop covers planted vegetation in the road median; the source frame supports the requested land-cover instance.", certainty="medium")
        elif index in context:
            set_label("positive", "negative", ["context", "reference"], "The crop covers road or median context useful for the question, but it is not the requested land-cover region.", certainty="medium", quality="too_broad" if index in {33, 35, 40, 41, 48, 50} else "useful", completeness="overbroad" if index in {33, 35, 40, 41, 48, 50} else "complete")
        elif index in tiny:
            set_label("ambiguous", "ambiguous", ["target"], "The tiny crop shows a vegetation-like or road-side fragment, but source pixels do not establish central-median identity.", certainty="low", quality="ambiguous", completeness="fragment")
        else:
            set_label("ambiguous", "ambiguous", ["target"], "The crop is too small to determine whether the vegetation belongs to the central median.", certainty="low", quality="ambiguous", completeness="fragment")

    elif qa == "train_003224":
        set_label("positive", "ambiguous", ["target"], "The crop is a covered or fenced structure on the baseball-field side identified by the source review; fine-grained structure class remains unresolved.", certainty="medium")

    elif qa == "train_005918":
        target = {55, 63, 66, 73, 75, 77, 84, 89, 95, 103, 109, 115, 123, 126, 140, 141, 143}
        road_context = {58, 59, 60, 62, 64, 69, 70, 76, 79, 85, 87, 88, 90, 91, 93, 94, 97, 98, 99, 100, 101, 103, 104, 105, 110, 111, 112, 113, 114, 117, 118, 119, 121, 125, 128, 130, 131, 133, 134, 136, 137, 144, 145}
        vegetation = {53, 54, 57, 67, 71, 74, 78, 82, 86, 92, 96, 102, 106, 108, 116, 120, 122, 127, 132, 135, 138, 139, 142}
        small = {56, 61, 65, 68, 72, 81, 83, 107, 129}
        if index in target:
            set_label("positive", "positive", ["target"], "The crop covers the central green-space island in the roundabout; the source review confirms that island is the unique scene region.", certainty="high")
        elif index in road_context:
            set_label("positive", "negative", ["reference", "context"], "The crop covers surrounding road or roundabout geometry, which is relevant context but not the requested green-space instance.", certainty="medium", quality="too_broad" if index in {60, 76, 87, 88, 90, 94, 98, 100, 101, 104, 105, 114, 125, 128, 134, 144, 145} else "useful", completeness="overbroad" if index in {60, 76, 87, 88, 90, 94, 98, 100, 101, 104, 105, 114, 125, 128, 134, 144, 145} else "complete")
        elif index in vegetation:
            set_label("negative", "negative", [], "The crop is a tree or peripheral vegetation patch; the source frame places it outside the central roundabout island.", certainty="high")
        else:
            set_label("ambiguous", "ambiguous", ["target"], "The crop is too small to separate central-island vegetation from nearby scene vegetation; the source review also finds the requested orientation operation under-specified.", certainty="low", quality="ambiguous", completeness="fragment")

    elif qa == "train_008163":
        broad = {149, 157, 158, 161, 162, 165}
        if index in broad:
            set_label("positive", "positive", ["target"], "The broad crop covers the cleared construction parcel visible in the source frame.", certainty="high", quality="too_broad", completeness="overbroad")
        else:
            set_label("positive", "ambiguous", ["target"], "The crop lies within or at the edge of the cleared construction area, but its small source fragment does not support a complete site boundary judgment.", certainty="medium", quality="ambiguous" if index in {146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 159, 160, 163, 164, 166, 167, 168, 169} else "useful", completeness="fragment")

    elif qa == "train_008503":
        # The source frame contains one large white-roofed building plus many
        # unrelated red-roofed homes. Only masks over the white-roof building
        # are target evidence; scene-spanning masks remain ambiguous.
        if index in {171, 178}:
            set_label("positive", "positive", ["target"], "The crop isolates the large white-roofed building already visible at clip start; first visibility is left-censored at t=0.", certainty="high")
        elif index in {184, 187}:
            set_label("positive", "ambiguous", ["target", "context"], "The crop includes the large white-roofed building together with surrounding grounds and roads, so target presence is supported but the mask is overbroad.", certainty="medium", quality="too_broad", completeness="overbroad")
        else:
            set_label("negative", "negative", [], "The crop is a separate red-roofed home, tree, road, or golf-course region; it is unrelated to the large white-roofed building target.", certainty="high", quality="ambiguous" if index in {173, 182} else "useful", completeness="fragment" if index in {173, 182} else "complete")

    elif qa == "train_010200":
        if row["packed_frame_index"] == 13:
            if index in {209, 210}:
                set_label("negative", "negative", [], "This packed-13 candidate crop covers highway and tennis-court context rather than a baseball field; the category-level source-review visibility is retained separately.", certainty="high", quality="too_broad", completeness="overbroad")
            else:
                set_label("negative", "negative", [], "This packed-13 candidate crop is a tiny unrelated region rather than a baseball field; the category-level source-review visibility is retained separately.", certainty="high", quality="ambiguous", completeness="fragment")
        else:
            broad = {190, 193, 197, 202, 204}
            for_field = "The crop covers a baseball diamond or field complex visible at clip start; multiple diamonds are present, so one singular field cannot be selected."
            set_label("positive", "ambiguous", ["target"], for_field, certainty="medium", quality="too_broad" if index in broad else "useful", completeness="overbroad" if index in broad else ("fragment" if index in {188, 189, 191, 192, 194, 195, 198, 199, 200, 201, 203, 205, 206, 207, 208} else "complete"))

    elif qa == "train_011679":
        target_building = {215, 223, 227, 229, 230, 232, 233, 234, 236}
        context = {217, 219, 235, 237}
        tiny = {214, 216, 218, 220, 221, 222, 224, 225, 226, 228, 231, 238}
        if index in target_building:
            set_label("positive", "positive", ["target"], "The crop covers the named theater complex visible in the source frame; its AMC facade is the source-review target landmark.", certainty="medium")
        elif index in context:
            set_label("positive", "positive", ["reference", "context"], "The crop covers parking or an adjacent building region used as a named distance reference in the source question.", certainty="medium", quality="too_broad" if index in {217, 235, 237} else "useful", completeness="overbroad" if index in {217, 235, 237} else "complete")
        else:
            set_label("ambiguous", "ambiguous", ["target", "reference", "context"], "A small theater, adjacent-building, or parking fragment is visible, but source pixels do not securely assign the named landmark role.", certainty="low", quality="ambiguous", completeness="fragment")

    elif qa == "train_012936":
        set_label("ambiguous", "not_assessable", [], "The crop is a small road user, but source review cannot distinguish a unique motorcycle or its disappearance interval from other vehicles.", certainty="low", quality="ambiguous", completeness="fragment", visibility="ambiguous")

    elif qa == "train_012957":
        building_like = {282, 392}
        broad = {263, 278, 301, 323, 342, 349, 354, 357, 383, 389, 396}
        if index in building_like:
            set_label("ambiguous", "ambiguous", ["reference"], "The crop is building-like and could be a scene reference, but source pixels do not securely verify the white building with green roof.", certainty="low", quality="ambiguous")
        elif index in broad:
            set_label("positive", "not_assessable", ["target", "context"], "The crop contains roadway or multiple road users relevant to the scene, but the question asks about a changing vehicle group rather than one unique instance.", certainty="medium", quality="too_broad", completeness="overbroad")
        else:
            set_label("positive", "not_assessable", ["target"], "The crop contains a visible vehicle in the intersection scene; source review does not support assigning one vehicle as the unique question target.", certainty="high")

    elif qa == "train_014154":
        target = {401, 402, 403, 404, 405, 406, 412, 413, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426}
        reference = {408, 409, 410, 411}
        broad = {407, 414, 415}
        if index in target:
            set_label("positive", "positive", ["target"], "The crop covers the source-visible commercial-looking building on the right side of the apartment complex; the adjudicated relation places it right of the reference.", certainty="medium")
        elif index in reference:
            set_label("positive", "positive", ["reference"], "The crop covers the source-visible multi-unit apartment complex used as the left/right reference landmark.", certainty="medium")
        elif index in broad:
            set_label("positive", "ambiguous", ["target", "reference", "context"], "The crop spans multiple buildings or the surrounding scene, so both named landmarks may be present but the candidate is not a single-instance crop.", certainty="medium", quality="too_broad", completeness="overbroad")
        else:
            set_label("ambiguous", "ambiguous", ["target", "reference"], "A building fragment is visible, but source pixels do not securely assign it to the commercial target or apartment reference.", certainty="low", quality="ambiguous", completeness="fragment")

    elif qa == "train_014264":
        highway_context = {430, 435, 447, 448, 449, 453, 455, 460, 466}
        if index in highway_context:
            set_label("positive", "not_assessable", ["context"], "The crop covers the highway context used by the group-level movement question; no individual vehicle identity is asserted.", certainty="high", quality="too_broad", completeness="overbroad")
        else:
            set_label("positive", "not_assessable", ["target"], "The crop contains a visible vehicle on the highway; the source review supports only group-level movement and not cross-frame identity.", certainty="high")

    else:
        raise AssertionError(f"unhandled QA {qa}")

    assert result["category_relevance"] in CAT
    assert result["correct_instance_relevance"] in INST
    assert set(result["relevant_roles"]) <= ROLES
    target_visibility = target_visibility_for(qa, int(row["packed_frame_index"]))
    category_visibility = category_visibility_for(qa, int(row["packed_frame_index"]))
    result["target_visibility"] = target_visibility
    result["category_visibility"] = category_visibility
    result["visibility"] = target_visibility  # compatibility alias with corrected semantics
    assert result["visibility"] in VIS
    assert result["target_visibility"] in VIS
    assert result["category_visibility"] in VIS
    assert result["crop_visibility"] in {"visible", "ambiguous"}
    assert result["contextual_crop_quality"] in QUALITY
    assert result["mask_completeness"] in COMPLETE
    return result


def counts(rows: list[dict]) -> dict:
    def count(field):
        return dict(sorted(Counter(row[field] for row in rows).items()))

    role_counts = Counter(role for row in rows for role in row["relevant_roles"])
    qa_counts = Counter(row["qa_id"] for row in rows)
    return {
        "total_rows": len(rows),
        "by_qa_id": dict(sorted(qa_counts.items())),
        "by_category_relevance": count("category_relevance"),
        "by_correct_instance_relevance": count("correct_instance_relevance"),
        "by_visibility": count("visibility"),
        "by_target_visibility": count("target_visibility"),
        "by_category_visibility": count("category_visibility"),
        "by_crop_visibility": count("crop_visibility"),
        "by_qa_frame_target_visibility": dict(sorted(Counter(f"{row['qa_id']}|frame_{row['packed_frame_index']}={row['target_visibility']}" for row in rows).items())),
        "by_relevant_role": dict(sorted(role_counts.items())),
        "by_contextual_crop_quality": count("contextual_crop_quality"),
        "by_mask_completeness": count("mask_completeness"),
        "by_certainty": count("certainty"),
    }


def main() -> None:
    manifest = load(MANIFEST)
    selected = manifest["selected_candidates"]
    questions = question_map()
    sources = source_map()
    relations = adjudication_map()
    # Load the independent source-only review as a provenance check.  The
    # adjudication text below resolves the one preliminary bus-note correction.
    second_review = load(SECOND)
    assert second_review["review_completed_before_model_calls"] is True
    rows = [make_row(row, i, questions, sources, relations) for i, row in enumerate(selected)]

    aliases = [row["alias"] for row in rows]
    assert len(rows) == 470, len(rows)
    assert len(set(aliases)) == 470
    assert set(aliases) == {row["alias"] for row in selected}
    required = {
        "alias", "qa_id", "question", "packed_frame_index", "source_frame_index", "timestamp_s",
        "category_relevance", "correct_instance_relevance", "relevant_roles", "visibility", "target_visibility", "category_visibility", "crop_visibility",
        "contextual_crop_quality", "mask_completeness", "source_only_reason", "source_evidence", "certainty",
    }
    assert all(set(row) == required for row in rows)
    assert all(row["source_only_reason"] for row in rows)

    payload = {
        "schema_version": 2,
        "annotation_status": "source_only_manual_candidate_relevance_complete",
        "blinding": {
            "used": [
                "selection_manifest.json",
                "tasks/qcsrr_final_benchmark_selection.json",
                "source_review_primary.json",
                "source_review_second.json",
                "source_review_adjudication.json",
                "anonymous fixed-frame contact sheets",
            ],
            "excluded": [
                "rankings.json.gz",
                "score files",
                "R0/R1/R2 outputs",
                "candidate provenance and proposal-arm metadata",
                "model/provider calls",
            ],
        },
        "label_definitions": {
            "category_relevance": "positive means the visible crop matches the requested category or a named scene reference/context; negative means it is visibly unrelated; ambiguous means source pixels do not resolve the category; not_assessable is reserved for source-resolution limits.",
            "correct_instance_relevance": "positive means the named instance is supported; negative means a visible wrong instance; ambiguous means multiple or unresolved instances; not_assessable means the question is group-level or source resolution cannot support singular identity.",
            "visibility": "compatibility alias for target_visibility; it now means visibility of the singular question target at this source timestamp.",
            "target_visibility": "visibility of the singular question target instance at the selected source timestamp, independent of whether this candidate crop is correct; ambiguous means the category is present but no one instance is uniquely identified.",
            "category_visibility": "visibility of the requested category or scene-level target at the selected source timestamp, independent of singular-instance identity.",
            "crop_visibility": "whether the candidate mask/crop itself contains visible source pixels; retained separately from target visibility.",
        },
        "summary": {
            **counts(rows),
            "denominator_note": "All denominators are the 470 frozen aliases. Counts are row-level and include both packed frames and both retained arms through the manifest selection, while arm identity is intentionally absent from this evaluator label file.",
            "caveats": [
                "train_001478 bus duration is right-censored at the clip end; only the source-visible white/red bus is labeled as the target.",
                "train_008503 first appearance is left-censored at clip start; its later source-review visibility is inconsistent, so any packed-13 extension is uncertain.",
                "train_008163 construction-site duration is framing-bounded and its exact last visible sample is unresolved.",
                "train_012936 has no defensible singular motorcycle identity at source resolution; correct-instance labels are not_assessable.",
                "train_010200 contains multiple baseball diamonds; category_visibility is visible in both source reviews, while target_visibility and correct-instance labels remain ambiguous. Packed-13 candidate crops are still judged by their own masked regions and remain negative where they show highway/tennis context.",
                "The direct packed-13 source frame for train_010200 visually shows highway/tennis context, while both source reviews mark the baseball-field category visible; the split fields preserve this disagreement instead of collapsing it.",
                "train_012957 and train_014264 are group-level vehicle questions; no cross-frame individual identity is asserted.",
                "train_005918 has a visible central island, but the requested orientation operation is under-specified; road/context crops are retained without answering that operation.",
                "train_014154 source review supports the commercial building to the right of the apartment complex; broad multi-building crops remain ambiguous.",
            ],
        },
        "corrections": [
            "train_008503: replaced the blanket positive labels; only aliases whose masked regions isolate the large white-roofed building are positive, while red-roofed homes and unrelated scene regions are negative and broad building-plus-ground masks are ambiguous.",
            "train_010200: kept packed-0 field candidates category-positive but instance-ambiguous; packed-13 candidate labels remain negative when their own masked regions show highway/tennis context, while category_visibility follows the source reviews.",
            "train_010200 visibility: category_visibility follows the two source reviews (visible at both frozen frames), while target_visibility remains ambiguous because no singular diamond is specified; candidate labels remain source-crop judgments.",
            "train_008503 visibility: category and singular target visibility are visible at packed frame 0; a hypothetical packed-13 extension is uncertain because primary and second reviews disagree.",
            "train_001478: target_visibility is absent at packed frame 0 and visible at packed frame 13; crop_visibility remains separate.",
            "train_008163: target_visibility remains visible at packed frame 0; the unresolved construction-site end is recorded as uncertain in the QA caveat rather than interpolated.",
        ],
        "visual_inspection": {
            "anonymous_contact_sheet_pages_checked": 24,
            "direct_source_frames_checked": [
                "source_frames/train_001478/frame-000000.jpg",
                "source_frames/train_001478/frame-000094.jpg",
                "source_frames/train_008503/frame-000000.jpg",
                "source_frames/train_010200/frame-000000.jpg",
                "source_frames/train_010200/frame-000146.jpg",
                "source_frames/train_011679/frame-000000.jpg",
                "source_frames/train_011679/frame-000195.jpg",
                "source_frames/train_014154/frame-000000.jpg",
                "source_frames/train_014154/frame-000195.jpg",
            ],
            "method": "Source context and masked crop panels were reviewed in alias order; no ranking, score, model output, or proposal provenance was used.",
        },
        "validation": {
            "assertions": [
                "470 rows present",
                "470 unique aliases covered exactly once",
                "required fields present on every row",
                "all enum values valid",
                "all source-only reasons non-empty",
            ],
            "selected_aliases": 470,
            "labeled_aliases": len(rows),
        },
        "labels": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
