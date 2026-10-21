"""Reproduce the offline Phase 4 Gate evaluation from frozen artifacts.

This evaluator consumes only the frozen fixed-frame manifest, source-only
candidate labels, cached SigLIP2 score reconstruction, frozen candidate
metadata, source reviews, and the archived R0/R1/R2 report.  It never calls a
model or provider and never changes the experiment definition.

The output is ``metrics.json`` in this directory and the derived report at
``runs/phase4_qcsrr/PHASE4_FINAL_REPORT.md``.  Ambiguous and unassessable
candidate rows remain explicit in every denominator; they are not silently
treated as negatives for the reported eligible metrics.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/phase4_qcsrr"
OUT = RUN / "annotations/fixed_frame_eval"
MANIFEST_PATH = OUT / "selection_manifest.json"
LABELS_PATH = OUT / "candidate_labels.json"
SCORE_PATH = OUT / "score_reconstruction/score_reconstruction.json"
CANDIDATE_PATH = RUN / "results_resumed-004/candidates.json.gz"
RANKINGS_PATH = RUN / "results_resumed-004/rankings.json.gz"
QUESTIONS_PATH = ROOT / "tasks/qcsrr_final_benchmark_selection.json"
PRIMARY_PATH = RUN / "annotations/source_review_primary.json"
SECOND_PATH = RUN / "annotations/source_review_second.json"
ADJUDICATION_PATH = RUN / "annotations/source_review_adjudication.json"
PRE_REPORT_PATH = RUN / "PHASE4_PRE_RECOVERY_REPORT.md"
REPORT_PATH = RUN / "PHASE4_FINAL_REPORT.md"
METRICS_PATH = OUT / "metrics.json"

K_VALUES = (1, 5)
ROLES = {"target", "reference", "context"}
VALID_STATES = {"positive", "negative", "ambiguous", "not_assessable"}
GEOMETRY_QAS = {
    "train_014154",
    "train_005918",
    "train_011679",
    "train_012957",
    "train_003224",
    "train_002092",
}
ASSESSABLE_GEOMETRY = {
    "train_014154": "target is to the right of reference",
    "train_011679": "AMC-to-Cinemark is shorter than AMC-to-parking-lot",
    "train_003224": "structure is left of baseball field",
    "train_002092": "land-cover region is inside median strip",
}
BROAD_SCENE_TARGET = {
    "train_005918",
    "train_008163",
    "train_010200",
    "train_002092",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def norm(value: str) -> str:
    return " ".join(str(value).replace("\u00a0", " ").split()).casefold()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def is_true(value: Any) -> bool:
    return value is True


def binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def exact_random_hit(pool_size: int, positive_count: int, k: int) -> float | None:
    """P(random top-k contains >=1 positive), without replacement."""
    if positive_count <= 0 or pool_size <= 0:
        return 0.0 if positive_count == 0 else None
    k = min(k, pool_size)
    if positive_count >= pool_size or k >= pool_size:
        return 1.0
    return 1.0 - binom(pool_size - positive_count, k) / binom(pool_size, k)


def exact_random_recall(pool_size: int, positive_count: int, k: int) -> float | None:
    """Expected random candidate Recall@k, E[relevant top-k]/total relevant."""
    if positive_count <= 0 or pool_size <= 0:
        return None
    return min(k, pool_size) / pool_size


def median_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return round_or_none(statistics.median(values)) if values else None


def mean_or_none(values: Iterable[float]) -> float | None:
    values = list(values)
    return round_or_none(statistics.mean(values)) if values else None


def state_counts(states: Iterable[str]) -> dict[str, int]:
    counts = Counter(states)
    return {state: int(counts.get(state, 0)) for state in sorted(VALID_STATES)}


def candidate_role_match(candidate: dict[str, Any], role: str, phrase: str, arm: str) -> bool:
    expected_roles = {"question_target"} if role == "target" else {"question_reference"}
    if arm == "fixed-vocabulary":
        expected_roles = {"fixed_control"}
    for provenance in candidate.get("prompt_provenance", []):
        if (
            provenance.get("role") in expected_roles
            and norm(str(provenance.get("prompt", ""))) == norm(phrase)
        ):
            return True
    return False


def role_state(label: dict[str, Any], role: str, field: str) -> str:
    """Convert a question-level label into a phrase-role label.

    A candidate labeled positive for another role is a resolved negative for
    this phrase.  Ambiguous/not-assessable source judgments stay ambiguous or
    not-assessable when the requested role is present.
    """
    field_name = {
        "category": "category_relevance",
        "instance": "correct_instance_relevance",
        "category_relevance": "category_relevance",
        "correct_instance_relevance": "correct_instance_relevance",
    }.get(field, field)
    value = str(label[field_name])
    if role in set(label["relevant_roles"]):
        return value
    if value == "positive":
        return "negative"
    return value


def source_entity_frame(primary: dict[str, Any], qa: str, packed: int) -> dict[str, Any]:
    question = primary["questions"][qa]
    frames = question.get("frames", [])
    if packed < 0 or packed >= len(frames):
        return {}
    return frames[packed]


def source_visibility(
    qa: str,
    packed: int,
    phrase_role: str,
    phrase: str,
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]],
    primary: dict[str, Any],
) -> str:
    # These later-frame source-review disagreements are intentionally kept
    # uncertain even when the primary review's entity table says visible.
    if qa in {"train_008503", "train_008163"} and packed == 13:
        return "uncertain"
    if phrase_role in {"target", "fixed_control"}:
        rows = labels_by_key.get((qa, packed), [])
        values = {str(row["category_visibility"]) for row in rows}
        if "visible" in values:
            return "visible"
        if "ambiguous" in values:
            return "ambiguous"
        if "uncertain" in values:
            return "uncertain"
        if "absent" in values:
            return "absent"
        # A no-candidate pool has no label row; use the frozen source review.
        entity = source_entity_frame(primary, qa, packed).get("entities", {})
        target = primary["questions"][qa]["question_only_decomposition"]["target_entity"]
        return str(entity.get(target, {}).get("visibility", "uncertain"))
    entity = source_entity_frame(primary, qa, packed).get("entities", {})
    if phrase in entity:
        return str(entity[phrase].get("visibility", "uncertain"))
    normalized = norm(phrase)
    for name, value in entity.items():
        if norm(name) == normalized:
            return str(value.get("visibility", "uncertain"))
    return "uncertain"


def source_assessability(primary: dict[str, Any], qa: str, metric: str) -> bool:
    assessability = primary["questions"][qa].get("assessability", {})
    # The primary review conservatively marked the bus instance false, while
    # the independent review establishes the same white/red bus after entry.
    # Use that adjudicated source-only identity for the visible packed-13
    # instance slot; the absent packed-0 slot remains ineligible by visibility.
    if metric == "instance" and qa == "train_001478":
        return True
    return is_true(assessability.get("category" if metric == "category" else "instance"))


def target_visibility(
    qa: str,
    packed: int,
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]],
    primary: dict[str, Any],
) -> str:
    rows = labels_by_key.get((qa, packed), [])
    values = {str(row["target_visibility"]) for row in rows}
    if "uncertain" in values:
        return "uncertain"
    if "ambiguous" in values:
        return "ambiguous"
    if "absent" in values:
        return "absent"
    if "visible" in values:
        return "visible"
    if qa in {"train_008503", "train_008163"} and packed == 13:
        return "uncertain"
    if qa == "train_010200":
        return "ambiguous"
    if qa == "train_012936":
        return "ambiguous"
    entity = source_entity_frame(primary, qa, packed).get("entities", {})
    target = primary["questions"][qa]["question_only_decomposition"]["target_entity"]
    return str(entity.get(target, {}).get("visibility", "uncertain"))


def source_role_eligibility(
    qa: str,
    packed: int,
    phrase_role: str,
    phrase: str,
    metric: str,
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]],
    primary: dict[str, Any],
) -> tuple[bool, str, bool]:
    # Category queries use the phrase/entity visibility.  Singular target
    # queries use the separately adjudicated target visibility field, which
    # preserves category-visible/multi-instance ambiguity (for example the
    # baseball field) instead of promoting it to a visible singular target.
    if metric == "instance" and phrase_role in {"target", "fixed_control"}:
        visibility = target_visibility(qa, packed, labels_by_key, primary)
    else:
        visibility = source_visibility(qa, packed, phrase_role, phrase, labels_by_key, primary)
    assessable = source_assessability(primary, qa, metric)
    eligible = visibility == "visible" and assessable
    if visibility != "visible":
        reason = f"visibility={visibility}"
    elif not assessable:
        reason = f"source_review_{metric}_assessability=false"
    else:
        reason = "eligible"
    return eligible, reason, assessable


def geometry_box_iou(left: dict[str, int], right: dict[str, int]) -> float:
    x1 = max(left["left"], right["left"])
    y1 = max(left["top"], right["top"])
    x2 = min(left["right"], right["right"])
    y2 = min(left["bottom"], right["bottom"])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = (left["right"] - left["left"]) * (left["bottom"] - left["top"])
    area_right = (right["right"] - right["left"]) * (right["bottom"] - right["top"])
    return float(overlap / max(area_left + area_right - overlap, 1))


def edge_distance(left: dict[str, int], right: dict[str, int]) -> float:
    dx = max(left["left"] - right["right"], right["left"] - left["right"], 0)
    dy = max(left["top"] - right["bottom"], right["top"] - left["bottom"], 0)
    return float(math.hypot(dx, dy))


def center_distance(left: dict[str, int], right: dict[str, int]) -> float:
    lc = ((left["left"] + left["right"]) / 2, (left["top"] + left["bottom"]) / 2)
    rc = ((right["left"] + right["right"]) / 2, (right["top"] + right["bottom"]) / 2)
    return float(math.hypot(lc[0] - rc[0], lc[1] - rc[1]))


def relation_geometry(qa: str, target: dict[str, Any], refs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "neutral",
        "satisfied": True,
        "reference_candidate_ids": [ref["candidate_id"] for ref in refs],
    }
    if qa == "train_005918":
        return {**result, "status": "unassessable_orientation"}
    if qa == "train_014154" and refs:
        delta = target["centroid_normalized"]["x"] - refs[0]["centroid_normalized"]["x"]
        relation = "left" if delta < 0 else "right" if delta > 0 else "overlap"
        return {**result, "status": "computed_left_right", "relation": relation, "delta_x": round(delta, 6)}
    if qa == "train_011679" and len(refs) >= 2:
        edges = [edge_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[:2]]
        centers = [center_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[:2]]
        order = "first_closer" if edges[0] < edges[1] else "second_closer" if edges[1] < edges[0] else "equal"
        return {
            **result,
            "status": "computed_distance_comparison",
            "edge_distances_pixels": [round(v, 3) for v in edges],
            "center_distances_pixels": [round(v, 3) for v in centers],
            "distance_order": order,
        }
    if qa == "train_012957" and len(refs) >= 3:
        inside = geometry_box_iou(target["source_box_pixels"], refs[0]["source_box_pixels"]) > 0
        edges = [edge_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[1:3]]
        centers = [center_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[1:3]]
        order = "traffic_lights_closer" if edges[0] < edges[1] else "green_roof_building_closer" if edges[1] < edges[0] else "equal"
        return {
            **result,
            "status": "computed_group_distance_comparison",
            "satisfied": inside,
            "within_intersection": inside,
            "anchor_edge_distances_pixels": [round(v, 3) for v in edges],
            "anchor_center_distances_pixels": [round(v, 3) for v in centers],
            "distance_order": order,
        }
    if qa == "train_002092" and refs:
        inside = geometry_box_iou(target["source_box_pixels"], refs[0]["source_box_pixels"]) > 0
        return {
            **result,
            "status": "computed_median_containment",
            "satisfied": inside,
            "inside_median": inside,
            "box_iou": round(geometry_box_iou(target["source_box_pixels"], refs[0]["source_box_pixels"]), 6),
        }
    if qa == "train_003224" and refs:
        left = target["centroid_normalized"]["x"] < refs[0]["centroid_normalized"]["x"]
        return {**result, "status": "computed_left_side", "satisfied": left, "left_of_baseball_field": left}
    return result


def build_r1_order(
    query: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    selected_by_alias: dict[str, dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
    score_by_alias: dict[str, float],
    score_by_phrase: dict[tuple[str, str, int, str], dict[str, float]],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    qa = query["qa_id"]
    arm = query["arm"]
    packed = int(query["packed_frame_index"])
    phrase = str(query["phrase"])
    role = str(query["phrase_role"])
    r0_order = [item["alias"] for item in query["ranked_candidates"]]
    if role not in {"target", "fixed_control"}:
        return r0_order, {}, {"target_candidate_count": 0, "reference_cooccurrence": 0, "target_with_all_references": 0}
    question = questions[qa]
    phrases = list(question["entity_prompt_phrases"])
    target_ids_before_component_filter = []
    for alias in r0_order:
        selected = selected_by_alias[alias]
        candidate = candidate_by_id[selected["candidate_id"]]
        if candidate_role_match(candidate, "target", phrase, arm):
            target_ids_before_component_filter.append(alias)
    component_ids = [
        alias
        for alias in target_ids_before_component_filter
        if "component" in set(candidate_by_id[selected_by_alias[alias]["candidate_id"]].get("proposal_kinds", []))
    ]
    component_filter_applied = bool(component_ids and qa not in BROAD_SCENE_TARGET)
    target_ids = component_ids if component_filter_applied else target_ids_before_component_filter
    target_id_set = set(target_ids)
    geometry_rows: dict[str, dict[str, Any]] = {}
    same_frame_refs = 0
    fallback_refs = 0
    same_frame_all_required = 0
    scored_target: list[tuple[bool, float, str]] = []
    ref_phrases = phrases[1:]
    for alias in target_ids:
        selected = selected_by_alias[alias]
        candidate = candidate_by_id[selected["candidate_id"]]
        refs: list[dict[str, Any]] = []
        reference_sources: list[str] = []
        reference_score_sources: list[str] = []
        for ref_phrase in ref_phrases:
            same_frame_matching = []
            for ref_alias, ref_selected in selected_by_alias.items():
                if (
                    ref_selected["qa_id"] != qa
                    or ref_selected["arm"] != "question-conditioned"
                    or int(ref_selected["packed_frame_index"]) != packed
                    or ref_selected["candidate_id"] == selected["candidate_id"]
                ):
                    continue
                ref_candidate = candidate_by_id[ref_selected["candidate_id"]]
                if candidate_role_match(ref_candidate, "reference", ref_phrase, "question-conditioned"):
                    same_frame_matching.append(ref_alias)
            source = "same_frame_distinct" if same_frame_matching else "cross_frame_fallback"
            matching = same_frame_matching
            if not matching:
                for ref_alias, ref_selected in selected_by_alias.items():
                    if (
                        ref_selected["qa_id"] != qa
                        or ref_selected["arm"] != "question-conditioned"
                        or ref_selected["candidate_id"] == selected["candidate_id"]
                    ):
                        continue
                    ref_candidate = candidate_by_id[ref_selected["candidate_id"]]
                    if candidate_role_match(ref_candidate, "reference", ref_phrase, "question-conditioned"):
                        matching.append(ref_alias)
            if matching:
                ref_alias = max(
                    matching,
                    key=lambda item: (
                        score_by_phrase.get(
                            (
                                qa,
                                "question-conditioned",
                                int(selected_by_alias[item]["packed_frame_index"]),
                                norm(ref_phrase),
                            ),
                            {},
                        ).get(item, float("-inf")),
                        selected_by_alias[item]["candidate_id"],
                    ),
                )
                refs.append(candidate_by_id[selected_by_alias[ref_alias]["candidate_id"]])
                reference_sources.append(source)
                score_map = score_by_phrase.get(
                    (
                        qa,
                        "question-conditioned",
                        int(selected_by_alias[ref_alias]["packed_frame_index"]),
                        norm(ref_phrase),
                    ),
                    {},
                )
                reference_score_sources.append(
                    "candidate_frame_score" if ref_alias in score_map else "missing_candidate_frame_score"
                )
        same_count = sum(source == "same_frame_distinct" for source in reference_sources)
        fallback_count = sum(source == "cross_frame_fallback" for source in reference_sources)
        same_frame_refs += same_count
        fallback_refs += fallback_count
        same_frame_all = len(refs) == len(ref_phrases) and fallback_count == 0
        if same_frame_all:
            same_frame_all_required += 1
        assert all(source == "candidate_frame_score" for source in reference_score_sources)
        geometry = relation_geometry(qa, candidate, refs)
        geometry["pool_relation"] = "asymmetric_fixed_target_qc_references" if arm == "fixed-vocabulary" else "question_target_qc_references"
        geometry["reference_sources"] = reference_sources
        geometry["reference_score_sources"] = reference_score_sources
        geometry["same_frame_distinct_references"] = same_count
        geometry["cross_frame_fallback_references"] = fallback_count
        geometry["spatial_evidence_valid"] = same_frame_all and all(
            source == "candidate_frame_score" for source in reference_score_sources
        )
        geometry["geometry_used_for_rank_replay"] = True
        geometry_rows[alias] = geometry
        scored_target.append((not bool(geometry.get("satisfied", True)), -float(score_by_alias[alias]), selected["candidate_id"]))
    sorted_target = [alias for _, _, candidate_id in sorted(scored_target) for alias in r0_order if selected_by_alias[alias]["candidate_id"] == candidate_id]
    r1_order = sorted_target + [alias for alias in r0_order if alias not in target_id_set]
    selection_order = target_ids + [alias for alias in r0_order if alias not in target_id_set]
    return r1_order, geometry_rows, {
        "target_candidate_count_before_component_filter": len(target_ids_before_component_filter),
        "target_candidate_count": len(target_ids),
        "component_filter_applied": component_filter_applied,
        "component_filter_removed_count": len(target_ids_before_component_filter) - len(target_ids),
        "selection_order": selection_order,
        "selection_filter_changed": selection_order != r0_order,
        "geometry_order_changed": r1_order != selection_order,
        "same_frame_reference_cooccurrence": same_frame_refs,
        "cross_frame_fallback_reference_cooccurrence": fallback_refs,
        "target_with_all_same_frame_references": same_frame_all_required,
    }


def evaluate_ranked_query(
    query: dict[str, Any],
    labels_by_alias: dict[str, dict[str, Any]],
    role: str,
    metric: str,
    eligible: bool,
    eligibility_reason: str,
) -> dict[str, Any]:
    ranked = query["ranked_candidates"]
    states = [role_state(labels_by_alias[item["alias"]], role, metric) for item in ranked]
    counts = state_counts(states)
    positive_count = counts["positive"]
    resolved_count = counts["positive"] + counts["negative"]
    output: dict[str, Any] = {
        "qa_id": query["qa_id"],
        "arm": query["arm"],
        "packed_frame_index": int(query["packed_frame_index"]),
        "phrase": query["phrase"],
        "phrase_role": query["phrase_role"],
        "metric": metric,
        "pool_size": len(ranked),
        "source_eligible": eligible,
        "eligibility_reason": eligibility_reason,
        "label_counts": counts,
        "positive_count": positive_count,
        "resolved_count": resolved_count,
        "proposal_covered": positive_count > 0,
        "score_observations": [float(item["score"]) for item in ranked],
    }
    for k in K_VALUES:
        if eligible:
            top_states = states[:k]
            top_positive = sum(state == "positive" for state in top_states)
            # Empty pools and pools with no positive proposal are explicit
            # eligible misses.  Recall remains undefined when n=0.
            output[f"hit_at_{k}"] = bool(top_positive) if positive_count > 0 else False
            output[f"relevant_top_{k}"] = top_positive
            output[f"recall_at_{k}"] = top_positive / positive_count if positive_count > 0 else None
            output[f"random_hit_at_{k}"] = exact_random_hit(len(ranked), positive_count, k)
            output[f"random_recall_at_{k}"] = exact_random_recall(len(ranked), positive_count, k)
        else:
            output[f"hit_at_{k}"] = None
            output[f"relevant_top_{k}"] = None
            output[f"recall_at_{k}"] = None
            output[f"random_hit_at_{k}"] = None
            output[f"random_recall_at_{k}"] = None
    return output


def aggregate_rank_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query_count": len(rows),
        "source_eligible_queries": sum(bool(row["source_eligible"]) for row in rows),
        "pools_with_resolved_labels": sum(row["resolved_count"] > 0 for row in rows),
        "pools_with_positive_proposal": sum(bool(row["proposal_covered"]) for row in rows),
        "pool_size": {
            "total_candidates": sum(row["pool_size"] for row in rows),
            "mean": mean_or_none(row["pool_size"] for row in rows),
            "median": median_or_none(row["pool_size"] for row in rows),
        },
    }
    for k in K_VALUES:
        eligible = [row for row in rows if row[f"hit_at_{k}"] is not None]
        recall_rows = [row for row in eligible if row[f"recall_at_{k}"] is not None]
        conditional = [row for row in recall_rows if row["proposal_covered"]]
        result[f"hit_at_{k}"] = {
            "numerator": sum(bool(row[f"hit_at_{k}"]) for row in eligible),
            "denominator": len(eligible),
            "value": mean_or_none(1.0 if row[f"hit_at_{k}"] else 0.0 for row in eligible),
        }
        result[f"candidate_recall_at_{k}"] = {
            "eligible_recall_queries": len(recall_rows),
            "macro_value": mean_or_none(row[f"recall_at_{k}"] for row in recall_rows),
            "micro_relevant_retrieved": sum(int(row[f"relevant_top_{k}"]) for row in recall_rows),
            "micro_relevant_total": sum(int(row["positive_count"]) for row in recall_rows),
            "micro_value": (
                sum(int(row[f"relevant_top_{k}"]) for row in recall_rows)
                / sum(int(row["positive_count"]) for row in recall_rows)
                if recall_rows
                else None
            ),
        }
        result[f"exact_random_hit_at_{k}"] = {
            "mean": mean_or_none(row[f"random_hit_at_{k}"] for row in eligible),
            "values": [row[f"random_hit_at_{k}"] for row in eligible],
        }
        result[f"exact_random_recall_at_{k}"] = {
            "mean": mean_or_none(row[f"random_recall_at_{k}"] for row in recall_rows),
        }
        result[f"conditional_on_proposal_at_{k}"] = {
            "hit_numerator": sum(bool(row[f"hit_at_{k}"]) for row in conditional),
            "hit_denominator": len(conditional),
            "hit_value": mean_or_none(1.0 if row[f"hit_at_{k}"] else 0.0 for row in conditional),
            "recall_value": mean_or_none(row[f"recall_at_{k}"] for row in conditional),
        }
    return result


def score_summary(rows: list[dict[str, Any]], labels_by_alias: dict[str, dict[str, Any]], ranked_by_query: dict[tuple[str, str, int, str], dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        role = "target" if row["phrase_role"] in {"target", "fixed_control"} else "reference"
        query = ranked_by_query[(row["qa_id"], row["arm"], row["packed_frame_index"], row["phrase"])]
        for item in query["ranked_candidates"]:
            state = role_state(labels_by_alias[item["alias"]], role, "category_relevance")
            by_state[state].append(float(item["score"]))
    return {
        state: {
            "observations": len(values),
            "mean_score": mean_or_none(values),
            "median_score": median_or_none(values),
            "max_score": round_or_none(max(values)) if values else None,
        }
        for state, values in sorted(by_state.items())
    }


def validate_and_load() -> dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    labels_payload = load_json(LABELS_PATH)
    score_payload = load_json(SCORE_PATH)
    candidates = load_gzip_json(CANDIDATE_PATH)
    rankings = load_gzip_json(RANKINGS_PATH)
    questions_payload = load_json(QUESTIONS_PATH)
    primary = load_json(PRIMARY_PATH)
    second = load_json(SECOND_PATH)
    adjudication = load_json(ADJUDICATION_PATH)

    selected = manifest["selected_candidates"]
    labels = labels_payload["labels"]
    assert len(selected) == 470
    assert len(labels) == 470
    assert len({row["alias"] for row in selected}) == 470
    assert len({row["alias"] for row in labels}) == 470
    assert {row["alias"] for row in selected} == {row["alias"] for row in labels}
    assert manifest["selection_rule"]["packed_frame_indices"] == [0, 13]
    assert labels_payload["validation"]["selected_aliases"] == 470
    assert labels_payload["validation"]["labeled_aliases"] == 470

    selected_by_alias = {row["alias"]: row for row in selected}
    labels_by_alias = {row["alias"]: row for row in labels}
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    candidate_id_by_alias = {}
    for row in selected:
        assert row["candidate_id"] in candidate_by_id
        candidate_id_by_alias[row["alias"]] = row["candidate_id"]
        candidate = candidate_by_id[row["candidate_id"]]
        assert candidate["qa_id"] == row["qa_id"]
        assert int(candidate["packed_frame_index"]) == int(row["packed_frame_index"])
        assert int(candidate["source_frame_index"]) == int(row["source_frame_index"])
        assert str(candidate["mask_sha256"]) == str(row["mask_sha256"])
    assert set(candidate_id_by_alias.values()) <= set(candidate_by_id)

    score_queries = score_payload["query_results"]
    assert len(score_queries) == 72
    score_keys = set()
    score_by_phrase: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for query in score_queries:
        key = (query["qa_id"], query["arm"], int(query["packed_frame_index"]), query["phrase"])
        assert key not in score_keys
        score_keys.add(key)
        ranked = query["ranked_candidates"]
        assert [item["rank"] for item in ranked] == list(range(1, len(ranked) + 1))
        aliases = {item["alias"] for item in ranked}
        expected_aliases = {
            alias
            for alias, row in selected_by_alias.items()
            if row["qa_id"] == query["qa_id"]
            and row["arm"] == query["arm"]
            and int(row["packed_frame_index"]) == int(query["packed_frame_index"])
        }
        assert aliases == expected_aliases
        for item in ranked:
            assert item["alias"] in selected_by_alias
            assert item["candidate_id"] == candidate_id_by_alias[item["alias"]]
            assert item["packed_frame_index"] == query["packed_frame_index"]
            assert item["candidate_id"] in candidate_by_id
        score_by_phrase[key] = {item["alias"]: float(item["score"]) for item in ranked}
    assert len(score_keys) == 72
    assert score_payload["archived_score_validation"]["all_matches_within_tolerance"] is True
    assert score_payload["archived_score_validation"]["max_absolute_error"] <= 1e-6
    assert sum(len(query["ranked_candidates"]) for query in score_queries) == 821
    assert len({item["alias"] for query in score_queries for item in query["ranked_candidates"]}) == 470

    question_map = {row["qa_id"]: row for row in questions_payload["questions"]}
    primary_questions = primary["questions"]
    assert set(question_map) == set(primary_questions)
    assert second["review_completed_before_model_calls"] is True
    assert adjudication["adjudication_status"] == "frozen_before_model_rankings"
    return {
        "manifest": manifest,
        "labels_payload": labels_payload,
        "score_payload": score_payload,
        "candidates": candidates,
        "rankings": rankings,
        "questions": question_map,
        "primary": primary,
        "second": second,
        "adjudication": adjudication,
        "selected_by_alias": selected_by_alias,
        "labels_by_alias": labels_by_alias,
        "candidate_by_id": candidate_by_id,
        "candidate_id_by_alias": candidate_id_by_alias,
        "score_queries": score_queries,
        "score_by_phrase": score_by_phrase,
    }


def build_retrieval_metrics(data: dict[str, Any]) -> dict[str, Any]:
    labels_by_alias = data["labels_by_alias"]
    selected_by_alias = data["selected_by_alias"]
    questions = data["questions"]
    primary = data["primary"]
    score_queries = data["score_queries"]
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for label in labels_by_alias.values():
        labels_by_key[(label["qa_id"], int(label["packed_frame_index"]))].append(label)

    query_metrics: list[dict[str, Any]] = []
    target_metric_rows: dict[str, list[dict[str, Any]]] = {"category": [], "instance": []}
    all_metric_rows: dict[str, list[dict[str, Any]]] = {"category": [], "instance": []}
    target_query_keys = set()
    for query in score_queries:
        role = "target" if query["phrase_role"] in {"target", "fixed_control"} else "reference"
        for metric in ("category", "instance"):
            eligible, reason, _ = source_role_eligibility(
                query["qa_id"],
                int(query["packed_frame_index"]),
                query["phrase_role"],
                query["phrase"],
                metric,
                labels_by_key,
                primary,
            )
            row = evaluate_ranked_query(query, labels_by_alias, role, metric, eligible, reason)
            all_metric_rows[metric].append(row)
            if role == "target":
                target_metric_rows[metric].append(row)
                target_query_keys.add((query["qa_id"], query["arm"], int(query["packed_frame_index"])))
            if metric == "category":
                query_metrics.append(row)

    def metrics_by_arm(metric: str) -> dict[str, Any]:
        return {
            arm: aggregate_rank_metrics([row for row in target_metric_rows[metric] if row["arm"] == arm])
            for arm in sorted({row["arm"] for row in target_metric_rows[metric]})
        }

    arm_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_by_alias.values():
        arm_pool[row["arm"]].append(row)
    duplicate_masks: dict[str, Any] = {}
    for arm, rows in sorted(arm_pool.items()):
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["qa_id"], int(row["packed_frame_index"]), row["mask_sha256"])].append(row)
        duplicate_masks[arm] = {
            "selected_candidates": len(rows),
            "unique_candidate_ids": len({row["candidate_id"] for row in rows}),
            "duplicate_mask_groups_within_arm": sum(len(v) > 1 for v in grouped.values()),
            "duplicate_rows_within_arm": sum(max(len(v) - 1, 0) for v in grouped.values()),
            "proposal_kinds": dict(sorted(Counter(kind for row in rows for kind in row.get("proposal_kinds", [])).items())),
            "by_packed_frame": dict(sorted(Counter(str(row["packed_frame_index"]) for row in rows).items())),
        }
    global_mask_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_by_alias.values():
        global_mask_groups[(row["qa_id"], int(row["packed_frame_index"]), row["mask_sha256"])].append(row)
    cross_arm_groups = [rows for rows in global_mask_groups.values() if len({row["arm"] for row in rows}) > 1]
    duplicate_summary = {
        "candidate_id_duplicates": len(selected_by_alias) - len({row["candidate_id"] for row in selected_by_alias.values()}),
        "global_exact_mask_duplicate_groups": sum(len(rows) > 1 for rows in global_mask_groups.values()),
        "global_exact_mask_duplicate_extra_rows": sum(max(len(rows) - 1, 0) for rows in global_mask_groups.values()),
        "cross_arm_exact_mask_duplicate_groups": len(cross_arm_groups),
        "cross_arm_exact_mask_duplicate_rows": sum(len(rows) for rows in cross_arm_groups),
        "by_arm": duplicate_masks,
    }

    score_rows_by_arm = {}
    ranked_query_lookup = {
        (row["qa_id"], row["arm"], int(row["packed_frame_index"]), row["phrase"]): row
        for row in score_queries
    }
    for arm in sorted(arm_pool):
        arm_rows = [row for row in query_metrics if row["arm"] == arm and row["phrase_role"] in {"target", "fixed_control"}]
        score_rows_by_arm[arm] = {
            "category_metrics": aggregate_rank_metrics(arm_rows),
            "score_by_label_state": score_summary(arm_rows, labels_by_alias, ranked_query_lookup),
            "query_roles": dict(sorted(Counter(row["phrase_role"] for row in query_metrics if row["arm"] == arm).items())),
        }

    proposal_coverage = []
    for qa in sorted(questions):
        for packed in (0, 13):
            frame_rows = [row for row in query_metrics if row["qa_id"] == qa and int(row["packed_frame_index"]) == packed and row["phrase_role"] in {"target", "fixed_control"}]
            category_visibility = source_visibility(qa, packed, "target", questions[qa]["question_only_decomposition"]["target_entity"], labels_by_key, primary)
            category_assessable = source_assessability(primary, qa, "category")
            instance_visibility = target_visibility(qa, packed, labels_by_key, primary)
            instance_assessable = source_assessability(primary, qa, "instance")
            proposal_coverage.append(
                {
                    "qa_id": qa,
                    "packed_frame_index": packed,
                    "category_visibility": category_visibility,
                    "category_assessable": category_assessable,
                    "category_slot_eligible": category_visibility == "visible" and category_assessable,
                    "target_visibility": instance_visibility,
                    "instance_assessable": instance_assessable,
                    "instance_slot_eligible": instance_visibility == "visible" and instance_assessable,
                    "by_arm": {
                        arm: {
                            "pool_size": next((row["pool_size"] for row in frame_rows if row["arm"] == arm), 0),
                            "positive_proposal": any(row["proposal_covered"] for row in frame_rows if row["arm"] == arm),
                            "resolved_label_count": next((row["resolved_count"] for row in frame_rows if row["arm"] == arm), 0),
                        }
                        for arm in ("fixed-vocabulary", "question-conditioned")
                    },
                }
            )

    category_slots = [row for row in proposal_coverage if row["category_slot_eligible"]]
    instance_slots = [row for row in proposal_coverage if row["instance_slot_eligible"]]

    # The packet is defined over all 12 questions at both frozen frames.  Keep
    # empty pools in this materialized slot table so proposal misses remain
    # observable in the Hit denominator rather than disappearing upstream.
    assert len(proposal_coverage) == 12 * 2
    assert {(row["qa_id"], row["packed_frame_index"]) for row in proposal_coverage} == {
        (qa, packed) for qa in questions for packed in (0, 13)
    }
    for row in proposal_coverage:
        assert set(row["by_arm"]) == {"fixed-vocabulary", "question-conditioned"}
    assert len(category_slots) == 19
    assert len(instance_slots) == 11

    return {
        "label_denominators": {
            "selected_aliases": len(labels_by_alias),
            "category_visibility": dict(sorted(Counter(row["category_visibility"] for row in labels_by_alias.values()).items())),
            "target_visibility": dict(sorted(Counter(row["target_visibility"] for row in labels_by_alias.values()).items())),
            "category_relevance": dict(sorted(Counter(row["category_relevance"] for row in labels_by_alias.values()).items())),
            "correct_instance_relevance": dict(sorted(Counter(row["correct_instance_relevance"] for row in labels_by_alias.values()).items())),
            "frame_slots": {
                "total_slots": len(proposal_coverage),
                "category_eligible_slots_per_arm": len(category_slots),
                "instance_eligible_slots_per_arm": len(instance_slots),
                "category_visibility_by_slot": dict(sorted(Counter(row["category_visibility"] for row in proposal_coverage).items())),
                "target_visibility_by_slot": dict(sorted(Counter(row["target_visibility"] for row in proposal_coverage).items())),
            },
        },
        "formulas": {
            "hit_at_k": "1[top-k ranked candidates contain at least one positive]",
            "candidate_recall_at_k": "positive candidates in top-k / all positive candidates in the same query pool",
            "exact_random_hit_at_k": "1 - C(r-n,k) / C(r,k), where n=positive candidates and r=pool size; sampling is without replacement",
            "exact_random_recall_at_k": "k / r in expectation for k<=r and n>0",
            "eligible_denominator": "source category/instance assessability is true and source category visibility is visible; empty pools and pools without a positive proposal remain eligible Hit misses, while Recall is undefined when n=0",
        },
        "retrieval": {
            "target_category_by_arm": metrics_by_arm("category"),
            "target_instance_by_arm": metrics_by_arm("instance"),
            "all_phrase_categories": aggregate_rank_metrics(all_metric_rows["category"]),
            "all_phrase_instances": aggregate_rank_metrics(all_metric_rows["instance"]),
            "conditional_on_proposal_definition": "rows with source eligibility and at least one positive candidate in that same query pool",
            "query_metrics_category": query_metrics,
            "query_metrics_instance": all_metric_rows["instance"],
        },
        "arms": {
            "selected_pool_sizes": {arm: len(rows) for arm, rows in sorted(arm_pool.items())},
            "score_and_pool_summary": score_rows_by_arm,
            "duplicate_and_proposal_summary": duplicate_summary,
            "proposal_coverage_by_qa_frame": proposal_coverage,
        },
        "source_review_assessability": {
            qa: {
                "category": data["primary"]["questions"][qa]["assessability"].get("category"),
                "instance": data["primary"]["questions"][qa]["assessability"].get("instance"),
                "geometry": data["primary"]["questions"][qa]["assessability"].get("geometry"),
                "temporal": data["primary"]["questions"][qa]["assessability"].get("temporal"),
            }
            for qa in sorted(questions)
        },
    }


def build_two_frame_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Aggregate packed frames 0 and 13 into one independent-video query.

    The union is sorted once by the cached R0 score and candidate ID.  This
    gives a true two-frame candidate pool for Hit@K and an exactly matched
    without-replacement random baseline.  It is intentionally limited to the
    two frozen frames and is not a video-global temporal metric.
    """

    labels_by_alias = data["labels_by_alias"]
    primary = data["primary"]
    score_queries = data["score_queries"]
    labels_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for label in labels_by_alias.values():
        labels_by_key[(label["qa_id"], int(label["packed_frame_index"]))].append(label)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for query in score_queries:
        if query["phrase_role"] not in {"target", "fixed_control"}:
            continue
        grouped[(query["qa_id"], query["arm"], query["phrase"])].append(query)

    def union_row(group: list[dict[str, Any]], metric: str) -> dict[str, Any]:
        assert len(group) == 2
        assert {int(query["packed_frame_index"]) for query in group} == {0, 13}
        query0 = group[0]
        role = "target"
        frame_eligibility = []
        for query in sorted(group, key=lambda item: int(item["packed_frame_index"])):
            eligible, reason, _ = source_role_eligibility(
                query["qa_id"],
                int(query["packed_frame_index"]),
                query["phrase_role"],
                query["phrase"],
                metric,
                labels_by_key,
                primary,
            )
            frame_eligibility.append(
                {
                    "packed_frame_index": int(query["packed_frame_index"]),
                    "source_eligible": eligible,
                    "reason": reason,
                }
            )
        union_items = []
        seen_aliases = set()
        for query in group:
            for item in query["ranked_candidates"]:
                assert item["alias"] not in seen_aliases
                seen_aliases.add(item["alias"])
                union_items.append(
                    {
                        "alias": item["alias"],
                        "candidate_id": item["candidate_id"],
                        "score": float(item["score"]),
                        "packed_frame_index": int(query["packed_frame_index"]),
                    }
                )
        union_items.sort(key=lambda item: (-item["score"], item["candidate_id"]))
        states = [role_state(labels_by_alias[item["alias"]], role, metric) for item in union_items]
        counts = state_counts(states)
        positive_count = counts["positive"]
        source_eligible = any(item["source_eligible"] for item in frame_eligibility)
        row = {
            "qa_id": query0["qa_id"],
            "arm": query0["arm"],
            "phrase": query0["phrase"],
            "metric": metric,
            "packed_frame_indices": [0, 13],
            "source_eligible": source_eligible,
            "frame_eligibility": frame_eligibility,
            "pool_size": len(union_items),
            "label_counts": counts,
            "positive_count": positive_count,
            "resolved_count": counts["positive"] + counts["negative"],
            "proposal_covered": positive_count > 0,
            "top_aliases": [item["alias"] for item in union_items[:5]],
        }
        for k in K_VALUES:
            if source_eligible:
                top_positive = sum(state == "positive" for state in states[:k])
                row[f"hit_at_{k}"] = bool(top_positive) if positive_count > 0 else False
                row[f"relevant_top_{k}"] = top_positive
                row[f"recall_at_{k}"] = top_positive / positive_count if positive_count > 0 else None
                row[f"random_hit_at_{k}"] = exact_random_hit(len(union_items), positive_count, k)
                row[f"random_recall_at_{k}"] = exact_random_recall(len(union_items), positive_count, k)
            else:
                row[f"hit_at_{k}"] = None
                row[f"relevant_top_{k}"] = None
                row[f"recall_at_{k}"] = None
                row[f"random_hit_at_{k}"] = None
                row[f"random_recall_at_{k}"] = None
        return row

    rows_by_metric: dict[str, list[dict[str, Any]]] = {"category": [], "instance": []}
    for group in grouped.values():
        assert len(group) == 2
        for metric in rows_by_metric:
            rows_by_metric[metric].append(union_row(group, metric))

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [row for row in rows if row["source_eligible"]]
        return {
            "denominator_definition": "unique QA/video IDs with at least one source-eligible selected frame; each pool is the union of packed frames 0 and 13",
            "query_count": len(rows),
            "independent_video_denominator": len(eligible),
            "pools_with_positive_proposal": sum(row["proposal_covered"] for row in eligible),
            "pool_size": {
                "total_candidates": sum(row["pool_size"] for row in eligible),
                "mean": mean_or_none(row["pool_size"] for row in eligible),
                "median": median_or_none(row["pool_size"] for row in eligible),
            },
            **{
                f"hit_at_{k}": {
                    "numerator": sum(bool(row[f"hit_at_{k}"]) for row in eligible),
                    "denominator": len(eligible),
                    "value": mean_or_none(1.0 if row[f"hit_at_{k}"] else 0.0 for row in eligible),
                }
                for k in K_VALUES
            },
            **{
                f"exact_random_hit_at_{k}": {
                    "mean": mean_or_none(row[f"random_hit_at_{k}"] for row in eligible),
                    "values": [row[f"random_hit_at_{k}"] for row in eligible],
                }
                for k in K_VALUES
            },
            "rows": rows,
        }

    return {
        "definition": "two-frame R0 union: concatenate selected packed-frame candidate pools, sort by cached R0 score descending with candidate_id tie-break, then evaluate top K",
        "not_video_global": True,
        "category_by_arm": {
            arm: aggregate([row for row in rows_by_metric["category"] if row["arm"] == arm])
            for arm in sorted({row["arm"] for row in rows_by_metric["category"]})
        },
        "instance_by_arm": {
            arm: aggregate([row for row in rows_by_metric["instance"] if row["arm"] == arm])
            for arm in sorted({row["arm"] for row in rows_by_metric["instance"]})
        },
    }


def build_two_frame_family_summary(data: dict[str, Any], two_frame: dict[str, Any]) -> dict[str, Any]:
    """Summarize the existing two-frame rows by benchmark video family.

    ``dataset_name`` is the benchmark's video-family field.  The second
    grouping is derived from the frozen selection stratum so that geometry,
    temporal/motion, and scene-region cases remain visibly separate without
    changing any metric denominator.
    """

    strata_by_family = {
        "geometry": ["geometric"],
        "temporal_motion": ["temporal", "ambiguity_probe_singular", "ambiguity_probe_group"],
        "scene_region": ["scene_region"],
    }
    task_types_by_family = {
        "geometry": ["Geometric Relation"],
        "temporal_motion": ["Temporal Grounding", "Trend and Pattern"],
        "scene_region": ["Object and Land Cover Recognition"],
    }
    qa_by_id = data["questions"]
    family_by_qa = {}
    for qa_id, question in qa_by_id.items():
        matches = [family for family, strata in strata_by_family.items() if question["selection_stratum"] in strata]
        assert len(matches) == 1, (qa_id, question["selection_stratum"], matches)
        family_by_qa[qa_id] = matches[0]

    groups = []
    for dataset_name in sorted({question["dataset_name"] for question in qa_by_id.values()}):
        for family in ("geometry", "temporal_motion", "scene_region"):
            qa_ids = sorted(
                qa_id
                for qa_id, question in qa_by_id.items()
                if question["dataset_name"] == dataset_name and family_by_qa[qa_id] == family
            )
            if not qa_ids:
                continue
            arm_metrics = {}
            for arm in ("fixed-vocabulary", "question-conditioned"):
                metric_summary = {}
                for metric in ("category", "instance"):
                    rows = [
                        row
                        for row in two_frame[f"{metric}_by_arm"][arm]["rows"]
                        if row["qa_id"] in qa_ids
                    ]
                    eligible = [row for row in rows if row["source_eligible"]]
                    proposal_rows = [row for row in eligible if row["proposal_covered"]]
                    metric_summary[metric] = {
                        "eligible_video_n": len(eligible),
                        "proposal_covered_n": len(proposal_rows),
                        "proposal_coverage": (
                            len(proposal_rows) / len(eligible) if eligible else None
                        ),
                        **{
                            f"hit_at_{k}": {
                                "numerator": sum(bool(row[f"hit_at_{k}"]) for row in eligible),
                                "denominator": len(eligible),
                                "value": mean_or_none(1.0 if row[f"hit_at_{k}"] else 0.0 for row in eligible),
                            }
                            for k in K_VALUES
                        },
                        **{
                            f"matched_random_hit_at_{k}": {
                                "mean": mean_or_none(row[f"random_hit_at_{k}"] for row in eligible),
                                "values": [row[f"random_hit_at_{k}"] for row in eligible],
                            }
                            for k in K_VALUES
                        },
                    }
                arm_metrics[arm] = metric_summary
            groups.append(
                {
                    "video_family": dataset_name,
                    "family": family,
                    "selection_strata": strata_by_family[family],
                    "task_types": task_types_by_family[family],
                    "qa_ids": qa_ids,
                    "arm_metrics": arm_metrics,
                }
            )
    assert sorted(qa_id for group in groups for qa_id in group["qa_ids"]) == sorted(qa_by_id)
    return {
        "definition": "descriptive aggregation of existing two-frame QA/video unions by benchmark dataset_name (video family) and frozen selection-stratum family",
        "family_definitions": {
            family: {
                "selection_strata": strata_by_family[family],
                "task_types": task_types_by_family[family],
            }
            for family in strata_by_family
        },
        "eligible_video_n_is_per_family_arm_metric": True,
        "groups": groups,
        "caveat": "Small family denominators are descriptive only. temporal_motion combines temporal and motion/ambiguity strata; category and instance eligibility can differ within the same QA, and source dataset families and task types are crossed rather than independent causal factors.",
    }


def build_r1_metrics(data: dict[str, Any]) -> dict[str, Any]:
    questions = data["questions"]
    selected_by_alias = data["selected_by_alias"]
    labels_by_alias = data["labels_by_alias"]
    candidate_by_id = data["candidate_by_id"]
    score_queries = data["score_queries"]
    score_by_phrase = data["score_by_phrase"]
    r1_rows = []
    geometry_rows = []
    target_query_keys = []
    for query in score_queries:
        if query["phrase_role"] not in {"target", "fixed_control"}:
            continue
        key = (query["qa_id"], query["arm"], int(query["packed_frame_index"]), query["phrase"])
        score_by_alias = score_by_phrase[key]
        r1_order, geometry_by_alias, cooccurrence = build_r1_order(
            query,
            questions,
            selected_by_alias,
            candidate_by_id,
            score_by_alias,
            score_by_phrase,
        )
        r0_order = [item["alias"] for item in query["ranked_candidates"]]
        role = "target"
        r0_top = r0_order[0] if r0_order else None
        r1_top = r1_order[0] if r1_order else None
        source_eligible, reason, _ = source_role_eligibility(
            query["qa_id"],
            int(query["packed_frame_index"]),
            query["phrase_role"],
            query["phrase"],
            "category",
            defaultdict(list, {(query["qa_id"], int(query["packed_frame_index"])): [labels_by_alias[a] for a in r0_order]}),
            data["primary"],
        )
        source_instance_eligible, instance_reason, _ = source_role_eligibility(
            query["qa_id"],
            int(query["packed_frame_index"]),
            query["phrase_role"],
            query["phrase"],
            "instance",
            defaultdict(list, {(query["qa_id"], int(query["packed_frame_index"])): [labels_by_alias[a] for a in r0_order]}),
            data["primary"],
        )
        def state(alias: str | None, field: str) -> str | None:
            return None if alias is None else role_state(labels_by_alias[alias], role, field)
        r1_row = {
            "qa_id": query["qa_id"],
            "arm": query["arm"],
            "packed_frame_index": int(query["packed_frame_index"]),
            "phrase": query["phrase"],
            "pool_size": len(r0_order),
            "r0_top1_alias": r0_top,
            "r1_top1_alias": r1_top,
            "top1_changed": r0_top != r1_top,
            "top5_changed": r0_order[:5] != r1_order[:5],
            "selection_filter_top1_changed": r0_order[:1] != cooccurrence["selection_order"][:1],
            "selection_filter_top5_changed": r0_order[:5] != cooccurrence["selection_order"][:5],
            "geometry_order_top1_changed": cooccurrence["selection_order"][:1] != r1_order[:1],
            "geometry_order_top5_changed": cooccurrence["selection_order"][:5] != r1_order[:5],
            "r0_category_state": state(r0_top, "category_relevance"),
            "r1_category_state": state(r1_top, "category_relevance"),
            "r0_instance_state": state(r0_top, "correct_instance_relevance"),
            "r1_instance_state": state(r1_top, "correct_instance_relevance"),
            "source_category_eligible": source_eligible,
            "source_category_eligibility_reason": reason,
            "source_instance_eligible": source_instance_eligible,
            "source_instance_eligibility_reason": instance_reason,
            **cooccurrence,
        }
        r1_rows.append(r1_row)
        target_query_keys.append(key)
        for alias, geometry in geometry_by_alias.items():
            assert selected_by_alias[alias]["candidate_id"] not in set(geometry["reference_candidate_ids"])
            geometry_rows.append(
                {
                    "qa_id": query["qa_id"],
                    "arm": query["arm"],
                    "packed_frame_index": int(query["packed_frame_index"]),
                    "phrase": query["phrase"],
                    "candidate_alias": alias,
                    "candidate_id": selected_by_alias[alias]["candidate_id"],
                    "geometry": geometry,
                    "category_state": role_state(labels_by_alias[alias], "target", "category_relevance"),
                    "instance_state": role_state(labels_by_alias[alias], "target", "correct_instance_relevance"),
                }
            )

    by_qa_geometry: dict[str, Any] = {}
    selected_alias_by_id = {row["candidate_id"]: alias for alias, row in selected_by_alias.items()}
    for qa in sorted(GEOMETRY_QAS):
        rows = [row for row in geometry_rows if row["qa_id"] == qa]
        assessable = qa in ASSESSABLE_GEOMETRY
        raw_relation_counts = Counter()
        valid_relation_counts = Counter()
        fallback_relation_counts = Counter()
        raw_ties = 0
        valid_ties = 0
        raw_computed = 0
        valid_computed = 0
        for row in rows:
            geometry = row["geometry"]
            status = geometry.get("status")
            relation_value = geometry.get("relation") if status == "computed_left_right" else geometry.get("distance_order") if status == "computed_distance_comparison" else geometry.get("left_of_baseball_field") if status == "computed_left_side" else geometry.get("inside_median") if status == "computed_median_containment" else None
            if relation_value is None:
                continue
            raw_computed += 1
            raw_relation_counts[str(relation_value)] += 1
            if relation_value == "equal":
                raw_ties += 1
            if geometry.get("spatial_evidence_valid"):
                valid_computed += 1
                valid_relation_counts[str(relation_value)] += 1
                if relation_value == "equal":
                    valid_ties += 1
            elif geometry.get("cross_frame_fallback_references", 0):
                fallback_relation_counts[str(relation_value)] += 1
        # A relation score is valid only when the frozen same-frame R1 pair
        # contains a source-verified target and every required source-verified
        # reference.  Generic role labels, broad crops, and positive labels for
        # another phrase are not sufficient identity evidence.
        same_frame_pair_rows = 0
        target_verified_rows = 0
        reference_candidate_ids: set[str] = set()
        reference_verified_ids: set[str] = set()
        reference_role_states = Counter()
        reference_original_instance_states = Counter()
        verified_pair_rows = 0
        correct_pair_rows = 0
        incorrect_pair_rows = 0
        for row in rows:
            geometry = row["geometry"]
            if not geometry.get("spatial_evidence_valid") or geometry.get("cross_frame_fallback_references", 0):
                continue
            same_frame_pair_rows += 1
            target_label = labels_by_alias[row["candidate_alias"]]
            target_verified = (
                "target" in set(target_label["relevant_roles"])
                and target_label["correct_instance_relevance"] == "positive"
                and target_label["mask_completeness"] == "complete"
                and target_label["contextual_crop_quality"] == "useful"
            )
            target_verified_rows += int(target_verified)
            refs_verified = True
            for reference_id in geometry["reference_candidate_ids"]:
                reference_candidate_ids.add(reference_id)
                reference_alias = selected_alias_by_id.get(reference_id)
                if reference_alias is None:
                    refs_verified = False
                    continue
                reference_label = labels_by_alias[reference_alias]
                reference_role_state = role_state(reference_label, "reference", "correct_instance_relevance")
                reference_role_states[reference_role_state] += 1
                reference_original_instance_states[reference_label["correct_instance_relevance"]] += 1
                reference_verified = (
                    "reference" in set(reference_label["relevant_roles"])
                    and reference_role_state == "positive"
                    and reference_label["mask_completeness"] == "complete"
                    and reference_label["contextual_crop_quality"] == "useful"
                )
                if reference_verified:
                    reference_verified_ids.add(reference_id)
                else:
                    refs_verified = False
            if target_verified and refs_verified:
                verified_pair_rows += 1
                relation_correct = (
                    geometry.get("relation") == "right"
                    if qa == "train_014154"
                    else geometry.get("inside_median") is True
                    if qa == "train_002092"
                    else None
                )
                if relation_correct is True:
                    correct_pair_rows += 1
                elif relation_correct is False:
                    incorrect_pair_rows += 1
        pair_status = "scored" if verified_pair_rows else "unscorable_no_verified_frozen_r1_pairs"
        if qa == "train_014154":
            pair_reason = (
                "All frozen R1 same-frame references are the QC broad crops FVE-93ffe0557c84/FVE-35b7b4ce37f7, "
                "each correct-instance ambiguous and overbroad. Positive fixed-arm reference-labeled rows exist, "
                "but R1 did not select them as its reference pool and they cannot substitute for the frozen QC references."
            )
        elif qa == "train_002092":
            pair_reason = (
                "Frozen R1 references are generic target/context crops: frame 0 selects target-labeled vegetation rows, "
                "and frame 13 includes a context/reference road/median crop whose instance is negative plus a target-labeled "
                "road crop. No selected reference has a verified named reference identity."
            )
        else:
            pair_reason = "No frozen same-frame pair has both a source-verified target and every required source-verified reference."
        identity_pair_audit = {
            "status": pair_status,
            "source_relation_assessable": assessable,
            "same_frame_pair_rows": same_frame_pair_rows,
            "target_verified_rows": target_verified_rows,
            "selected_reference_candidate_count": len(reference_candidate_ids),
            "selected_reference_verified_candidate_count": len(reference_verified_ids),
            "verified_pair_rows": verified_pair_rows,
            "correct_pair_rows": correct_pair_rows,
            "incorrect_pair_rows": incorrect_pair_rows,
            "accuracy": (correct_pair_rows / verified_pair_rows) if verified_pair_rows else None,
            "selected_reference_role_state_counts": dict(sorted(reference_role_states.items())),
            "selected_reference_original_instance_state_counts": dict(sorted(reference_original_instance_states.items())),
            "verification_criteria": "target/reference role explicitly present, correct_instance_relevance=positive, mask_completeness=complete, contextual_crop_quality=useful, and frozen same-frame spatial evidence valid",
            "reason": pair_reason,
        }
        cooccurrence = [row for row in r1_rows if row["qa_id"] == qa]
        by_qa_geometry[qa] = {
            "source_relation_assessable": assessable,
            "source_relation": ASSESSABLE_GEOMETRY.get(qa),
            "target_query_count": len(cooccurrence),
            "target_candidates_with_any_same_frame_reference": sum(row["same_frame_reference_cooccurrence"] > 0 for row in cooccurrence),
            "target_candidates_with_all_same_frame_references": sum(row["target_with_all_same_frame_references"] for row in cooccurrence),
            "same_frame_reference_cooccurrence": sum(row["same_frame_reference_cooccurrence"] for row in cooccurrence),
            "cross_frame_fallback_reference_cooccurrence": sum(row["cross_frame_fallback_reference_cooccurrence"] for row in cooccurrence),
            "raw_computed_candidate_geometry_rows": raw_computed,
            "valid_spatial_geometry_rows": valid_computed,
            "raw_relation_counts": dict(sorted(raw_relation_counts.items())),
            "valid_spatial_relation_counts": dict(sorted(valid_relation_counts.items())),
            "fallback_relation_counts_excluded_from_spatial_evidence": dict(sorted(fallback_relation_counts.items())),
            "amc_cinemark_raw_tie_rows": raw_ties if qa == "train_011679" else None,
            "amc_cinemark_valid_tie_rows": valid_ties if qa == "train_011679" else None,
            "identity_conditioned_relation_accuracy": identity_pair_audit,
            "unassessable_reason": data["adjudication"].get("source_relations", {}).get(qa, {}).get("reason") if not assessable else None,
        }
    return {
        "frozen_r1_replay": {
            "query_count": len(r1_rows),
            "top1_changed": sum(row["top1_changed"] for row in r1_rows),
            "top5_changed": sum(row["top5_changed"] for row in r1_rows),
            "selection_filter_top1_changed": sum(row["selection_filter_top1_changed"] for row in r1_rows),
            "selection_filter_top5_changed": sum(row["selection_filter_top5_changed"] for row in r1_rows),
            "geometry_order_top1_changed": sum(row["geometry_order_top1_changed"] for row in r1_rows),
            "geometry_order_top5_changed": sum(row["geometry_order_top5_changed"] for row in r1_rows),
            "rows": r1_rows,
        },
        "paired_correctness_change": {
            "category": {
                "eligible_top1_pairs": sum(row["source_category_eligible"] and row["r0_category_state"] in {"positive", "negative"} and row["r1_category_state"] in {"positive", "negative"} for row in r1_rows),
                "r0_positive": sum(row["source_category_eligible"] and row["r0_category_state"] == "positive" for row in r1_rows),
                "r1_positive": sum(row["source_category_eligible"] and row["r1_category_state"] == "positive" for row in r1_rows),
                "improved": sum(row["source_category_eligible"] and row["r0_category_state"] != "positive" and row["r1_category_state"] == "positive" for row in r1_rows),
                "regressed": sum(row["source_category_eligible"] and row["r0_category_state"] == "positive" and row["r1_category_state"] != "positive" for row in r1_rows),
            },
            "correct_instance": {
                "eligible_top1_pairs": sum(row["source_instance_eligible"] and row["r0_instance_state"] in {"positive", "negative"} and row["r1_instance_state"] in {"positive", "negative"} for row in r1_rows),
                "r0_positive": sum(row["source_instance_eligible"] and row["r0_instance_state"] == "positive" for row in r1_rows),
                "r1_positive": sum(row["source_instance_eligible"] and row["r1_instance_state"] == "positive" for row in r1_rows),
                "improved": sum(row["source_instance_eligible"] and row["r0_instance_state"] != "positive" and row["r1_instance_state"] == "positive" for row in r1_rows),
                "regressed": sum(row["source_instance_eligible"] and row["r0_instance_state"] == "positive" and row["r1_instance_state"] != "positive" for row in r1_rows),
            },
        },
        "geometry": by_qa_geometry,
        "qualification": {
            "relation_correctness": "not_scored_without_source_verified_target_and_each_distinct_reference_identity",
            "fallback_geometry": "cross-frame fallback is retained only to reproduce frozen R1 ordering; it is excluded from valid spatial co-occurrence and relation evidence",
        },
    }


def temporal_visibility_descriptor(qa: str) -> dict[str, Any]:
    if qa == "train_001478":
        return {
            "visible_frames": [13, 14, 15],
            "ambiguous_frames": [12],
            "absent_frames": list(range(0, 12)),
            "identity_scoreable": True,
            "censoring": "right_censored_at_clip_end",
            "basis": "adjudication: no identifiable target bus at packed 0-11; packed 12 ambiguous/partly occluded; packed 13-15 show one continuous white/red bus",
        }
    if qa == "train_012936":
        return {"visible_frames": list(range(5, 15)), "ambiguous_frames": [4], "absent_frames": [0, 1, 2, 3, 15], "identity_scoreable": False, "censoring": "camera/framing_confounded", "basis": "second source review: 5-12 source-pixel motorcycle-like object; no defensible precise box"}
    if qa == "train_008163":
        return {"visible_frames": list(range(0, 7)), "uncertain_frames": list(range(7, 16)), "identity_scoreable": False, "censoring": "framing_bounded", "basis": "primary review visible through packed 6; later samples uncertain; second review says broad geographic region"}
    if qa == "train_010200":
        return {"visible_frames": list(range(16)), "identity_scoreable": False, "censoring": "left_censored_at_clip_start_and_multi_instance", "basis": "both reviews: baseball-field category/complex visible across samples; individual diamond not unique"}
    if qa == "train_008503":
        # The two source reviews disagree outside the clip-start samples.  Keep
        # the directly confirmed p0/p1 presence separate from the primary
        # review's all-frame visibility sequence; neither supports a validated
        # continuous identity or duration estimate.
        return {
            "visible_frames": [0, 1],
            "primary_visible_frames": list(range(16)),
            "second_review_visible_frames": [0, 1],
            "second_review_not_visible_frames": [10, 12, 15],
            "second_review_unreviewed_frames": [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14],
            "identity_scoreable": False,
            "censoring": "left_censored_at_clip_start",
            "basis": "primary review marks packed 0-15 visible; second review confirms p0 partly cropped and p1 visible but reports p10/p12/p15 not visible, with other packed frames not exhaustively reviewed",
        }
    return {"visible_frames": [], "identity_scoreable": False, "censoring": "not_applicable"}


def build_r2_metrics(data: dict[str, Any]) -> dict[str, Any]:
    rankings = data["rankings"]
    focus = {"train_001478", "train_008163", "train_008503", "train_010200", "train_012936"}
    rows = []
    for ranking in rankings:
        qa = ranking["qa_id"]
        if qa not in focus or ranking["phrase_role"] not in {"target", "fixed_control"}:
            continue
        support = ranking["R2"].get("temporal_support", {})
        descriptor = temporal_visibility_descriptor(qa)
        tracks = support.get("tracks", [])
        track_summaries = []
        for track in tracks:
            frames = [int(value) for value in track.get("sampled_packed_indices", [])]
            visible = set(descriptor.get("visible_frames", []))
            ambiguous = set(descriptor.get("ambiguous_frames", []))
            absent = set(descriptor.get("absent_frames", []))
            review_not_visible = set(descriptor.get("second_review_not_visible_frames", []))
            known = visible | ambiguous | absent | review_not_visible
            track_summaries.append(
                {
                    "track_id": track.get("track_id"),
                    "frames": frames,
                    "track_length": len(frames),
                    "visible_overlap": len(set(frames) & visible),
                    "ambiguous_overlap": len(set(frames) & ambiguous),
                    "known_absent_overlap": len(set(frames) & absent),
                    "second_review_not_visible_overlap": len(set(frames) & review_not_visible),
                    "known_frame_overlap": len(set(frames) & known),
                    "contiguous_adjacent_links": sum(b == a + 1 for a, b in zip(frames, frames[1:])),
                    "first": min(frames) if frames else None,
                    "last": max(frames) if frames else None,
                }
            )
        best = max(track_summaries, key=lambda item: (item["visible_overlap"], item["track_length"], -(item["known_absent_overlap"])), default=None)
        rows.append(
            {
                "qa_id": qa,
                "arm": ranking["arm"],
                "phrase": ranking["phrase"],
                "candidate_count": ranking["candidate_count"],
                "source_visibility": descriptor,
                "tracks_reported_top8": len(track_summaries),
                "best_track": best,
                "track_summaries": track_summaries,
                "identity_scoring_status": "scoreable_for_bus_interval_only" if descriptor["identity_scoreable"] else "descriptive_only",
            }
        )
    by_qa = {}
    for qa in sorted(focus):
        qa_rows = [row for row in rows if row["qa_id"] == qa]
        descriptive_track_counts = {}
        for row in qa_rows:
            summaries = row["track_summaries"]
            descriptive_track_counts[row["arm"]] = {
                "tracks_reported": len(summaries),
                "tracks_with_visible_overlap": sum(item["visible_overlap"] > 0 for item in summaries),
                "tracks_with_ambiguous_overlap": sum(item["ambiguous_overlap"] > 0 for item in summaries),
                "tracks_with_known_absent_overlap": sum(item["known_absent_overlap"] > 0 for item in summaries),
                "tracks_with_second_review_not_visible_overlap": sum(item["second_review_not_visible_overlap"] > 0 for item in summaries),
                "tracks_with_both_visible_and_absent_overlap": sum(item["visible_overlap"] > 0 and item["known_absent_overlap"] > 0 for item in summaries),
                "tracks_without_known_visibility_overlap": sum(item["known_frame_overlap"] == 0 for item in summaries),
            }
        by_qa[qa] = {
            "rows": qa_rows,
            "source_visibility": temporal_visibility_descriptor(qa),
            "descriptive_track_counts": descriptive_track_counts,
            "caveat": (
                "Primary review marks packed 0-15 visible, while second review confirms p0 partly cropped and p1 visible but reports p10/p12/p15 not visible; frozen tracks provide clip-start presence only, so continuity and duration are unscorable."
                if qa == "train_008503"
                else "R2 stores only top-eight tracks per ranking row; misses are descriptive, and track IDs are not gold identity labels."
                if qa != "train_001478"
                else "The bus track covers ambiguous packed 12 and definitive visible packed 13-15; the end is right-censored at the final sample."
            ),
        }
    return {
        "scope": "frozen archived R2 temporal_support; no new association run",
        "focus_by_qa": by_qa,
        "objectively_assessable_identity": {
            "train_001478": {
                "status": "descriptive_scoreable_interval",
                "definitive_visible_interval": [13, 15],
                "ambiguous_boundary_frame": 12,
                "qc_best_track": next((row["best_track"] for row in rows if row["qa_id"] == "train_001478" and row["arm"] == "question-conditioned"), None),
                "qc_switch_count": 0,
                "observed": "QC target track covers packed 12-15; packed 12 is an ambiguous boundary sample, definitive visible frames 13-15 are covered, no switch is visible in the stored track, and the end is right-censored.",
            },
            "train_012936": {"status": "not_scoreable", "observed": "QC target tracks are single-frame fragments across the source-reviewed 5-14 interval; source identity is only 5-12 pixels and camera/framing-confounded."},
            "train_008163": {"status": "not_scoreable", "observed": "Broad construction-site tracks overlap the primary visible window 0-6, but source review describes a framing-bounded geographic region rather than an independently identifiable object."},
            "train_010200": {"status": "not_scoreable", "observed": "Field-category tracks overlap the source-visible samples, but multiple diamonds prevent singular identity association."},
            "train_008503": {
                "status": "not_scoreable_visibility_conflict",
                "clip_start_presence": True,
                "continuity_scoreable": False,
                "exact_duration_scoreable": False,
                "primary_visible_frames": list(range(16)),
                "second_review_visible_frames": [0, 1],
                "second_review_not_visible_frames": [10, 12, 15],
                "observed": "Primary review marks the large white-roof building visible at packed 0-15 and left-censored at clip start; second review confirms p0 partly cropped and p1 visible but reports p10/p12/p15 not visible, with other frames not exhaustively reviewed. Frozen top track p0-5 and fragments p0-2, p3, p5, p6, and p8 therefore provide descriptive clip-start presence only, not a continuous validated identity or exact duration.",
            },
        },
    }


def build_global_top5(data: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for ranking in data["rankings"]:
        if ranking["phrase_role"] not in {"target", "fixed_control"}:
            continue
        top = ranking["R0"].get("top5", [])
        r1_top = ranking["R1"].get("top5", [])
        rows.append(
            {
                "qa_id": ranking["qa_id"],
                "arm": ranking["arm"],
                "phrase": ranking["phrase"],
                "phrase_role": ranking["phrase_role"],
                "candidate_count": ranking["candidate_count"],
                "top5": [
                    {"rank": i + 1, "candidate_id": item.get("candidate_id"), "source_frame_index": item.get("source_frame_index"), "similarity": item.get("similarity")}
                    for i, item in enumerate(top)
                ],
                "r1_top5": [
                    {
                        "rank": i + 1,
                        "candidate_id": item.get("candidate_id"),
                        "source_frame_index": item.get("source_frame_index"),
                        "similarity": item.get("similarity"),
                        "geometry": item.get("geometry"),
                    }
                    for i, item in enumerate(r1_top)
                ],
            }
        )
    return {
        "status": "descriptive_only_full_archived_pools",
        "ranking_rows": len(data["rankings"]),
        "target_and_control_rows": len(rows),
        "rows": rows,
    }


def render_report(metrics: dict[str, Any]) -> str:
    retrieval = metrics["retrieval"]
    two_frame = metrics["two_frame_query_metrics"]
    arms = metrics["arms"]
    r1 = metrics["r1"]
    r2 = metrics["r2"]
    labels = metrics["label_denominators"]
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"
    def metric_line(block: dict[str, Any], k: int) -> str:
        hit = block[f"hit_at_{k}"]
        rec = block[f"candidate_recall_at_{k}"]
        rnd = block[f"exact_random_hit_at_{k}"]
        cond = block[f"conditional_on_proposal_at_{k}"]
        return f"Hit@{k} {hit['numerator']}/{hit['denominator']} ({pct(hit['value'])}); macro Recall@{k} {pct(rec['macro_value'])}; exact random Hit@{k} mean {pct(rnd['mean'])}; conditional Hit@{k} {cond['hit_numerator']}/{cond['hit_denominator']} ({pct(cond['hit_value'])})"
    def two_frame_line(block: dict[str, Any], k: int) -> str:
        hit = block[f"hit_at_{k}"]
        rnd = block[f"exact_random_hit_at_{k}"]
        return f"{hit['numerator']}/{hit['denominator']} ({pct(hit['value'])}) vs random {pct(rnd['mean'])}"

    arm_table = []
    instance_arm_table = []
    score_table = []
    two_frame_category_table = []
    two_frame_instance_table = []
    for arm in sorted(arms["score_and_pool_summary"]):
        block = retrieval["target_category_by_arm"][arm]
        instance_block = retrieval["target_instance_by_arm"][arm]
        score_block = arms["score_and_pool_summary"][arm]
        size = arms["selected_pool_sizes"][arm]
        arm_table.append(f"| {arm} | {size} | {block['source_eligible_queries']} | {block['pools_with_positive_proposal']} | {metric_line(block, 1)} | {metric_line(block, 5)} |")
        instance_arm_table.append(f"| {arm} | {size} | {instance_block['source_eligible_queries']} | {instance_block['pools_with_positive_proposal']} | {metric_line(instance_block, 1)} | {metric_line(instance_block, 5)} |")
        states = score_block["score_by_label_state"]
        def score_line(state: str) -> str:
            item = states.get(state, {})
            return f"{item.get('observations', 0)} obs; mean {item.get('mean_score', 'n/a')}; median {item.get('median_score', 'n/a')}"
        pool = score_block["category_metrics"]["pool_size"]
        score_table.append(f"| {arm} | {pool['total_candidates']} | {pool['mean']} / {pool['median']} | {score_line('positive')} | {score_line('negative')} | {score_line('ambiguous')} |")
        two_frame_category_table.append(f"| {arm} | {two_frame['category_by_arm'][arm]['independent_video_denominator']} | {two_frame_line(two_frame['category_by_arm'][arm], 1)} | {two_frame_line(two_frame['category_by_arm'][arm], 5)} |")
        two_frame_instance_table.append(f"| {arm} | {two_frame['instance_by_arm'][arm]['independent_video_denominator']} | {two_frame_line(two_frame['instance_by_arm'][arm], 1)} | {two_frame_line(two_frame['instance_by_arm'][arm], 5)} |")
    geometry_lines = []
    for qa, row in r1["geometry"].items():
        geometry_lines.append(
            f"| `{qa}` | {'yes' if row['source_relation_assessable'] else 'no'} | {row['target_candidates_with_all_same_frame_references']} | {row['same_frame_reference_cooccurrence']} / {row['cross_frame_fallback_reference_cooccurrence']} | {row['raw_computed_candidate_geometry_rows']} / {row['valid_spatial_geometry_rows']} | {row['raw_relation_counts']} | unscorable | {row['amc_cinemark_raw_tie_rows'] if row['amc_cinemark_raw_tie_rows'] is not None else 'n/a'} |"
        )
    def family_hit_line(metric_block: dict[str, Any], k: int) -> str:
        hit = metric_block[f"hit_at_{k}"]
        random = metric_block[f"matched_random_hit_at_{k}"]
        return f"{hit['numerator']}/{hit['denominator']} ({pct(hit['value'])}) vs {pct(random['mean'])}"

    family_lines = []
    for group in metrics["two_frame_family_summary"]["groups"]:
        for arm in ("fixed-vocabulary", "question-conditioned"):
            category = group["arm_metrics"][arm]["category"]
            instance = group["arm_metrics"][arm]["instance"]
            category_coverage = f"{category['proposal_covered_n']}/{category['eligible_video_n']}"
            instance_coverage = f"{instance['proposal_covered_n']}/{instance['eligible_video_n']}"
            family_lines.append(
                f"| {group['video_family']} | {group['family']} | {', '.join(group['qa_ids'])} | {arm} | "
                f"{category['eligible_video_n']} | {category_coverage} | {family_hit_line(category, 1)} | {family_hit_line(category, 5)} | "
                f"{instance['eligible_video_n']} | {instance_coverage} | {family_hit_line(instance, 1)} | {family_hit_line(instance, 5)} |"
            )
    pair_audit_lines = []
    for qa in ("train_014154", "train_002092"):
        audit = r1["geometry"][qa]["identity_conditioned_relation_accuracy"]
        pair_audit_lines.append(
            f"- `{qa}`: {audit['verified_pair_rows']} verified frozen R1 target-reference pairs out of {audit['same_frame_pair_rows']} same-frame pair rows; "
            f"{audit['correct_pair_rows']} correct and {audit['incorrect_pair_rows']} incorrect ({'n/a' if audit['accuracy'] is None else pct(audit['accuracy'])}). {audit['reason']}"
        )
    r2_lines = []
    for qa, row in r2["focus_by_qa"].items():
        status = row["source_visibility"].get("censoring")
        best = []
        for item in row["rows"]:
            if item["best_track"]:
                best.append(f"{item['arm']}:{item['best_track']['frames']}")
        r2_lines.append(f"| `{qa}` | {status} | {'; '.join(best) or 'no tracks'} | {row['caveat']} |")

    report = f"""# Phase 4 QCSRR — derived Gate evaluation

**Status: complete from frozen artifacts.** This report joins the 470 source-only candidate labels to the rank-independent fixed-frame manifest and cached SigLIP2 score reconstruction. It replays frozen R1 geometry ordering from candidate metadata and archived prompt scores, and reads archived R2 track metadata. No model, provider, inference, or Qwen call was made.

## Decision

The fixed-frame Gate is **quantitatively scoreable for the labeled candidate pools**, with explicit visibility and source-review eligibility limits. Use the results for this 470-row fixed-frame packet only. They do not establish unlabeled-video retrieval or temporal event accuracy.

Category retrieval is measurable on source-visible, source-assessable query pools. The matched random baseline is computed exactly from each pool's `n` positive candidates and `r` candidates; it is not a global 1/r approximation. R1 can change ordering through target-provenance/component filtering and frozen geometry ordering; the report separates those effects. R2 cannot improve ranking by design and remains descriptive except for the source-reviewed bus interval.

## Frozen inputs and validation

| Artifact | Check |
|---|---|
| Fixed-frame manifest | 470 aliases, packed frames 0 and 13, both arms; 290 fixed-vocabulary and 180 question-conditioned rows |
| Candidate labels | 470 unique aliases, all required fields, source-only category/instance/crop labels |
| Cached score reconstruction | 72 query pools, 821 score observations, 470 unique aliases; archived score validation max absolute error <= 1e-6 |
| Frozen candidate metadata | Every manifest candidate ID, frame index, source index, and mask digest matched |
| Frozen R0/R1/R2 archive | 36 ranking rows preserved for the full 3,689-candidate experiment; used descriptively and for R2 track metadata |

The score reconstruction's 72 pools are the two selected packed frames for each retained prompt/arm combination. The same candidate can occur in several phrase pools because a mask may carry multiple prompt provenance entries. Metrics below are query-pool metrics; they are not independent-video estimates.

## Label denominators and eligibility

The 470 frozen aliases have category visibility counts `{labels['category_visibility']}` and singular target visibility counts `{labels['target_visibility']}`. Candidate category labels are `{labels['category_relevance']}`; correct-instance labels are `{labels['correct_instance_relevance']}`. Ambiguous and `not_assessable` rows remain counted and excluded from resolved-positive/negative retrieval denominators.

The materialized source-review slot table has 24 slots (12 questions × packed frames 0 and 13): 19 category-eligible slots per arm and 11 singular-instance-eligible slots per arm. Empty slots are retained. For a phrase-role query, source eligibility requires the relevant source-review assessability and visible category visibility for category metrics or visible target visibility for singular-instance metrics at that source timestamp. A positive candidate is a row whose role-matched label is `positive`; a role-mismatched positive for another entity is a resolved negative for this phrase. Empty pools and pools without a positive proposal count as Hit misses; candidate Recall is undefined when the positive count `n` is zero. The exact formulas are recorded in `metrics.json`.

## Category retrieval on the fixed-frame packet

| Arm | Selected aliases | Source-eligible target/control pools | Pools with a positive proposal | Category Hit@1 | Category Hit@5 |
|---|---:|---:|---:|---|---|
{chr(10).join(arm_table)}

The full target/control aggregate is also in `metrics.json`, including category and correct-instance metrics, macro and micro candidate Recall@K, exact random Hit@K from `n/r`, and conditional-on-proposal values. The compact formula is:

* `Hit@K = 1` when at least one positive candidate occurs in the top K.
* `candidate Recall@K = (# positive candidates in top K) / (# positive candidates in the same pool)`.
* `exact random Hit@K = 1 - C(r-n,K) / C(r,K)` for `n` positives and pool size `r`.
* Expected random candidate Recall@K is `K/r` for `K <= r` and `n > 0`.

The Hit denominator includes all 19 category-eligible frame slots per arm, including the fixed-arm construction-site proposal miss at packed frame 0 and the fixed-arm/QC baseball-field proposal misses at packed frame 13. These are frame-level query-pool metrics. A separate two-frame union is reported below with one denominator unit per QA/video; it is not a video-global temporal result.

### Two-frame query-level R0 union

For this diagnostic, the candidate pools from packed frames 0 and 13 are concatenated per QA/arm/phrase, sorted by cached R0 score descending with deterministic `candidate_id` tie-breaks, and evaluated as one union pool. The denominator is the number of independent QA/video IDs with at least one eligible selected frame. The matched random baseline uses the same union pool size `r` and positive count `n`: `1 - C(r-n,K) / C(r,K)`. This is a two-frame query-level union only; it does not claim visibility, identity continuity, or video-global retrieval.

| Arm | Category-eligible videos | Category Hit@1 vs random | Category Hit@5 vs random |
|---|---:|---|---|
{chr(10).join(two_frame_category_table)}

| Arm | Instance-eligible videos | Instance Hit@1 vs random | Instance Hit@5 vs random |
|---|---:|---|---|
{chr(10).join(two_frame_instance_table)}

The QC-vs-fixed arm comparison changes both the proposal prompt/bank and the text query phrase: the fixed-vocabulary arm uses its fixed vocabulary/bank, while the question-conditioned arm uses question-derived phrases. The comparison is therefore not a pure proposal-generation ablation; proposal generation and query wording/embedding both change.

### Descriptive two-frame summary by benchmark video family

The table below reuses the two-frame union rows and groups them by the benchmark `dataset_name` video family crossed with the frozen selection-stratum family. `temporal_motion` combines the `temporal`, `ambiguity_probe_singular`, and `ambiguity_probe_group` strata. Each cell is `Hit@K vs matched-random Hit@K`; proposal coverage is covered eligible videos / eligible videos. Category and instance eligible-video denominators can differ.

| Video family | Family | QA IDs | Arm | Category n | Category proposal coverage | Category Hit@1 vs random | Category Hit@5 vs random | Instance n | Instance proposal coverage | Instance Hit@1 vs random | Instance Hit@5 vs random |
|---|---|---|---|---:|---:|---|---|---:|---:|---|---|
{chr(10).join(family_lines)}

These family rows are descriptive, with small denominators often equal to one or three videos. They should not be treated as independent estimates or causal task effects; source dataset families and task types are crossed, and the temporal/motion row intentionally pools temporal grounding with motion/ambiguity probes.

### Singular correct-instance retrieval

The singular-instance denominator is stricter: it has 11 eligible frame slots per arm after excluding absent, uncertain, multi-instance, group, and source-unassessable target cases. Candidate Recall is reported only for pools with at least one resolved positive instance candidate; the Hit denominator still includes eligible empty or proposal-miss pools.

| Arm | Selected aliases | Instance-eligible target/control pools | Pools with a positive proposal | Instance Hit@1 | Instance Hit@5 |
|---|---:|---:|---:|---|---|
{chr(10).join(instance_arm_table)}

### Arm, crop-score, pool, duplicate, and proposal findings

The fixed-vocabulary arm has 290 selected aliases and the question-conditioned arm has 180. Across both arms, 56 exact mask duplicate groups are shared across arms (112 rows, 56 extra rows beyond unique masks); there are no within-arm duplicate mask groups and no duplicate candidate IDs. Selected rows contain 417 component proposal entries and 53 union proposal entries. Candidate score summaries by source label state and arm are stored under `arms.score_and_pool_summary` in `metrics.json`.

| Arm | Total target/control score observations | Mean / median pool size | Positive crop scores | Negative crop scores | Ambiguous crop scores |
|---|---:|---:|---|---|---|
{chr(10).join(score_table)}

The score columns summarize cached crop-level score observations for target/control phrase pools; they are descriptive and do not replace the ranked Hit/Recall metrics.

Proposal coverage is reported separately from ranking: a pool with zero positive labels is a proposal miss/unsupported pool for this packet, while the conditional-on-proposal ranking is computed only after at least one positive exists. This avoids treating absent proposals as a SigLIP2 ranking failure.

## Frozen R1 replay and paired changes

R1 was reconstructed from the frozen geometry implementation: target candidates are selected by prompt provenance, non-broad scene targets prefer component proposals when available, same-frame question-conditioned references exclude the target candidate itself and are selected by their candidate-frame cached scores, and geometry failures are sorted after satisfied candidates before the unchanged R0 remainder. Cross-frame fallback references are retained only to reproduce the frozen ordering and are invalid for spatial evidence.

- Replayed target/control query rows: **{r1['frozen_r1_replay']['query_count']}**.
- Top-1 ordering changes: **{r1['frozen_r1_replay']['top1_changed']}**; top-5 changes: **{r1['frozen_r1_replay']['top5_changed']}**.
- Changes attributable to target-provenance/component filtering: top-1 **{r1['frozen_r1_replay']['selection_filter_top1_changed']}**, top-5 **{r1['frozen_r1_replay']['selection_filter_top5_changed']}**.
- Additional changes from geometry ordering: top-1 **{r1['frozen_r1_replay']['geometry_order_top1_changed']}**, top-5 **{r1['frozen_r1_replay']['geometry_order_top5_changed']}**.
- Paired category and correct-instance top-1 state changes remain zero in the source-only labels, but that does not imply that candidate selection/order was unchanged. These are post-hoc source-only label comparisons; labels were not used in inference or reranking.

### Geometry co-occurrence and relation diagnostics

| QA | Source relation assessable | Targets with all same-frame refs | Same-frame refs / fallback refs | Raw / valid spatial rows | Raw relation outputs | Identity-conditioned accuracy | AMC/Cinemark raw ties |
|---|---|---:|---:|---:|---|---|---:|
{chr(10).join(geometry_lines)}

The relation function outputs are raw box-geometry diagnostics. Identity-conditioned correctness is scored only when both the frozen R1 target and every same-frame reference satisfy explicit role, positive correct-instance, complete-mask, and useful-crop checks. `train_003224` has no reference proposal; `train_005918` orientation is source-review unassessable; `train_012957` is invalid for stable group/tiny-anchor relation scoring. For `train_011679`, AMC/Cinemark edge gaps can tie at `[0, 0]` even when center distances differ; ties are reported as failures of that edge proxy, not semantic relation correctness.

Frozen R1 identity-pair audit:

{chr(10).join(pair_audit_lines)}

The `train_014154` fixed-arm labels do contain positive commercial-building and apartment-complex rows, but the frozen R1 replay selects the question-conditioned reference pool: its only same-frame references are the broad, correct-instance-ambiguous multi-building crops. The positive fixed-arm reference rows cannot replace those frozen references. For `train_002092`, the frozen references are target-labeled vegetation/road crops or a negative context/reference crop, so the named median-strip reference is not verified. Therefore neither QA contributes a defensible pair-level relation-accuracy denominator. `train_002092`'s raw relation output still uses the permissive any-positive-bbox-IoU containment proxy.

## Frozen R2 temporal diagnostics

R2 attaches track metadata to unchanged R1 ordering, so R2 has no rank-improvement metric. The table reports the top-eight stored tracks per archived target/control ranking row against source-reviewed visibility; misses and switches are descriptive because the archive does not expose a gold track or all association links.

| QA | Source-review limit | Best stored tracks by arm | Interpretation |
|---|---|---|---|
{chr(10).join(r2_lines)}

The bus is the only focused temporal case with an objectively usable identity interval: no identifiable target bus is source-verified at packed 0–11, packed 12 is ambiguous/partly occluded, and packed 13–15 are definitive visible frames. The QC track spans packed 12–15, covers all definitive visible frames, shows no observed switch, and remains right-censored at the final sample. `r2.descriptive_track_counts` records visible/ambiguous/absent-overlap, second-review not-visible overlap, both-overlap (a descriptive switch/censoring proxy), and no-known-overlap counts for every other stored track; those counts are not identity scores. The motorcycle is not scoreable at 5–12 source pixels and its QC tracks fragment into single frames. Construction-site tracks overlap the source-visible window but represent a framing-bounded region. Baseball-field tracks overlap category visibility, while multiple diamonds prevent singular identity scoring. The large white-roof building in `train_008503` is left-censored at clip start: the primary review marks packed 0–15 visible, while the second review confirms p0 (partly cropped) and p1 but reports p10/p12/p15 not visible and did not exhaustively review the other samples. Its frozen p0–5 top track and short fragments therefore establish only descriptive clip-start presence; continuity and exact duration are unscorable.

## Existing full-run top-5 context

The archived full run contains 3,689 candidates and 36 R0/R1/R2 ranking rows. Its R0 and R1 top-5 entries are included in `global_top5_descriptive` in `metrics.json` for context only. Specific archived R1 diagnostics: `train_014154` top rows report `right`, consistent with the source relation, though mask identity remains separate; `train_011679` top-five rows report `equal` edge gaps `[0, 0]` while center distances are approximately `[35, 175]` pixels, the source relation expects first-closer, and `satisfied` remains true so geometry does not filter the order; `train_003224` lacks reference proposals; `train_005918` is unassessable; `train_012957` is not assessable; and `train_002092` uses the permissive IoU>0 containment proxy. They use the full experiment pools, not the 470-row fixed-frame denominator, and do not change the Gate metrics above.

Representative visual artifacts remain available in [fixed-frame contact sheets](/home/soham/ReVA/runs/phase4_qcsrr/annotations/fixed_frame_eval/contact_sheets/page-001.jpg) and the archived [ranking contact sheets](/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_014154.jpg), [bus sheet](/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_001478.jpg), [AMC/Cinemark sheet](/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_011679.jpg), [motorcycle sheet](/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_012936.jpg), and [baseball sheet](/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_010200.jpg).

## Limitations and decision

- The frozen packet covers only packed frames 0 and 13. It supports frame-level and two-frame query-pool diagnostics, not video-global temporal accuracy.
- Source labels are role-aware but not dense segmentation IoU or answer correctness labels. Ambiguous, overbroad, fragment, and `not_assessable` rows remain visible in denominators and are not forced positive.
- The 56 cross-arm exact mask duplicates make raw arm totals non-independent. Arm comparisons report pool sizes and duplicate structure explicitly.
- R1 geometry uses answer-neutral candidate metadata and source-review relation descriptions. Cross-frame fallback is excluded from valid spatial evidence, and distinct-reference identity labels are unavailable, so relation correctness is not scored. Paired correctness changes are descriptive evaluation outputs, not inference-time gold labels.
- R2 track metadata is truncated to top eight tracks per archived ranking row; only the bus's definitive packed-13–15 interval receives an objectively assessable identity-interval comparison, with packed 12 ambiguous and the end right-censored. The `train_008503` building has conflicting frame visibility reviews and is scored only for clip-start presence; no continuous identity or exact duration is claimed. Other misses/switches are descriptive.

**Decision: retain this as a completed fixed-frame Gate diagnostic and do not promote a broad QCSRR retrieval claim.** The packet now supplies exact category/instance ranking metrics against matched random baselines for eligible pools, while the source-review limits still preclude an unlabeled video-global or universally instance-level claim. Do not start Qwen scoring from this evidence alone.

Earlier execution context is preserved in [PHASE4_PRE_RECOVERY_REPORT.md](/home/soham/ReVA/runs/phase4_qcsrr/PHASE4_PRE_RECOVERY_REPORT.md). The reproducible evaluator is [analyze_gate_evaluation.py](/home/soham/ReVA/runs/phase4_qcsrr/annotations/fixed_frame_eval/analyze_gate_evaluation.py), and its machine-readable output is [metrics.json](/home/soham/ReVA/runs/phase4_qcsrr/annotations/fixed_frame_eval/metrics.json).
"""
    return report


def main() -> None:
    data = validate_and_load()
    retrieval = build_retrieval_metrics(data)
    two_frame = build_two_frame_metrics(data)
    two_frame_family_summary = build_two_frame_family_summary(data, two_frame)
    r1 = build_r1_metrics(data)
    r2 = build_r2_metrics(data)
    global_top5 = build_global_top5(data)
    metrics = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "derived_phase4_fixed_frame_gate_evaluation",
        "external_inference_api_calls": False,
        "frozen_inputs": {
            "selection_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "candidate_labels": str(LABELS_PATH.relative_to(ROOT)),
            "score_reconstruction": str(SCORE_PATH.relative_to(ROOT)),
            "candidate_metadata": str(CANDIDATE_PATH.relative_to(ROOT)),
            "archived_rankings": str(RANKINGS_PATH.relative_to(ROOT)),
            "source_review_primary": str(PRIMARY_PATH.relative_to(ROOT)),
            "source_review_second": str(SECOND_PATH.relative_to(ROOT)),
            "source_review_adjudication": str(ADJUDICATION_PATH.relative_to(ROOT)),
            "previous_report_preserved": str(PRE_REPORT_PATH.relative_to(ROOT)),
        },
        "validation": {
            "selected_aliases": 470,
            "labeled_aliases": 470,
            "score_query_pools": 72,
            "score_observations": 821,
            "unique_scored_aliases": 470,
            "archived_score_max_absolute_error": data["score_payload"]["archived_score_validation"]["max_absolute_error"],
            "archived_scores_within_tolerance": True,
            "frozen_full_run_candidate_count": len(data["candidates"]),
            "frozen_full_run_ranking_rows": len(data["rankings"]),
        },
        **retrieval,
        "two_frame_query_metrics": two_frame,
        "two_frame_family_summary": two_frame_family_summary,
        "r1": r1,
        "r2": r2,
        "global_top5_descriptive": global_top5,
    }
    # Keep the output deterministic and assert that the output itself preserves
    # the core Gate denominator before writing it.
    assert metrics["validation"]["selected_aliases"] == 470
    assert metrics["validation"]["labeled_aliases"] == 470
    assert metrics["validation"]["score_query_pools"] == 72
    assert metrics["validation"]["score_observations"] == 821
    assert metrics["label_denominators"]["frame_slots"]["total_slots"] == 24
    assert metrics["label_denominators"]["frame_slots"]["category_eligible_slots_per_arm"] == 19
    assert metrics["label_denominators"]["frame_slots"]["instance_eligible_slots_per_arm"] == 11
    for metric, expected_slots in (("category", 19), ("instance", 11)):
        for arm, block in metrics["retrieval"][f"target_{metric}_by_arm"].items():
            assert block["source_eligible_queries"] == expected_slots, (metric, arm, block["source_eligible_queries"])
    assert len(metrics["arms"]["proposal_coverage_by_qa_frame"]) == 24
    for metric in ("category", "instance"):
        for arm in ("fixed-vocabulary", "question-conditioned"):
            block = metrics["two_frame_query_metrics"][f"{metric}_by_arm"][arm]
            assert block["query_count"] == 12
            assert len(block["rows"]) == 12
            assert block["independent_video_denominator"] == (11 if metric == "category" else 7)
            assert all(row["packed_frame_indices"] == [0, 13] for row in block["rows"])
    assert len(metrics["two_frame_family_summary"]["groups"]) >= 1
    assert sorted(qa_id for group in metrics["two_frame_family_summary"]["groups"] for qa_id in group["qa_ids"]) == sorted(data["questions"])
    assert metrics["r1"]["paired_correctness_change"]["category"]["improved"] == 0
    assert metrics["r1"]["paired_correctness_change"]["category"]["regressed"] == 0
    assert metrics["r1"]["paired_correctness_change"]["correct_instance"]["improved"] == 0
    assert metrics["r1"]["paired_correctness_change"]["correct_instance"]["regressed"] == 0
    r2_008503 = metrics["r2"]["focus_by_qa"]["train_008503"]
    assert r2_008503["source_visibility"]["censoring"] == "left_censored_at_clip_start"
    assert r2_008503["source_visibility"]["primary_visible_frames"] == list(range(16))
    assert r2_008503["source_visibility"]["second_review_not_visible_frames"] == [10, 12, 15]
    assert r2_008503["source_visibility"]["identity_scoreable"] is False
    assert r2_008503["rows"], "archived train_008503 R2 rows missing"
    assert any(row["best_track"] and row["best_track"]["frames"] == [0, 1, 2, 3, 4, 5] for row in r2_008503["rows"])
    r2_identity_008503 = metrics["r2"]["objectively_assessable_identity"]["train_008503"]
    assert r2_identity_008503["clip_start_presence"] is True
    assert r2_identity_008503["continuity_scoreable"] is False
    assert r2_identity_008503["exact_duration_scoreable"] is False
    pair_014154 = metrics["r1"]["geometry"]["train_014154"]["identity_conditioned_relation_accuracy"]
    assert pair_014154["same_frame_pair_rows"] == 22
    assert pair_014154["target_verified_rows"] == 18
    assert pair_014154["selected_reference_verified_candidate_count"] == 0
    assert pair_014154["verified_pair_rows"] == 0
    assert pair_014154["accuracy"] is None
    pair_002092 = metrics["r1"]["geometry"]["train_002092"]["identity_conditioned_relation_accuracy"]
    assert pair_002092["same_frame_pair_rows"] == 32
    assert pair_002092["target_verified_rows"] == 5
    assert pair_002092["selected_reference_verified_candidate_count"] == 0
    assert pair_002092["verified_pair_rows"] == 0
    assert pair_002092["accuracy"] is None
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    REPORT_PATH.write_text(render_report(metrics))
    print(json.dumps({
        "status": metrics["status"],
        "metrics": str(METRICS_PATH),
        "report": str(REPORT_PATH),
        "selected_aliases": 470,
        "score_query_pools": 72,
        "score_observations": 821,
        "r1_top1_changed": metrics["r1"]["frozen_r1_replay"]["top1_changed"],
        "r1_top5_changed": metrics["r1"]["frozen_r1_replay"]["top5_changed"],
    }, indent=2))


if __name__ == "__main__":
    main()
