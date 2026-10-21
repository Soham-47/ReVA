"""Strictly 2D relation text for the Phase 3 geometry pilot.

The scorer consumes this module's text beside the unchanged video prompt.  It
never turns image coordinates into depth, metric distance, or physical claims.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any


ENTITY_PHRASES = (
    "parking lot",
    "basketball court",
    "soccer field",
    "baseball field",
    "tennis court",
    "sports field",
    "swimming pool",
    "train track",
    "red barn",
    "industrial buildings",
    "building",
    "house",
    "bridge",
    "road",
    "river",
    "water",
    "boat",
    "boats",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "train",
    "field",
    "court",
    "pool",
    "barn",
    "path",
    "vehicle",
    "vehicles",
    "person",
    "tree",
)

_COORD_KEYS = ("left", "top", "right", "bottom")
_COLOR_PREFIX = r"(?:red|blue|black|white|yellow|green|gray|grey|large|small)"


def relation_subtype(question: str) -> str:
    """Classify wording without inspecting answers or gold labels."""

    text = str(question).casefold()
    if any(
        word in text
        for word in (
            "distance",
            "closer",
            "farther",
            "nearer",
            "behind",
            "in front",
            "front of",
            "depth",
            "perspective",
            "viewpoint",
            "point of view",
            "foreground",
            "background",
        )
    ):
        return "depth_or_distance"
    if any(word in text for word in ("overlap", "inside", "contain", "surround", "within", "enclose")):
        return "overlap_containment"
    if any(word in text for word in ("left", "right", "above", "below", "top", "bottom")):
        return "image_plane_direction"
    if any(word in text for word in ("parallel", "perpendicular", "orientation", "aligned", "angle")):
        return "orientation"
    return "other_2d_or_unsupported"


def question_entities(question: str, *, limit: int = 6) -> list[str]:
    """Extract noun phrases from the question only for detector/SAM prompts."""

    text = str(question).casefold()
    found: list[tuple[int, str]] = []
    colored_spans: list[tuple[int, int]] = []
    colored_phrases: set[str] = set()
    for phrase in ENTITY_PHRASES:
        position = text.find(phrase)
        if position >= 0:
            found.append((position, phrase))
    # Preserve color + noun strings where the question names one explicitly.
    for match in re.finditer(
        rf"\b({_COLOR_PREFIX})\s+((?:[a-z]+\s+){{0,2}}(?:car|truck|bus|boat|field|court|barn|building|house|pool|vehicle))\b",
        text,
    ):
        phrase = " ".join(match.group(0).split())
        found.append((match.start(), phrase))
        colored_spans.append((match.start(), match.end()))
        colored_phrases.add(phrase)
    result: list[str] = []
    for _, phrase in sorted(found, key=lambda item: (item[0], item[1])):
        if phrase not in colored_phrases and any(start <= text.find(phrase) < end for start, end in colored_spans):
            continue
        if any(phrase != prior and re.search(rf"\b{re.escape(phrase)}\b", prior) for prior in result + list(colored_phrases)):
            continue
        if phrase not in result:
            result.append(phrase)
        if len(result) == limit:
            break
    return result


def _box(value: Mapping[str, Any]) -> dict[str, float] | None:
    raw = value.get("box", value)
    if not isinstance(raw, Mapping) or any(key not in raw for key in _COORD_KEYS):
        return None
    try:
        result = {key: float(raw[key]) for key in _COORD_KEYS}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in result.values()):
        return None
    if not (0 <= result["left"] <= result["right"] <= 1 and 0 <= result["top"] <= result["bottom"] <= 1):
        return None
    return result


def _area(box: Mapping[str, float]) -> float:
    return max(0.0, float(box["right"]) - float(box["left"])) * max(
        0.0, float(box["bottom"]) - float(box["top"])
    )


def _iou(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    intersection = {
        "left": max(left["left"], right["left"]),
        "top": max(left["top"], right["top"]),
        "right": min(left["right"], right["right"]),
        "bottom": min(left["bottom"], right["bottom"]),
    }
    overlap = _area(intersection)
    union = _area(left) + _area(right) - overlap
    return overlap / union if union else 0.0


def _centroid(box: Mapping[str, float]) -> tuple[float, float]:
    return ((box["left"] + box["right"]) / 2, (box["top"] + box["bottom"]) / 2)


def image_plane_relations(left: Mapping[str, Any], right: Mapping[str, Any], *, margin: float = 0.05) -> list[str]:
    """Return conservative 2D relations for two same-frame boxes."""

    if margin < 0:
        raise ValueError("margin must be non-negative")
    left_box = _box(left)
    right_box = _box(right)
    if left_box is None or right_box is None:
        return []
    left_x, left_y = _centroid(left_box)
    right_x, right_y = _centroid(right_box)
    relations: list[str] = []
    if _iou(left_box, right_box) >= 0.10:
        relations.append("overlap")
    if left_box["left"] <= right_box["left"] and left_box["top"] <= right_box["top"] and left_box["right"] >= right_box["right"] and left_box["bottom"] >= right_box["bottom"]:
        relations.append("contains")
    elif right_box["left"] <= left_box["left"] and right_box["top"] <= left_box["top"] and right_box["right"] >= left_box["right"] and right_box["bottom"] >= left_box["bottom"]:
        relations.append("inside")
    if left_x + margin < right_x:
        relations.append("left of")
    elif right_x + margin < left_x:
        relations.append("right of")
    if left_y + margin < right_y:
        relations.append("above")
    elif right_y + margin < left_y:
        relations.append("below")
    return relations


def _records_by_frame(records: Iterable[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        try:
            frame = int(record["frame_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if _box(record) is not None:
            by_frame.setdefault(frame, []).append(record)
    return by_frame


def relation_context(
    records: Iterable[Mapping[str, Any]],
    entities: Iterable[str],
    *,
    max_chars: int = 1600,
    min_common_frames: int = 1,
) -> tuple[str, dict[str, Any]]:
    """Build bounded, auditable text from same-frame 2D observations.

    ``records`` may contain detector observations or SAM 3 mask boxes.  Each
    record must have ``label``, ``frame_index`` and normalized ``box`` fields.
    """

    if max_chars <= 0 or min_common_frames <= 0:
        raise ValueError("max_chars and min_common_frames must be positive")
    clean_entities = list(dict.fromkeys(str(entity).strip().casefold() for entity in entities if str(entity).strip()))
    by_label: dict[str, list[Mapping[str, Any]]] = {entity: [] for entity in clean_entities}
    for record in records:
        label = str(record.get("label", "")).strip().casefold()
        if label in by_label:
            by_label[label].append(record)
    lines = [
        "Explicit geometry context: uncertain normalized [0,1] 2D image-plane evidence only;",
        "no metric distance, physical depth, or 3D relation is inferred.",
    ]
    evidence: list[dict[str, Any]] = []
    for index, left_label in enumerate(clean_entities):
        for right_label in clean_entities[index + 1 :]:
            left_frames = _records_by_frame(by_label[left_label])
            right_frames = _records_by_frame(by_label[right_label])
            common = sorted(set(left_frames) & set(right_frames))
            pair_relations: list[tuple[int, list[str]]] = []
            for frame in common:
                # One best candidate per label/frame keeps context bounded.
                left = max(left_frames[frame], key=lambda row: float(row.get("confidence", 0.0)))
                right = max(right_frames[frame], key=lambda row: float(row.get("confidence", 0.0)))
                rel = image_plane_relations(left, right)
                if rel:
                    pair_relations.append((frame, rel))
            if len(common) < min_common_frames or not pair_relations:
                lines.append(f"{left_label} / {right_label}: insufficient reliable same-frame evidence; fall back to baseline.")
                evidence.append({"left": left_label, "right": right_label, "status": "insufficient", "common_frames": len(common)})
                continue
            relation_counts: dict[str, int] = {}
            for _, relations in pair_relations:
                for relation in relations:
                    relation_counts[relation] = relation_counts.get(relation, 0) + 1
            chosen = [relation for relation, count in sorted(relation_counts.items()) if count * 2 >= len(pair_relations)]
            if not chosen:
                lines.append(f"{left_label} / {right_label}: ambiguous 2D relation; fall back to baseline.")
                status = "ambiguous"
            else:
                lines.append(f"{left_label} is {', '.join(chosen)} {right_label} in {len(pair_relations)} sampled frame(s).")
                status = "reliable_2d"
            evidence.append({"left": left_label, "right": right_label, "status": status, "common_frames": len(common), "relations": chosen})
    if len(clean_entities) < 2:
        lines.append("Fewer than two named entities were available; fall back to baseline.")
    text = "\n".join(lines)[:max_chars]
    return text, {"entities": clean_entities, "pairs": evidence, "context_chars": len(text)}


def _self_check() -> None:
    assert relation_subtype("Is the court above the road?") == "image_plane_direction"
    assert (
        relation_subtype(
            "From the camera's perspective, how is the row of green trees positioned "
            "relative to the buildings on the left side of the street?"
        )
        == "depth_or_distance"
    )
    assert question_entities("Is the red car left of the bridge?") == ["red car", "bridge"]
    left = {"box": {"left": 0.1, "top": 0.1, "right": 0.3, "bottom": 0.3}}
    right = {"box": {"left": 0.6, "top": 0.5, "right": 0.8, "bottom": 0.7}}
    assert image_plane_relations(left, right) == ["left of", "above"]
    text, audit = relation_context(
        [
            {"label": "car", "frame_index": 1, "confidence": 0.8, **left},
            {"label": "bridge", "frame_index": 1, "confidence": 0.9, **right},
        ],
        ["car", "bridge"],
    )
    assert "2D image-plane" in text and audit["pairs"][0]["status"] == "reliable_2d"
    assert "depth" in text and "3D" in text


if __name__ == "__main__":
    _self_check()
    print("geometry_relations self-check passed")
