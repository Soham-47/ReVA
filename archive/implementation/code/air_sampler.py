"""Deterministic A.I.R.-style frame selection for ReVA.

The public A.I.R. implementation targets other video backbones and decoders.
This module ports its selector stages without importing any GPU dependency:
query-conditioned event segmentation, density-aware allocation, and a bounded
VLM refinement hook. The answer model remains owned by the ReVA scorer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

AIR_VERSION = "reva-air-v1"
AIR_REFERENCE = "UCF-AIR/A.I.R.@400fb52f1cfbf64debc6408cd215cfa375fd9797 (retrieved 2026-10-30); ReVA selector port with SigLIP2 query signal"
AIR_SIMILARITY_METHOD = "siglip2_query_similarity_adaptation"


def candidate_policy(count: int) -> dict[str, Any]:
    """Return the exact AIR candidate-grid identity used by feature caches."""

    value = int(count)
    if value < 2:
        raise ValueError("AIR candidate count must be at least 2")
    return {
        "cap": value,
        "count": value,
        "grid_algorithm": "uniform-endpoints",
        "grid_version": "air-candidate-grid-v1",
    }


@dataclass(frozen=True)
class AIRConfig:
    """Frozen short-clip-safe selector settings."""

    candidate_count: int = 64
    final_budget: int = 16
    initial_budget: int | None = None
    max_iterations: int = 3
    vlm_batch_size: int = 6
    vlm_keep_threshold: float = 0.5
    gmm_coefficient: float = 0.5
    merge_gap_fraction: float = 1.5
    min_event_fraction: float = 0.0
    local_radius_fraction: float = 1.5
    max_interval_candidates: int = 8
    seed: int = 42
    feature_encoder_revision: str = "google/siglip2-base-patch16-224@022b6f71160ffb0169ca4709e2d7e25be659598a"
    vlm_model_revision: str = "Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

    def __post_init__(self) -> None:
        if self.candidate_count < 2 or self.final_budget < 1:
            raise ValueError("candidate_count must be >= 2 and final_budget positive")
        if self.initial_budget is not None and self.initial_budget < 1:
            raise ValueError("initial_budget must be positive")
        if self.max_iterations < 0 or self.vlm_batch_size < 1:
            raise ValueError("max_iterations must be non-negative and vlm_batch_size positive")
        if not 0.0 <= self.vlm_keep_threshold <= 1.0:
            raise ValueError("vlm_keep_threshold must be in [0, 1]")
        if self.gmm_coefficient < 0 or self.merge_gap_fraction < 0 or self.local_radius_fraction < 0:
            raise ValueError("relative selector settings must be non-negative")
        if not 0.0 <= self.min_event_fraction <= 1.0:
            raise ValueError("min_event_fraction must be in [0, 1]")

    @property
    def initial_target(self) -> int:
        return int(self.initial_budget or min(self.final_budget, max(4, self.final_budget // 2)))


@dataclass(frozen=True)
class GMMThreshold:
    threshold: float
    means: tuple[float, float]
    stds: tuple[float, float]
    weights: tuple[float, float]
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "means": list(self.means), "stds": list(self.stds), "weights": list(self.weights), "method": self.method}


@dataclass
class AIRSelection:
    frame_indices: list[int]
    timestamps_s: list[float]
    events: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    stop_reason: str
    raw_selected_count: int
    gmm: GMMThreshold
    config: AIRConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": AIR_VERSION,
            "reference": AIR_REFERENCE,
            "similarity_method": AIR_SIMILARITY_METHOD,
            "frame_indices": list(self.frame_indices),
            "frame_timestamps_s": list(self.timestamps_s),
            "events": self.events,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "raw_selected_count": int(self.raw_selected_count),
            "gmm": self.gmm.to_dict(),
            "config": config_to_dict(self.config),
        }


def config_to_dict(config: AIRConfig) -> dict[str, Any]:
    return {
        "candidate_count": config.candidate_count,
        "final_budget": config.final_budget,
        "initial_budget": config.initial_budget,
        "max_iterations": config.max_iterations,
        "vlm_batch_size": config.vlm_batch_size,
        "vlm_keep_threshold": config.vlm_keep_threshold,
        "gmm_coefficient": config.gmm_coefficient,
        "merge_gap_fraction": config.merge_gap_fraction,
        "min_event_fraction": config.min_event_fraction,
        "local_radius_fraction": config.local_radius_fraction,
        "max_interval_candidates": config.max_interval_candidates,
        "seed": config.seed,
        "feature_encoder_revision": config.feature_encoder_revision,
        "vlm_model_revision": config.vlm_model_revision,
    }


def candidate_grid(n_frames: int, count: int, fps: float) -> list[dict[str, Any]]:
    """Create a unique endpoint-inclusive source-frame grid."""
    n = max(int(n_frames), 1)
    target = max(1, min(int(count), n))
    if target == 1:
        indices = [0]
    else:
        indices = [round(i * (n - 1) / (target - 1)) for i in range(target)]
    unique: list[int] = []
    for index in indices:
        index = min(n - 1, max(0, int(index)))
        if not unique or index != unique[-1]:
            unique.append(index)
    rate = max(float(fps), 1e-9)
    return [{"frame_index": index, "timestamp": index / rate} for index in unique]


def _finite_scores(scores: Sequence[float]) -> list[float]:
    result = []
    for value in scores:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        result.append(number if math.isfinite(number) else 0.0)
    return result


def _normalise(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def fit_two_component_gmm(scores: Sequence[float], coefficient: float = 0.5) -> GMMThreshold:
    """Fit the GMM and use ReVA's documented short-clip threshold adaptation.

    The paper's exact threshold convention is not asserted here: ReVA uses
    the more conservative maximum of the component midpoint and the
    high-component-minus-coefficient-times-high-std rule, then clips to the
    observed score range.
    """
    values = _finite_scores(scores)
    if not values:
        raise ValueError("at least one similarity score is required")
    if len(values) < 2 or max(values) - min(values) <= 1e-12:
        mean = float(values[0])
        return GMMThreshold(mean, (mean, mean), (0.0, 0.0), (1.0, 0.0), "degenerate")
    ordered = sorted(values)
    means = [ordered[len(ordered) // 3], ordered[(2 * len(ordered)) // 3]]
    variance = max(sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values), 1e-6)
    variances = [variance, variance]
    weights = [0.5, 0.5]
    for _ in range(32):
        responsibilities: list[tuple[float, float]] = []
        for value in values:
            likelihoods = [
                weight * math.exp(-0.5 * (value - mean) ** 2 / max(var, 1e-9)) / math.sqrt(max(var, 1e-9))
                for mean, var, weight in zip(means, variances, weights)
            ]
            total = sum(likelihoods)
            responsibilities.append((likelihoods[0] / total, likelihoods[1] / total) if total > 0 else (0.5, 0.5))
        for component in (0, 1):
            mass = sum(item[component] for item in responsibilities)
            if mass <= 1e-9:
                continue
            means[component] = sum(resp[component] * value for resp, value in zip(responsibilities, values)) / mass
            variances[component] = max(sum(resp[component] * (value - means[component]) ** 2 for resp, value in zip(responsibilities, values)) / mass, 1e-9)
            weights[component] = mass / len(values)
    order = sorted(range(2), key=lambda index: means[index])
    means = [means[index] for index in order]
    stds = [math.sqrt(variances[index]) for index in order]
    weights = [weights[index] for index in order]
    threshold = max(means[0] + 0.5 * (means[1] - means[0]), means[1] - coefficient * stds[1])
    threshold = min(max(threshold, min(values)), max(values))
    return GMMThreshold(float(threshold), tuple(map(float, means)), tuple(map(float, stds)), tuple(map(float, weights)), "em_reva_threshold_adaptation")


def _candidate_values(candidates: Sequence[Mapping[str, Any] | int], fps: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    rate = max(float(fps), 1e-9)
    for value in candidates:
        if isinstance(value, Mapping):
            index = int(value.get("frame_index", value.get("index", -1)))
            timestamp = float(value.get("timestamp", value.get("timestamp_s", index / rate)))
            record = dict(value)
        else:
            index = int(value)
            timestamp = index / rate
            record = {}
        if index < 0:
            raise ValueError("candidate frame indices must be non-negative")
        record.update({"frame_index": index, "timestamp": timestamp})
        result.append(record)
    if not result:
        raise ValueError("at least one candidate is required")
    indices = [item["frame_index"] for item in result]
    if indices != sorted(set(indices)):
        raise ValueError("candidate frame indices must be unique and sorted")
    return result


def find_events(candidates: Sequence[Mapping[str, Any] | int], similarities: Sequence[float], threshold: GMMThreshold | float, *, merge_gap_fraction: float = 1.5, min_event_fraction: float = 0.0, fps: float = 1.0) -> list[dict[str, Any]]:
    """Segment high-similarity candidates and merge only local short gaps."""
    items = _candidate_values(candidates, fps)
    values = _finite_scores(similarities)
    if len(items) != len(values):
        raise ValueError("candidate and similarity lengths differ")
    cutoff = float(threshold.threshold if isinstance(threshold, GMMThreshold) else threshold)
    above = [index for index, value in enumerate(values) if value >= cutoff]
    if not above:
        return []
    events: list[list[int]] = [[above[0]]]
    gaps = [items[i + 1]["timestamp"] - items[i]["timestamp"] for i in range(len(items) - 1)]
    positive_gaps = [gap for gap in gaps if gap > 0]
    typical_gap = sorted(positive_gaps)[len(positive_gaps) // 2] if positive_gaps else 1.0 / max(fps, 1e-9)
    max_gap = typical_gap * max(0.0, float(merge_gap_fraction))
    for index in above[1:]:
        previous = events[-1][-1]
        if items[index]["timestamp"] - items[previous]["timestamp"] <= max_gap + 1e-12:
            events[-1].append(index)
        else:
            events.append([index])
    min_size = max(1, int(math.ceil(len(items) * max(0.0, min_event_fraction))))
    result = []
    for positions in events:
        if len(positions) < min_size:
            continue
        frame_indices = [items[position]["frame_index"] for position in positions]
        event_values = [values[position] for position in positions]
        result.append({
            "candidate_positions": positions,
            "frame_indices": frame_indices,
            "start_frame": frame_indices[0],
            "end_frame": frame_indices[-1],
            "start_time": items[positions[0]]["timestamp"],
            "end_time": items[positions[-1]]["timestamp"],
            "duration": max(0.0, items[positions[-1]]["timestamp"] - items[positions[0]]["timestamp"]),
            "mean_similarity": sum(event_values) / len(event_values),
            "max_similarity": max(event_values),
            "candidate_count": len(positions),
        })
    return result


def allocate_event_frames(events: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any] | int], similarities: Sequence[float], budget: int, *, fps: float = 1.0) -> list[int]:
    """Allocate an exact initial budget proportionally, with stable ties."""
    items = _candidate_values(candidates, fps)
    target = max(1, min(int(budget), len(items)))
    if not events:
        ranking = sorted(range(len(items)), key=lambda index: (-float(similarities[index]), items[index]["frame_index"]))
        return [items[index]["frame_index"] for index in ranking[:target]]
    ordered = sorted(events, key=lambda event: (event["start_frame"], event["end_frame"]))
    chosen_events = list(ordered)
    if target < len(chosen_events):
        chosen_events.sort(key=lambda event: (-float(event.get("max_similarity", 0.0)), event["start_frame"]))
        chosen_events = sorted(chosen_events[:target], key=lambda event: event["start_frame"])
    counts = [1 for _ in chosen_events]
    remaining = target - len(counts)
    weights = [max(1e-9, float(event.get("duration", 0.0)) + 1.0 / max(fps, 1e-9)) * (0.5 + max(0.0, float(event.get("mean_similarity", 0.0)))) for event in chosen_events]
    while remaining > 0:
        eligible = [index for index, event in enumerate(chosen_events) if counts[index] < len(event["candidate_positions"])]
        if not eligible:
            break
        pick = max(eligible, key=lambda index: (weights[index] / counts[index], -chosen_events[index]["start_frame"]))
        counts[pick] += 1
        remaining -= 1
    selected: list[int] = []
    for event, count in zip(chosen_events, counts):
        positions = list(event["candidate_positions"])
        if count >= len(positions):
            selected.extend(items[position]["frame_index"] for position in positions)
        else:
            for offset in range(count):
                position = positions[round(offset * (len(positions) - 1) / max(count - 1, 1))]
                selected.append(items[position]["frame_index"])
    return sorted(set(selected))[:target]


def parse_vlm_relevance(response: Any) -> dict[int, float]:
    """Parse tolerant JSON/text VLM output into ``candidate index -> score``."""
    if isinstance(response, Mapping):
        payload: Any = response
    elif isinstance(response, (list, tuple)):
        payload = response
    elif isinstance(response, str):
        try:
            payload = json.loads(response.strip())
        except json.JSONDecodeError:
            matches = re.findall(r"(?:frame\s*)?(\d+)\s*[:=,-]\s*(0(?:\.\d+)?|1(?:\.0+)?)", response, re.I)
            return {int(index): min(1.0, max(0.0, float(score))) for index, score in matches}
    else:
        payload = None
    if isinstance(payload, Mapping):
        iterable = payload.items()
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        iterable = enumerate(payload)
    else:
        return {}
    parsed: dict[int, float] = {}
    for key, value in iterable:
        if isinstance(value, Mapping):
            value = value.get("relevance", value.get("score", value.get("keep")))
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        try:
            index, score = int(key), float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            parsed[index] = min(1.0, max(0.0, score))
    return parsed


def _vlm_scores(
    callback: Callable[[list[dict[str, Any]]], Any] | None,
    batch: list[dict[str, Any]],
    fallback: Mapping[int, float],
) -> tuple[dict[int, float], set[int]]:
    """Return scores plus the indices that have an actual measured score.

    Original-video neighbors are deliberately not assigned a copied cache
    score.  They remain pending until the VLM callback measures them; this
    prevents synthetic relevance from entering the final ranking.
    """
    if callback is None:
        scores = {int(item["frame_index"]): float(fallback.get(int(item["frame_index"]), 0.0)) for item in batch}
        return scores, {index for index in scores if index in fallback}
    try:
        parsed = parse_vlm_relevance(callback(batch))
    except Exception:
        parsed = {}
    scores = {}
    for item in batch:
        index = int(item["frame_index"])
        scores[index] = float(parsed.get(index, fallback.get(index, 0.0)))
    measured = {index for index in parsed if index in scores}
    measured.update(index for index in scores if index in fallback)
    return scores, measured


def select_air_frames(candidates: Sequence[Mapping[str, Any] | int], similarities: Sequence[float], *, fps: float, budget: int, config: AIRConfig | None = None, vlm_score: Callable[[list[dict[str, Any]]], Any] | None = None, temporal_offset: float = 0.0, neighbor_provider: Callable[[dict[str, Any]], Sequence[Mapping[str, Any]]] | None = None) -> AIRSelection:
    """Run deterministic AIR stages and return an exact-budget selection."""
    config = config or AIRConfig(final_budget=budget)
    if not 0.0 <= float(temporal_offset) < 1.0:
        raise ValueError("temporal_offset must be in [0, 1)")
    if config.final_budget != int(budget):
        config = AIRConfig(**{**config_to_dict(config), "final_budget": int(budget)})
    items = _candidate_values(candidates, fps)
    candidate_items = list(items)
    values = _normalise(_finite_scores(similarities))
    if len(items) != len(values):
        raise ValueError("candidate and similarity lengths differ")
    threshold = fit_two_component_gmm(values, config.gmm_coefficient)
    events = find_events(items, values, threshold, merge_gap_fraction=config.merge_gap_fraction, min_event_fraction=config.min_event_fraction, fps=fps)
    steps: list[dict[str, Any]] = [{"stage": "gmm_event_segmentation", "candidate_count": len(items), "threshold": threshold.threshold, "temporal_offset": float(temporal_offset), "events": events}]
    fallback_scores = {int(item["frame_index"]): float(score) for item, score in zip(items, values)}
    initial = allocate_event_frames(events, items, values, min(config.initial_target, config.final_budget), fps=fps)
    selected = set(initial)
    steps.append({"stage": "density_sampling", "target": config.initial_target, "selected_indices": sorted(selected)})
    for iteration in range(config.max_iterations):
        if len(selected) >= config.final_budget:
            break
        unexplored = [item for item in items if int(item["frame_index"]) not in selected]
        if not unexplored:
            break
        # Rank bounded intervals, not isolated synthetic neighbors.  The
        # interval score is an explicit ReVA adaptation of AIR's potential:
        # max relevance + .25*signal range + .05*temporal span.  Intervals are
        # bounded by selected/endpoint candidates and aggregate all measured
        # candidate points between those boundaries.
        ordered = sorted(
            (item for item in items if int(item["frame_index"]) in fallback_scores),
            key=lambda item: int(item["frame_index"]),
        )
        positions = {int(item["frame_index"]): index for index, item in enumerate(ordered)}
        potentials: dict[int, float] = {}
        interval_records: dict[str, dict[str, Any]] = {}
        selected_positions = {positions[index] for index in selected if index in positions}
        boundary_positions = {0, max(0, len(ordered) - 1), *selected_positions}
        boundaries = sorted(boundary_positions)
        for left_position, right_position in zip(boundaries, boundaries[1:]):
            members = [
                item
                for item in ordered[left_position : right_position + 1]
                if int(item["frame_index"]) not in selected
            ]
            if not members:
                continue
            member_scores = [fallback_scores[int(item["frame_index"])] for item in members]
            left_item, right_item = ordered[left_position], ordered[right_position]
            potential = max(member_scores) + 0.25 * (max(member_scores) - min(member_scores)) + 0.05 * max(
                0.0, float(right_item["timestamp"]) - float(left_item["timestamp"])
            )
            interval_key = f"{int(left_item['frame_index'])}:{int(right_item['frame_index'])}"
            interval_records[interval_key] = {
                "start_frame": int(left_item["frame_index"]),
                "end_frame": int(right_item["frame_index"]),
                "potential": potential,
                "candidate_indices": [int(item["frame_index"]) for item in members],
            }
            for item in members:
                potentials[int(item["frame_index"])] = potential
        known_unexplored = [item for item in unexplored if int(item["frame_index"]) in fallback_scores]
        pending_neighbors = [item for item in unexplored if int(item["frame_index"]) not in fallback_scores]
        ranked = sorted(
            known_unexplored,
            key=lambda item: (
                -potentials.get(int(item["frame_index"]), fallback_scores[int(item["frame_index"])]),
                (int(item["frame_index"]) - round(float(temporal_offset) * len(items))) % len(items),
            ),
        )
        pending_neighbors.sort(key=lambda item: int(item["frame_index"]))
        batch = (pending_neighbors + ranked)[: min(config.max_interval_candidates, config.vlm_batch_size, config.final_budget - len(selected))]
        scored, measured = _vlm_scores(vlm_score, batch, fallback_scores)
        for index in measured:
            fallback_scores[index] = scored[index]
        passing = [
            item
            for item in batch
            if int(item["frame_index"]) in measured
            and scored[int(item["frame_index"])] >= config.vlm_keep_threshold
        ]
        selected.update(int(item["frame_index"]) for item in passing)
        added_neighbors: list[dict[str, Any]] = []
        if neighbor_provider is not None:
            known = {int(item["frame_index"]) for item in items}
            for item in passing:
                try:
                    neighbors = neighbor_provider(item)
                except Exception:
                    neighbors = []
                for neighbor in neighbors:
                    if not isinstance(neighbor, Mapping):
                        continue
                    index = int(neighbor.get("frame_index", -1))
                    if index < 0 or index in known:
                        continue
                    record = dict(neighbor)
                    record.setdefault("timestamp", float(index) / max(fps, 1e-9))
                    record["source"] = "original_video_neighbor"
                    record["measured_by"] = "pending_vlm"
                    record.pop("similarity", None)
                    items.append(record)
                    known.add(index)
                    added_neighbors.append(record)
            if added_neighbors:
                items.sort(key=lambda item: int(item["frame_index"]))
        if passing and len(selected) < config.final_budget:
            timestamps = [float(item["timestamp"]) for item in items]
            gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1) if timestamps[i + 1] > timestamps[i]]
            radius = (sorted(gaps)[len(gaps) // 2] if gaps else 1.0 / max(fps, 1e-9)) * config.local_radius_fraction
            for item in known_unexplored:
                if len(selected) >= config.final_budget:
                    break
                if int(item["frame_index"]) not in fallback_scores:
                    continue
                if any(abs(float(item["timestamp"]) - float(pass_item["timestamp"])) <= radius + 1e-12 for pass_item in passing):
                    selected.add(int(item["frame_index"]))
        steps.append({"stage": "vlm_refinement", "iteration": iteration, "candidate_indices": [int(item["frame_index"]) for item in batch], "interval_potentials": interval_records, "vlm_scores": {str(index): score for index, score in scored.items()}, "measured_indices": sorted(measured), "passing_indices": [int(item["frame_index"]) for item in passing], "neighbor_indices": [int(item["frame_index"]) for item in added_neighbors], "selected_indices": sorted(selected)})
    raw_selected_count = len(selected)
    measured_items = [item for item in items if int(item["frame_index"]) in fallback_scores]
    ranking = sorted(measured_items, key=lambda item: (-fallback_scores[int(item["frame_index"])], (int(item["frame_index"]) - round(float(temporal_offset) * len(items))) % len(items)))
    for item in ranking:
        if len(selected) >= config.final_budget:
            break
        selected.add(int(item["frame_index"]))
    endpoint_indices = {int(candidate_items[0]["frame_index"]), int(candidate_items[-1]["frame_index"])} if len(candidate_items) >= 2 else {int(candidate_items[0]["frame_index"])}
    if config.final_budget >= 2 and len(candidate_items) >= 2:
        selected.update(endpoint_indices)
    if len(selected) > config.final_budget:
        protected = endpoint_indices if config.final_budget >= 2 else set()
        remaining = sorted(selected - protected, key=lambda index: (-fallback_scores.get(index, 0.0), index))
        selected = protected | set(remaining[: max(0, config.final_budget - len(protected))])
    target = min(config.final_budget, len(candidate_items))
    for item in measured_items:
        if len(selected) >= target:
            break
        selected.add(int(item["frame_index"]))
    final_indices = sorted(selected)
    steps.append({"stage": "exact_budget_fallback", "raw_selected_count": raw_selected_count, "final_indices": final_indices})
    by_index = {int(item["frame_index"]): item for item in items}
    return AIRSelection(frame_indices=final_indices, timestamps_s=[float(by_index[index]["timestamp"]) for index in final_indices], events=events, steps=steps, stop_reason="exact_budget" if raw_selected_count != len(final_indices) else "adaptive_budget", raw_selected_count=raw_selected_count, gmm=threshold, config=config)


__all__ = ["AIR_VERSION", "AIR_REFERENCE", "AIR_SIMILARITY_METHOD", "AIRConfig", "AIRSelection", "GMMThreshold", "allocate_event_frames", "candidate_grid", "candidate_policy", "config_to_dict", "find_events", "fit_two_component_gmm", "parse_vlm_relevance", "select_air_frames"]
