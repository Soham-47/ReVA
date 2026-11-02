"""DSE-VTG-style similarity curves + MSF for ReVA multiple-choice TG."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import reze as R

LETTERS = R.LETTERS
DEFAULT_FPS = 3.0
DEFAULT_CLIP_SEC = 4.0
DEFAULT_MARGIN_GATE = 0.12
DEFAULT_FUSION_ALPHA = 1.0


def letter_margin(probs: list[float]) -> float:
    sorted_p = sorted(float(x) for x in probs)
    if len(sorted_p) < 2:
        return 1.0
    return sorted_p[-1] - sorted_p[-2]


def sample_frame_indices(*, fps: float, duration: float, total_frames: int, target_fps: float = DEFAULT_FPS) -> list[int]:
    if fps <= 0 or duration <= 0 or total_frames <= 0:
        raise ValueError("invalid video metadata for 3fps sampling")
    step = max(1, int(round(fps / target_fps)))
    last = max(0, total_frames - 1)
    indices = list(range(0, last + 1, step))
    if indices[-1] != last:
        indices.append(last)
    return indices


def msf_similarity(times: list[float], frame_sims: list[float], *, clip_sec: float = DEFAULT_CLIP_SEC) -> list[float]:
    """Multi-scale fusion: point similarity + local clip mean (DSE-VTG MSF idea)."""

    if len(times) != len(frame_sims) or not times:
        raise ValueError("times and frame_sims must be aligned and non-empty")
    sims = np.asarray(frame_sims, dtype=np.float64)
    fused = []
    for index, time_s in enumerate(times):
        window = [float(sims[i]) for i, t in enumerate(times) if abs(t - time_s) <= clip_sec / 2.0]
        clip_mean = float(np.mean(window)) if window else float(sims[index])
        fused.append(0.5 * float(sims[index]) + 0.5 * clip_mean)
    return fused


def query_tta_energies(
    row: dict[str, Any],
    times: list[float],
    sim_curves: list[list[float]],
    duration: float,
) -> list[float]:
    """Average option energy over query paraphrases (light Q-TTA)."""

    if not sim_curves:
        raise ValueError("sim_curves must be non-empty")
    acc = np.zeros(4, dtype=np.float64)
    for curve in sim_curves:
        energy = R.option_energy(row, times, curve, duration)
        acc += np.asarray(energy, dtype=np.float64)
    return (acc / len(sim_curves)).tolist()


def gated_fuse(
    letter_probs: list[float],
    energy: list[float],
    *,
    margin_gate: float = DEFAULT_MARGIN_GATE,
    alpha: float = DEFAULT_FUSION_ALPHA,
) -> tuple[list[float], dict[str, Any]]:
    margin = letter_margin(letter_probs)
    meta = {"letter_margin": margin, "fusion_applied": False, "alpha": alpha}
    if margin >= margin_gate:
        return list(letter_probs), meta
    meta["fusion_applied"] = True
    fused = R.fuse_probs(letter_probs, energy, alpha)
    return fused, meta


def normalize_similarities(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return [0.5] * len(values)
    return [float((x - lo) / (hi - lo)) for x in arr]
