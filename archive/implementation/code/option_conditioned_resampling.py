"""Option-conditioned temporal window resampling for ReVA VideoQA.

Pass 1 evaluates the video with standard uniform 16-frame sampling.
For questions with numeric time options (e.g., Temporal Grounding), Pass 2
densely resamples 16 frames spanning the option window [t_min - margin, t_max + margin].
This reduces boundary discretization error from ~0.4s down to ~0.1s.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np

# Regular expressions for explicit timestamps and clock formats (mm:ss or m:ss)
TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\b", re.IGNORECASE)
CLOCK_RE = re.compile(r"\b(\d{1,2}):(\d{2}(?:\.\d+)?)\b")


def parse_numeric_timestamps(text: str) -> list[float]:
    """Extract all explicit timestamps in seconds from option text."""
    if not isinstance(text, str) or not text.strip():
        return []
    timestamps: list[float] = []
    # Match seconds: e.g. "3.5s", "4.0 seconds"
    for m in TIME_RE.finditer(text):
        try:
            val = float(m.group(1))
            if math.isfinite(val) and val >= 0:
                timestamps.append(val)
        except (ValueError, TypeError):
            continue
    # Match clock format: e.g. "01:23", "0:08"
    for m in CLOCK_RE.finditer(text):
        try:
            mins = float(m.group(1))
            secs = float(m.group(2))
            val = mins * 60.0 + secs
            if math.isfinite(val) and val >= 0:
                timestamps.append(val)
        except (ValueError, TypeError):
            continue
    return timestamps


def extract_option_time_window(
    options: dict[str, str],
    duration: float,
    margin_s: float = 0.8,
    min_span_s: float = 1.5,
) -> tuple[float, float] | None:
    """Extract the bounding time interval [t_start, t_end] from multiple-choice options.

    Returns None if fewer than 2 distinct timestamps are found, or if the
    extracted window already covers >= 90% of the clip duration (no zoom benefit).
    """
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return None

    all_times: list[float] = []
    for opt_text in options.values():
        all_times.extend(parse_numeric_timestamps(opt_text))

    if len(all_times) < 2:
        return None

    t_min = min(all_times)
    t_max = max(all_times)
    if t_max <= t_min:
        return None

    # Left-censoring safeguard: if t_min is near start, anchor window strictly at t=0
    if t_min <= 1.2:
        t_start = 0.0
    else:
        t_start = max(0.0, t_min - float(margin_s))

    t_end = min(float(duration), t_max + float(margin_s))

    # Ensure minimum window duration
    span = t_end - t_start
    if span < min_span_s:
        t_end = min(float(duration), t_start + min_span_s)
        span = t_end - t_start

    # If the window spans nearly the entire video, uniform sampling is already equivalent
    if span >= 0.88 * duration:
        return None

    return (t_start, t_end)


def sample_window_indices(
    n_frames: int,
    fps: float,
    window: tuple[float, float],
    offset: float = 0.0,
    k: int = 16,
) -> list[int]:
    """Sample k frames uniformly inside the temporal window [t_start, t_end]."""
    n = max(int(n_frames), 1)
    rate = max(float(fps), 1e-6)
    off = float(offset)
    if not 0.0 <= off < 1.0:
        raise ValueError(f"offset must be in [0, 1), got {offset}")

    t_start, t_end = window
    f_start = max(0, min(n - 1, int(math.floor(t_start * rate))))
    f_end = max(0, min(n - 1, int(math.ceil(t_end * rate))))

    if f_end <= f_start:
        return [f_start] * k

    span_frames = f_end - f_start
    indices: list[int] = []
    for i in range(k):
        rel = (i + off) / float(k)
        idx = f_start + int(math.floor(rel * span_frames))
        indices.append(max(0, min(n - 1, idx)))
    return indices


def fuse_pass_probabilities(
    p_pass1: list[float] | np.ndarray,
    p_pass2: list[float] | np.ndarray,
    mode: str = "confidence_weighted",
) -> list[float]:
    """Combine predictions from Pass 1 (global uniform) and Pass 2 (dense window).

    Modes:
      - 'pass2_only': Directly use dense window probabilities.
      - 'ensemble_average': Simple 50/50 average.
      - 'confidence_weighted': Weight toward the pass with higher top-1 vs top-2 margin.
      - 'pass1_fallback': Use Pass 2 only if Pass 1 margin is low (< 0.25).
    """
    p1 = np.asarray(p_pass1, dtype=np.float64)
    p2 = np.asarray(p_pass2, dtype=np.float64)
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()

    if mode == "pass2_only":
        return p2.tolist()
    if mode == "ensemble_average":
        fused = 0.5 * p1 + 0.5 * p2
        return (fused / fused.sum()).tolist()
    if mode == "pass1_fallback":
        sorted_p1 = np.sort(p1)[::-1]
        margin1 = sorted_p1[0] - sorted_p1[1]
        if margin1 >= 0.25:
            return p1.tolist()
        return p2.tolist()

    # Default: confidence_weighted
    sorted_p1 = np.sort(p1)[::-1]
    sorted_p2 = np.sort(p2)[::-1]
    m1 = max(0.01, float(sorted_p1[0] - sorted_p1[1]))
    m2 = max(0.01, float(sorted_p2[0] - sorted_p2[1]))
    w1 = m1 / (m1 + m2)
    w2 = m2 / (m1 + m2)
    fused = w1 * p1 + w2 * p2
    return (fused / fused.sum()).tolist()
