"""REZE-style option scores from per-frame Yes probabilities. No timestamp drawing."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

LETTERS = ("A", "B", "C", "D")
LAMBDA_OUT = 1.0

_POINT = re.compile(
    r"(?:(?:at|around|near)\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b",
    re.IGNORECASE,
)
_RANGE = re.compile(
    r"(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:-|–|to)\s*"
    r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?",
    re.IGNORECASE,
)
_BARE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*$", re.IGNORECASE)


def frame_times(meta: Any) -> list[float]:
    fps = float(getattr(meta, "fps", 0.0) or 0.0)
    if fps <= 0:
        raise ValueError("video metadata is missing a positive fps")
    return [int(index) / fps for index in meta.frames_indices]


def clip_duration(meta: Any) -> float:
    duration = float(getattr(meta, "duration", 0.0) or 0.0)
    if duration > 0:
        return duration
    times = frame_times(meta)
    fps = float(meta.fps)
    return max(times[-1] if times else 0.0, 1.0 / fps)


def interpolate_yes(times: list[float], yes: list[float], query_t: float) -> float:
    if len(times) != len(yes) or not times:
        raise ValueError("times and yes scores must be aligned and non-empty")
    if query_t <= times[0]:
        return float(yes[0])
    if query_t >= times[-1]:
        return float(yes[-1])
    for left, right in zip(range(len(times) - 1), range(1, len(times))):
        t0, t1 = times[left], times[right]
        if t0 <= query_t <= t1:
            if t1 == t0:
                return float(yes[left])
            frac = (query_t - t0) / (t1 - t0)
            return float(yes[left] * (1.0 - frac) + yes[right] * frac)
    return float(yes[-1])


def _window_energy(times: list[float], yes: np.ndarray, start: float, end: float) -> float:
    if end < start:
        start, end = end, start
    inside = []
    outside = []
    for time_s, score in zip(times, yes):
        (inside if start <= time_s <= end else outside).append(float(score))
    mean_in = float(np.mean(inside)) if inside else interpolate_yes(times, yes.tolist(), (start + end) / 2.0)
    mean_out = float(np.mean(outside)) if outside else 0.0
    return mean_in - LAMBDA_OUT * mean_out


def _qual_kind(text: str) -> str | None:
    lowered = text.casefold()
    if "intermittent" in lowered:
        return "intermittent"
    if "throughout" in lowered or "entire" in lowered or "whole" in lowered:
        return "throughout"
    if "most of" in lowered:
        return "most"
    if "first half" in lowered:
        return "first_half"
    if "briefly" in lowered and ("begin" in lowered or "start" in lowered):
        return "brief_start"
    if "begin" in lowered or "start" in lowered:
        return "start"
    if "middle" in lowered:
        return "middle"
    if "end" in lowered:
        return "end"
    return None


def parse_option(text: str, *, duration: float, duration_question: bool) -> dict[str, Any]:
    raw = str(text).strip()
    ranged = _RANGE.search(raw)
    if ranged:
        start, end = float(ranged.group(1)), float(ranged.group(2))
        return {"kind": "range", "start": start, "end": end}
    point = _POINT.search(raw)
    bare = _BARE.match(raw)
    if point or bare:
        value = float((point or bare).group(1))
        if duration_question:
            return {"kind": "duration", "seconds": value}
        return {"kind": "point", "time": value}
    kind = _qual_kind(raw)
    if kind is None:
        return {"kind": "unknown"}
    spans = {
        "start": (0.0, 0.25 * duration),
        "brief_start": (0.0, 0.20 * duration),
        "first_half": (0.0, 0.50 * duration),
        "middle": (0.30 * duration, 0.70 * duration),
        "end": (0.75 * duration, duration),
    }
    if kind in spans:
        start, end = spans[kind]
        return {"kind": "range", "start": start, "end": end, "qual": kind}
    return {"kind": kind}


def is_duration_question(question: str) -> bool:
    lowered = question.casefold()
    return any(token in lowered for token in ("how long", "how much time", "duration", "lasts"))


def predicted_span(times: list[float], yes: np.ndarray) -> float:
    if yes.size == 0:
        return 0.0
    shifted = yes - float(yes.mean())
    best = -math.inf
    best_len = 1
    for start in range(len(yes)):
        total = 0.0
        for end in range(start, len(yes)):
            total += float(shifted[end])
            if total > best:
                best = total
                best_len = max(1, end - start + 1)
    if len(times) >= 2:
        step = (times[-1] - times[0]) / max(1, len(times) - 1)
        return best_len * step
    return float(best_len)


def option_energy(row: dict[str, Any], times: list[float], yes: list[float], duration: float) -> list[float]:
    scores = np.asarray(yes, dtype=np.float64)
    duration_q = is_duration_question(row["question"])
    span = predicted_span(times, scores) if duration_q else None
    energy = []
    for letter in LETTERS:
        parsed = parse_option(row["options"][letter], duration=duration, duration_question=duration_q)
        kind = parsed["kind"]
        if kind == "point":
            energy.append(interpolate_yes(times, yes, parsed["time"]))
        elif kind == "range":
            energy.append(_window_energy(times, scores, parsed["start"], parsed["end"]))
        elif kind == "duration":
            energy.append(-abs((span or 0.0) - parsed["seconds"]))
        elif kind == "most":
            energy.append(float(scores.mean()))
        elif kind == "throughout":
            energy.append(float(scores.mean() - scores.std()))
        elif kind == "intermittent":
            energy.append(float(scores.std()))
        else:
            energy.append(0.0)
    return energy


def fuse_probs(letter_probs: list[float], energy: list[float], alpha: float) -> list[float]:
    if len(letter_probs) != 4 or len(energy) != 4:
        raise ValueError("letter probs and REZE energy must have length 4")
    logits = np.log(np.clip(np.asarray(letter_probs, dtype=np.float64), 1e-12, 1.0))
    fused = logits + float(alpha) * np.asarray(energy, dtype=np.float64)
    fused = fused - fused.max()
    probs = np.exp(fused)
    probs = probs / probs.sum()
    return [float(value) for value in probs]


def yes_no_ids(tokenizer) -> tuple[int, int]:
    yes = no = None
    for text in ("Yes", " Yes"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            yes = int(ids[0])
            break
    for text in ("No", " No"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            no = int(ids[0])
            break
    if yes is None or no is None or yes == no:
        raise ValueError("tokenizer does not map Yes/No to distinct single tokens")
    return yes, no


def yes_from_logits(logits, yes_id: int, no_id: int) -> float:
    yes = float(logits[yes_id])
    no = float(logits[no_id])
    m = max(yes, no)
    ey, en = math.exp(yes - m), math.exp(no - m)
    return ey / (ey + en)
