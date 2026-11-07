"""Horizontal-flip rewrite for Geometric Relation, Perspective, and Change Detection.

A row is mirrored only when every left/right phrase can be swapped and the
original gold text, after that same swap, lands on exactly one rewritten
option. Directional language that a horizontal flip does not invert is left
untouched and the row is skipped.
"""

from __future__ import annotations

import re

import common as C

MIRROR_TASKS = (
    "Geometric Relation",
    "Perspective and Viewpoint",
    "Change Detection",
)

_UNSAFE_DIRECTION = re.compile(
    r"\b("
    r"north|south|east|west|clockwise|counterclockwise|anticlockwise|"
    r"forward|backward|forwards|backwards|toward|towards|away|"
    r"above|below|front|behind|upward|downward|incoming|outgoing|"
    r"approaching|leaving|ascending|descending|uphill|downhill|"
    r"upstream|downstream|up|down"
    r")\b",
    re.IGNORECASE,
)
_LEFT_RIGHT = re.compile(r"\b(leftmost|rightmost|left|right)\b", re.IGNORECASE)
_SWAP = {
    "leftmost": "rightmost",
    "rightmost": "leftmost",
    "left": "right",
    "right": "left",
}


def has_left_right(text: str) -> bool:
    return _LEFT_RIGHT.search(text) is not None


def has_unsafe_direction(text: str) -> bool:
    return _UNSAFE_DIRECTION.search(text) is not None


def swap_left_right(text: str) -> str:
    """Swap horizontal direction words and preserve each token's capitalization."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        swapped = _SWAP[word.lower()]
        if word.isupper():
            return swapped.upper()
        if word[0].isupper():
            return swapped[:1].upper() + swapped[1:]
        return swapped

    return _LEFT_RIGHT.sub(replace, text)


def mirror_qa(row: dict) -> dict | None:
    """Return a rewritten QA row, or None when a horizontal flip is not safe."""

    if row.get("task") not in MIRROR_TASKS:
        return None
    options = row.get("options")
    answer = row.get("correct_answer")
    question = row.get("question")
    if not isinstance(question, str) or not isinstance(options, dict) or answer not in C.LETTERS:
        return None
    if any(letter not in options or not isinstance(options[letter], str) for letter in C.LETTERS):
        return None
    blob = " ".join([question, *(options[letter] for letter in C.LETTERS)])
    if has_unsafe_direction(blob) or not has_left_right(blob):
        return None
    rewritten_options = {letter: swap_left_right(options[letter]) for letter in C.LETTERS}
    mirrored_gold = swap_left_right(options[answer])
    matches = [letter for letter in C.LETTERS if rewritten_options[letter] == mirrored_gold]
    if len(matches) != 1:
        return None
    rewritten = dict(row)
    rewritten["question"] = swap_left_right(question)
    rewritten["options"] = rewritten_options
    rewritten["correct_answer"] = matches[0]
    return rewritten


def flip_horizontal(frames):
    """Reverse width on a THWC frame stack and return a contiguous copy."""

    import numpy as np

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[2] < 1:
        raise ValueError(f"expected THWC frames, got {array.shape}")
    return np.ascontiguousarray(array[:, :, ::-1, ...])
