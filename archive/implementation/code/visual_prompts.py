"""Training-free pixel/text overlays for frozen Stage A scoring.

Progress bars follow VTimeCoT's frame-sync bar (seconds + filled fraction).
Camera-motion sentences follow Peekaboo-VL's ORB+RANSAC vocabulary. Neither
path changes weights, frame indices, or letter scoring.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

BAR_HEIGHT = 32
MOTION_UNAVAILABLE = "Camera motion: unavailable"

_DIGIT = {
    "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00100", "00100"],
    "s": ["00000", "01111", "10000", "01110", "00001", "11110", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def frame_timestamps(meta: Any) -> list[float]:
    fps = float(getattr(meta, "fps", 0.0) or 0.0)
    if fps <= 0:
        raise ValueError("video metadata is missing a positive fps")
    return [int(index) / fps for index in meta.frames_indices]


def clip_duration(meta: Any) -> float:
    duration = float(getattr(meta, "duration", 0.0) or 0.0)
    if duration > 0:
        return duration
    indices = [int(index) for index in meta.frames_indices]
    fps = float(meta.fps)
    last = (max(indices) + 1) / fps if indices else 0.0
    return max(last, 1.0 / fps)


def _stamp_glyph(bar: np.ndarray, text: str, *, x: int, y: int) -> None:
    on = np.array([255, 255, 255], dtype=np.uint8)
    cursor = x
    for char in text:
        pattern = _DIGIT.get(char, _DIGIT[" "])
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit != "1":
                    continue
                yy, xx = y + row, cursor + col
                if 0 <= yy < bar.shape[0] and 0 <= xx < bar.shape[1]:
                    bar[yy, xx] = on
        cursor += 6


def draw_progress_bar(
    frames: np.ndarray,
    timestamps: Iterable[float],
    duration: float,
    *,
    bar_height: int = BAR_HEIGHT,
) -> np.ndarray:
    """Concatenate a filled timebar under each frame. Shape becomes (T, H+bar, W, C)."""

    if frames.ndim != 4 or frames.shape[-1] not in (1, 3):
        raise ValueError(f"expected THWC frames, got {frames.shape}")
    times = [float(value) for value in timestamps]
    if len(times) != frames.shape[0]:
        raise ValueError("one timestamp is required per frame")
    duration = max(float(duration), 1e-6)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    channels = int(frames.shape[-1])
    if bar_height < 12:
        raise ValueError("bar_height must be at least 12 pixels")
    bars = np.zeros((len(times), bar_height, width, channels), dtype=frames.dtype)
    track_top = 3
    track_bot = bar_height - 8
    for index, time_s in enumerate(times):
        frac = min(max(time_s / duration, 0.0), 1.0)
        bars[index, :, :] = (28, 28, 28)[:channels]
        bars[index, track_top:track_bot, 1 : width - 1] = (70, 70, 70)[:channels]
        fill = max(1, int(round(frac * (width - 2))))
        bars[index, track_top:track_bot, 1 : 1 + fill] = (46, 196, 92)[:channels]
        cx = min(width - 2, max(2, 1 + fill))
        cy = (track_top + track_bot) // 2
        for yy in range(cy - 3, cy + 4):
            for xx in range(cx - 3, cx + 4):
                if 0 <= yy < bar_height and 0 <= xx < width and (yy - cy) ** 2 + (xx - cx) ** 2 <= 9:
                    bars[index, yy, xx] = (255, 255, 255)[:channels]
        _stamp_glyph(bars[index], f"{time_s:.1f}s", x=4, y=bar_height - 9)
    return np.concatenate([frames, bars], axis=1)


def overlay_progress_bar(frames: np.ndarray, meta: Any, *, bar_height: int = BAR_HEIGHT):
    """Return bar-augmented frames and metadata with the new pixel height."""

    from common import _video_metadata

    stamped = draw_progress_bar(
        np.asarray(frames),
        frame_timestamps(meta),
        clip_duration(meta),
        bar_height=bar_height,
    )
    new_meta = _video_metadata(
        total_num_frames=int(meta.total_num_frames),
        fps=float(meta.fps),
        frames_indices=list(meta.frames_indices),
        width=int(stamped.shape[2]),
        height=int(stamped.shape[1]),
        duration=float(getattr(meta, "duration", 0.0) or 0.0),
    )
    return stamped, new_meta


def classify_pairwise_motion(dx: float, dy: float, scale: float, width: int) -> str:
    """Map a homography/affine step into Peekaboo-style UAV motion words."""

    if width <= 0:
        raise ValueError("frame width must be positive")
    nx, ny = dx / width, dy / width
    ds = scale - 1.0
    translation = float(np.hypot(nx, ny))
    if abs(ds) >= 0.045 and abs(ds) >= translation:
        verb = "moving forward" if ds > 0 else "moving backward"
        speed = abs(ds)
    elif translation < 0.012:
        return "hovering stationary"
    elif abs(nx) >= abs(ny):
        # Scene shift right (positive dx) means the camera panned left.
        verb = "panning left" if nx > 0 else "panning right"
        speed = translation
    else:
        verb = "tilting down" if ny > 0 else "tilting up"
        speed = translation
    pace = "rapid" if speed >= 0.05 else "slow"
    return f"{verb}, {pace}"


def merge_motion_segments(
    timestamps: Iterable[float],
    labels: Iterable[str],
    duration: float,
) -> str:
    times = [float(value) for value in timestamps]
    names = [str(value) for value in labels]
    if not times or len(names) != max(0, len(times) - 1):
        raise ValueError("need N timestamps and N-1 pairwise labels")
    end = max(float(duration), times[-1])
    edges = [*times, end]
    parts: list[str] = []
    start_i = 0
    for index in range(1, len(names) + 1):
        if index < len(names) and names[index] == names[start_i]:
            continue
        end_t = end if index == len(names) else times[index]
        parts.append(f"{edges[start_i]:.1f}s–{end_t:.1f}s {names[start_i]}")
        start_i = index
    return "Camera motion: " + "; ".join(parts)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("expected HWC frame")
    if frame.shape[-1] == 1:
        gray = frame[:, :, 0]
    else:
        gray = np.round(0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.uint8)
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > 480:
        scale = 480 / longest
        new_w = max(8, int(round(width * scale)))
        new_h = max(8, int(round(height * scale)))
        try:
            import cv2
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            yy = (np.linspace(0, height - 1, new_h)).astype(np.int64)
            xx = (np.linspace(0, width - 1, new_w)).astype(np.int64)
            gray = gray[yy][:, xx]
    return gray


def _pairwise_affine(prev: np.ndarray, nxt: np.ndarray) -> tuple[float, float, float] | None:
    import cv2

    orb = cv2.ORB_create(nfeatures=800)
    k0, d0 = orb.detectAndCompute(prev, None)
    k1, d1 = orb.detectAndCompute(nxt, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d0, d1)
    if len(matches) < 8:
        return None
    matches = sorted(matches, key=lambda item: item.distance)[:80]
    src = np.float32([k0[item.queryIdx].pt for item in matches])
    dst = np.float32([k1[item.trainIdx].pt for item in matches])
    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine is None or inliers is None or int(inliers.sum()) < 6:
        return None
    dx = float(affine[0, 2])
    dy = float(affine[1, 2])
    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    return dx, dy, scale


def format_cotr_timestamp(time_s: float) -> str:
    """CoTR-style absolute time glyph (mm:ss + bracketed seconds)."""

    time_s = max(0.0, float(time_s))
    minutes = int(time_s // 60)
    seconds = time_s - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f} [{time_s:.2f} s]"


def overlay_cotr_timestamp(
    frames: np.ndarray,
    meta: Any,
    *,
    margin: int = 6,
    band_height: int = 22,
) -> np.ndarray:
    """Burn CoTR visual time anchors into the top-right of each frame (no extra pixels)."""

    array = np.asarray(frames).copy()
    if array.ndim != 4:
        raise ValueError(f"expected THWC frames, got {array.shape}")
    times = frame_timestamps(meta)
    if len(times) != array.shape[0]:
        raise ValueError("one timestamp per frame is required")
    channels = int(array.shape[-1])
    dark = np.array((16, 16, 16), dtype=array.dtype)[:channels]
    for index, time_s in enumerate(times):
        label = format_cotr_timestamp(time_s)
        glyph_w = 6 * len(label)
        height, width = int(array.shape[1]), int(array.shape[2])
        x0 = max(margin, width - margin - glyph_w)
        y0 = margin
        y1 = min(height, y0 + band_height)
        x1 = min(width, x0 + glyph_w + 4)
        array[index, y0:y1, x0:x1] = dark
        _stamp_glyph(array[index], label, x=x0 + 2, y=y0 + 2)
    return array


def camera_motion_hint(frames: np.ndarray, meta: Any) -> str:
    """One Peekaboo-style sentence, or MOTION_UNAVAILABLE if ORB cannot lock."""

    array = np.asarray(frames)
    if array.shape[0] < 2:
        return MOTION_UNAVAILABLE
    times = frame_timestamps(meta)
    labels: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        return MOTION_UNAVAILABLE
    width = int(array.shape[2])
    try:
        for index in range(array.shape[0] - 1):
            step = _pairwise_affine(_gray(array[index]), _gray(array[index + 1]))
            if step is None:
                labels.append("hovering stationary")
            else:
                labels.append(classify_pairwise_motion(*step, width=min(width, 480)))
        return merge_motion_segments(times, labels, clip_duration(meta))
    except Exception:
        return MOTION_UNAVAILABLE
