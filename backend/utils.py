"""
utils.py - Shared utility functions for the Smart Surveillance System.
Provides helpers for image encoding, drawing, FPS calculation, and colour
palette management used across all CV modules.
"""

import time
import base64
from collections import deque

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Colour palette – one distinct BGR colour per class index (up to 80 COCO)   #
# --------------------------------------------------------------------------- #
_RNG = np.random.default_rng(42)
PALETTE: list[tuple[int, int, int]] = [
    tuple(int(c) for c in _RNG.integers(60, 230, size=3))
    for _ in range(100)
]


def get_color(idx: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given integer index."""
    return PALETTE[idx % len(PALETTE)]


# --------------------------------------------------------------------------- #
#  FPS tracker                                                                 #
# --------------------------------------------------------------------------- #
class FPSCounter:
    """Rolling-window FPS counter.

    Keeps a deque of the last `window` frame timestamps and computes the
    average frame rate from them.
    """

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a new frame and return the current FPS estimate."""
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps(self) -> float:
        """Current FPS without recording a new tick."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Drawing helpers                                                             #
# --------------------------------------------------------------------------- #
def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay the FPS counter in the top-left corner of *frame*."""
    label = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (0, 255, 0),
    bg: bool = True,
) -> np.ndarray:
    """Draw a text label with an optional filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    if bg:
        cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw, y + baseline), color, -1)
        text_color = (255, 255, 255)
    else:
        text_color = color
    cv2.putText(frame, text, (x, y), font, scale, text_color, thick, cv2.LINE_AA)
    return frame


def draw_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box with a label on *frame*."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_label(frame, label, x1, y1, color)
    return frame


# --------------------------------------------------------------------------- #
#  Image encoding / decoding                                                   #
# --------------------------------------------------------------------------- #
def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR numpy array to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR numpy array to a base-64 JPEG string (for JSON transport)."""
    return base64.b64encode(encode_frame_to_jpeg(frame, quality)).decode()


def decode_base64_to_frame(b64: str) -> np.ndarray:
    """Decode a base-64 JPEG string back to a BGR numpy array."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def resize_keep_aspect(
    frame: np.ndarray, max_dim: int = 1280
) -> np.ndarray:
    """Resize *frame* so that its largest dimension is at most *max_dim*,
    preserving the aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
