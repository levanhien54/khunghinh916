"""Phát hiện khuôn mặt CPU bằng Haar cascade có sẵn trong OpenCV — KHÔNG cần tải model.

Là fallback mặc định (zero-download) cho FaceDetector Protocol. Cũng là nơi đặt
hàm `mouth_roi()` (vùng miệng) cho bước phát hiện người nói.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .base import FaceBox

log = logging.getLogger(__name__)

DEFAULT_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def mouth_roi(
    face: FaceBox,
    frame_w: int,
    frame_h: int,
    top_frac: float = 0.60,
    height_frac: float = 0.40,
    pad_frac: float = 0.0,
) -> tuple[int, int, int, int]:
    """Trả về (x, y, w, h) vùng miệng (dải dưới của mặt), kẹp trong khung. w,h >= 1.

    Thuần hình học, không truy cập ảnh → ASD tự slice frame[y:y+h, x:x+w].
    """
    y0 = face.y + top_frac * face.h
    h = height_frac * face.h
    pad = pad_frac * face.w
    x0 = face.x - pad
    w = face.w + 2 * pad

    xi = int(round(max(0.0, min(x0, frame_w - 1))))
    yi = int(round(max(0.0, min(y0, frame_h - 1))))
    x1 = int(round(max(0.0, min(x0 + w, float(frame_w)))))
    y1 = int(round(max(0.0, min(y0 + h, float(frame_h)))))
    return (xi, yi, max(1, x1 - xi), max(1, y1 - yi))


class HaarFaceDetector:
    """FaceDetector dùng cv2.CascadeClassifier trên cascade frontal face có sẵn."""

    def __init__(
        self,
        cascade_path: str | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size_frac: float = 0.06,
        max_size_frac: float = 0.9,
        detect_width: int = 640,
        score_mode: str = "neighbors",
    ):
        path = cascade_path or DEFAULT_CASCADE_PATH
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Không nạp được Haar cascade: {path}")
        self.cascade_path = path
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size_frac = float(min_size_frac)
        self.max_size_frac = float(max_size_frac)
        self.detect_width = int(detect_width)
        self.score_mode = score_mode

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return []

        # Thu nhỏ TRƯỚC khi cvtColor/equalizeHist (chạy trên ảnh nhỏ rẻ hơn nhiều
        # so với chạy trên ảnh gốc rồi mới thu nhỏ — đặc biệt quan trọng khi xử lý
        # offline hàng loạt frame trên CPU-only).
        scale = min(1.0, self.detect_width / w)  # chỉ thu nhỏ, không phóng to
        if scale < 1.0:
            small_frame = cv2.resize(
                frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        else:
            small_frame = frame

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY) if small_frame.ndim == 3 else small_frame
        small = cv2.equalizeHist(gray)

        short = min(small.shape[:2])
        min_sz = max(1, int(self.min_size_frac * short))
        max_sz = int(self.max_size_frac * short)
        kwargs = {"scaleFactor": self.scale_factor, "minNeighbors": self.min_neighbors, "minSize": (min_sz, min_sz)}
        if max_sz > min_sz:
            kwargs["maxSize"] = (max_sz, max_sz)

        weights = None
        try:
            res = self._cascade.detectMultiScale3(small, outputRejectLevels=True, **kwargs)
            if isinstance(res, tuple) and len(res) == 3:
                rects, _levels, weights = res
            else:  # pragma: no cover
                rects = res
        except Exception:  # pragma: no cover - một số build không có detectMultiScale3
            rects = self._cascade.detectMultiScale(small, **kwargs)
            weights = None

        inv = 1.0 / scale
        boxes: list[FaceBox] = []
        for i, (x, y, bw, bh) in enumerate(rects):
            X = int(round(x * inv))
            Y = int(round(y * inv))
            BW = int(round(bw * inv))
            BH = int(round(bh * inv))
            # Kẹp trọn trong khung gốc.
            X = max(0, min(X, w - 1))
            Y = max(0, min(Y, h - 1))
            BW = max(1, min(BW, w - X))
            BH = max(1, min(BH, h - Y))
            boxes.append(FaceBox(X, Y, BW, BH, self._score(weights, i)))

        boxes.sort(key=lambda b: b.w * b.h, reverse=True)
        return boxes

    def _score(self, weights, i: int) -> float:
        if self.score_mode == "constant" or weights is None or len(weights) <= i:
            return 0.5
        try:
            wv = float(weights[i])
        except Exception:  # pragma: no cover
            return 0.5
        return float(max(0.05, min(0.99, 0.5 + 0.5 * math.tanh((wv - 2.0) / 2.5))))
