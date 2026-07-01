"""Đặc trưng cảnh để phát hiện cắt cảnh: độ sáng (luma) + khác biệt histogram HSV.

Tách khỏi camera_path để detect_scene_cuts test được trên mảng tổng hợp (không cần cv2).
Ưu tiên gọi `compute_features_step` ngay trong vòng giải mã Pass1 (tránh decode 2 lần).
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_features_step(prev_hist, frame: np.ndarray, downscale_width: int = 320):
    """Một frame → (luma_mean, hist_diff_với_prev[0..1], hist_hiện_tại)."""
    h, w = frame.shape[:2]
    scale = min(1.0, downscale_width / w)
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luma = float(gray.mean())

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    if prev_hist is None:
        diff = 0.0
    else:
        diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
    return luma, diff, hist


def compute_scene_features(reader, downscale_width: int = 320):
    """Quét toàn video → (luma[N], hist_diffs[N]). hist_diffs[0] = 0."""
    reader.rewind()
    lumas: list[float] = []
    diffs: list[float] = []
    prev_hist = None
    while True:
        frame = reader.read_next()
        if frame is None:
            break
        luma, diff, prev_hist = compute_features_step(prev_hist, frame, downscale_width)
        lumas.append(luma)
        diffs.append(diff)
    return np.array(lumas, dtype=np.float64), np.array(diffs, dtype=np.float64)
