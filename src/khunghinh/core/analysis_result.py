"""Kết quả phân tích auto-reframe — dataclass thuần (không phụ thuộc Qt), dễ kiểm thử."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnalysisResult:
    frame_count: int
    fps: float
    src_w: int
    src_h: int
    centers_px: np.ndarray            # (N, 2) float32 — tâm camera đã làm mượt (zoom-independent)
    faces_per_frame: list             # list[list[FaceBox]]
    active_track_per_frame: np.ndarray  # (N,) int32 — track_id người nói (-1 nếu không có)
    scene_cut_frames: np.ndarray      # (M,) int32
    params_fingerprint: str           # định danh để cache (loại trừ zoom)

    def __post_init__(self) -> None:
        n = self.frame_count
        if not (len(self.centers_px) == n and len(self.faces_per_frame) == n
                and len(self.active_track_per_frame) == n):
            raise ValueError("Độ dài mảng phân tích không khớp frame_count")
        cuts = np.asarray(self.scene_cut_frames)
        if n == 0:
            if cuts.size > 0:
                raise ValueError("scene_cut_frames phải rỗng khi frame_count == 0")
        elif cuts.size > 0 and (int(cuts.min()) < 0 or int(cuts.max()) >= n):
            raise ValueError("scene_cut_frames chứa chỉ số ngoài phạm vi [0, frame_count)")

    def centers_for_frame(self, i: int) -> tuple[float, float]:
        if self.frame_count == 0:
            return (self.src_w / 2.0, self.src_h / 2.0)
        j = min(max(i, 0), self.frame_count - 1)
        return (float(self.centers_px[j, 0]), float(self.centers_px[j, 1]))

    def faces_for_frame(self, i: int):
        if self.frame_count == 0:
            return [], -1
        j = min(max(i, 0), self.frame_count - 1)
        return self.faces_per_frame[j], int(self.active_track_per_frame[j])

    def make_center_provider(self):
        c = self.centers_px
        n = self.frame_count
        sw, sh = self.src_w, self.src_h

        def provider(i: int):
            if n == 0:
                return (sw / 2.0, sh / 2.0)
            j = min(max(i, 0), n - 1)
            return (float(c[j, 0]), float(c[j, 1]))

        return provider
