"""Giao diện (Protocol) cho các bộ phát hiện — cắm model mà không sửa engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    track_id: int | None = None

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@runtime_checkable
class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        """Trả về danh sách khuôn mặt trong 1 frame BGR."""
        ...


@runtime_checkable
class ActiveSpeakerDetector(Protocol):
    def score_speaking(
        self, face_crops: list[np.ndarray], audio_chunk: np.ndarray, sample_rate: int
    ) -> list[float]:
        """Trả về điểm 'đang nói' (0..1) cho từng khuôn mặt, cùng thứ tự đầu vào."""
        ...
