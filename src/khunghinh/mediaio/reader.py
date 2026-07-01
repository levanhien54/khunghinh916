"""Đọc video bằng OpenCV — dò metadata (độ phân giải, fps, tỉ lệ) và lấy frame."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def aspect_label(self) -> str:
        if not self.width or not self.height:
            return "?"
        g = gcd(self.width, self.height) or 1
        return f"{self.width // g}:{self.height // g}"


class VideoReader:
    """Bọc cv2.VideoCapture với context manager và thông báo lỗi rõ ràng.

    Lưu ý đa luồng: KHÔNG dùng chung một VideoReader giữa preview và luồng xuất —
    hãy tạo reader riêng cho mỗi luồng (cv2.VideoCapture không an toàn đa luồng).
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None

    def open(self) -> VideoInfo:
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"Không tìm thấy file video: {self.path}")
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"Không mở được video (thiếu codec?): {self.path}")
        self._cap = cap

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.info = VideoInfo(
            path=self.path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps) if fps > 0 else 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        log.info(
            "Mở video %s | %dx%d @ %.2ffps | %d frames",
            self.path, self.info.width, self.info.height, self.info.fps, self.info.frame_count,
        )
        return self.info

    def read_at(self, frame_index: int) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Chưa open() video.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = self._cap.read()
        return frame if ok else None

    def read_next(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Chưa open() video.")
        ok, frame = self._cap.read()
        return frame if ok else None

    def rewind(self) -> None:
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
