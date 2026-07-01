"""Phát hiện khuôn mặt YuNet (OpenCV FaceDetectorYN) — nhẹ, MIT, chạy tốt trên CPU.

Cần file model ONNX `face_detection_yunet_*.onnx` (tải từ OpenCV Zoo — xem
models/README.md). Đây là thành phần TÙY CHỌN: app vẫn chạy ở chế độ thủ công
khi chưa có model. Class này hiện thực Protocol `FaceDetector`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .base import FaceBox

log = logging.getLogger(__name__)

_DOWNLOAD_HINT = (
    "Tải YuNet (MIT) tại: "
    "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet "
    "rồi trỏ đường dẫn .onnx qua cấu hình `yunet_model_path`."
)


class YuNetFaceDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(f"Không thấy model YuNet: {model_path!r}. {_DOWNLOAD_HINT}")
        self.model_path = model_path
        self._det = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        log.info("Đã nạp YuNet: %s", model_path)

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        boxes: list[FaceBox] = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = f[:4]
                boxes.append(FaceBox(int(x), int(y), int(bw), int(bh), float(f[-1])))
        return boxes
