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
    def __init__(self, model_path: str, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, detect_width: int = 320):
        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(f"Không thấy model YuNet: {model_path!r}. {_DOWNLOAD_HINT}")
        self.model_path = model_path
        self.detect_width = max(1, int(detect_width))
        self._det = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        log.info("Đã nạp YuNet: %s (detect_width=%d)", model_path, self.detect_width)

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        # Thu nhỏ về detect_width TRƯỚC khi suy luận (YuNet @320 ~4x nhanh hơn full-res
        # mà vẫn bắt tốt mặt lớn của video dọc), rồi quy đổi hộp về toạ độ gốc — cùng
        # kiểu Haar. detect_width mặc định 320 cân bằng tốc độ/độ chính xác.
        scale = min(1.0, self.detect_width / w)
        if scale < 1.0:
            small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        self._det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._det.detect(small)
        inv = 1.0 / scale if scale > 0 else 1.0
        boxes: list[FaceBox] = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = f[:4]
                boxes.append(FaceBox(int(round(x * inv)), int(round(y * inv)),
                                     int(round(bw * inv)), int(round(bh * inv)), float(f[-1])))
        return boxes
