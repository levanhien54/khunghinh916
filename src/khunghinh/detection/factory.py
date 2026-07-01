"""Factory chọn bộ phát hiện khuôn mặt — YuNet nếu có .onnx, ngược lại Haar (zero-download)."""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import cv2

from ..config import AppConfig
from .base import FaceDetector
from .haar_face import DEFAULT_CASCADE_PATH, HaarFaceDetector, mouth_roi
from .yunet_face import YuNetFaceDetector

log = logging.getLogger(__name__)

__all__ = ["build_face_detector", "mouth_roi", "DEFAULT_CASCADE_PATH"]


def _find_default_yunet_model() -> str:
    """Tìm model YuNet đã bundle sẵn ở models/ (dev hoặc trong gói PyInstaller).

    Trả về đường dẫn đầu tiên khớp `face_detection_yunet_*.onnx`, '' nếu không thấy.
    """
    dirs = []
    meipass = getattr(sys, "_MEIPASS", None)  # PyInstaller: thư mục giải nén
    if meipass:
        dirs.append(Path(meipass) / "models")
    dirs.append(Path.cwd() / "models")
    dirs.append(Path(__file__).resolve().parents[3] / "models")  # <root>/models khi chạy dev
    for d in dirs:
        hits = sorted(glob.glob(str(d / "face_detection_yunet_*.onnx")))
        if hits:
            return hits[0]
    return ""


def build_face_detector(config: AppConfig) -> FaceDetector:
    """Trả về một FaceDetector luôn hoạt động. Không bao giờ raise ở nhánh offline mặc định.

    Ưu tiên YuNet (chính xác hơn Haar): dùng `yunet_model_path` nếu đặt, ngược lại tự
    tìm model đã bundle ở models/. Không thấy/không nạp được → Haar (zero-download).
    """
    model_path = (getattr(config, "yunet_model_path", "") or "").strip()
    if not model_path:
        model_path = _find_default_yunet_model()  # tự dùng model bundle nếu có
    if model_path and Path(model_path).is_file():
        try:
            det = YuNetFaceDetector(
                model_path,
                score_threshold=config.face_score_threshold,
                detect_width=getattr(config, "yunet_input_width", 320),
            )
            log.info("Face detector: YuNet (%s)", model_path)
            return det
        except (FileNotFoundError, cv2.error) as exc:
            log.warning("Nạp YuNet thất bại (%s) — chuyển sang Haar.", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("Lỗi YuNet (%s) — chuyển sang Haar.", exc)

    det = HaarFaceDetector(
        scale_factor=config.haar_scale_factor,
        min_neighbors=config.haar_min_neighbors,
        min_size_frac=config.face_min_size_frac,
        max_size_frac=config.face_max_size_frac,
        detect_width=config.face_detect_width,
    )
    log.info("Face detector: Haar cascade (zero-download).")
    return det
