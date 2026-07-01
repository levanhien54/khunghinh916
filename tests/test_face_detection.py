from __future__ import annotations

import os

import numpy as np

from khunghinh.config import AppConfig
from khunghinh.detection.base import FaceBox, FaceDetector
from khunghinh.detection.factory import build_face_detector
from khunghinh.detection.haar_face import DEFAULT_CASCADE_PATH, HaarFaceDetector, mouth_roi


def test_default_cascade_exists():
    assert DEFAULT_CASCADE_PATH.endswith("haarcascade_frontalface_default.xml")
    assert os.path.isfile(DEFAULT_CASCADE_PATH)


def test_haar_constructs_and_handles_empty_input():
    det = HaarFaceDetector()
    assert det.detect(None) == []
    assert det.detect(np.zeros((0, 0, 3), np.uint8)) == []


def test_haar_detect_returns_list_no_crash():
    det = HaarFaceDetector()
    out = det.detect(np.zeros((480, 640, 3), np.uint8))
    assert isinstance(out, list)
    for b in out:
        assert isinstance(b, FaceBox)
        assert 0 <= b.x and 0 <= b.y
        assert b.x + b.w <= 640 and b.y + b.h <= 480
        assert b.track_id is None


def test_mouth_roi_exact():
    assert mouth_roi(FaceBox(100, 100, 200, 200), 1920, 1080) == (100, 220, 200, 80)


def test_mouth_roi_bottom_edge_clamps():
    x, y, w, h = mouth_roi(FaceBox(0, 1060, 100, 100), 1920, 1080)
    assert x >= 0 and y >= 0
    assert x + w <= 1920 and y + h <= 1080
    assert w >= 1 and h >= 1


def test_factory_default_prefers_yunet_when_model_bundled():
    # Path rỗng → tự tìm model YuNet bundle ở models/. Có model → YuNet (chính xác
    # hơn); không có (env khác) → fallback Haar. Luôn trả FaceDetector hợp lệ.
    from khunghinh.detection.factory import _find_default_yunet_model
    from khunghinh.detection.yunet_face import YuNetFaceDetector
    det = build_face_detector(AppConfig(yunet_model_path=""))
    assert isinstance(det, FaceDetector)
    if _find_default_yunet_model():
        assert isinstance(det, YuNetFaceDetector)
    else:
        assert isinstance(det, HaarFaceDetector)


def test_factory_nonexistent_model_falls_back():
    # Path YuNet đặt TƯỜNG MINH nhưng sai → KHÔNG tự dò default, fallback Haar.
    det = build_face_detector(AppConfig(yunet_model_path="C:/khong/ton/tai/model.onnx"))
    assert isinstance(det, HaarFaceDetector)
