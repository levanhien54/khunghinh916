from __future__ import annotations

import numpy as np
import pytest

from khunghinh.core.analysis_result import AnalysisResult
from khunghinh.detection.base import FaceBox


def _make(n=5):
    centers = np.stack([np.arange(n) * 10.0, np.arange(n) * 5.0], axis=1).astype(np.float32)
    faces = [[FaceBox(0, 0, 10, 10, 0.9, track_id=1)] for _ in range(n)]
    active = np.arange(n, dtype=np.int32)
    return AnalysisResult(n, 30.0, 1920, 1080, centers, faces, active, np.array([0], np.int32), "fp")


def test_centers_for_frame_clamps():
    r = _make(5)
    assert r.centers_for_frame(-5) == (0.0, 0.0)
    assert r.centers_for_frame(999) == (40.0, 20.0)


def test_faces_for_frame():
    r = _make(5)
    faces, active = r.faces_for_frame(2)
    assert len(faces) == 1 and active == 2


def test_center_provider_matches():
    r = _make(5)
    p = r.make_center_provider()
    for i in range(5):
        assert p(i) == r.centers_for_frame(i)
    assert p(99) == r.centers_for_frame(4)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        AnalysisResult(
            5, 30.0, 100, 100,
            np.zeros((3, 2), np.float32),               # sai độ dài
            [[] for _ in range(5)],
            np.zeros(5, np.int32),
            np.array([0], np.int32),
            "fp",
        )


def test_scene_cut_frames_out_of_range_raises():
    with pytest.raises(ValueError):
        AnalysisResult(
            3, 30.0, 100, 100,
            np.zeros((3, 2), np.float32),
            [[] for _ in range(3)],
            np.zeros(3, np.int32),
            np.array([0, 5], np.int32),  # 5 vượt ngoài [0, 3)
            "fp",
        )


def test_zero_frames_requires_empty_scene_cut_frames():
    with pytest.raises(ValueError):
        AnalysisResult(
            0, 30.0, 100, 100,
            np.zeros((0, 2), np.float32),
            [],
            np.zeros(0, np.int32),
            np.array([0], np.int32),  # claim cắt cảnh tại frame 0 của video 0-frame -> sai
            "fp",
        )
    # Rỗng thì hợp lệ.
    AnalysisResult(0, 30.0, 100, 100, np.zeros((0, 2), np.float32), [], np.zeros(0, np.int32),
                   np.zeros(0, np.int32), "fp")


def test_fingerprint_differs_on_dims():
    a = _make(3).params_fingerprint
    centers = np.zeros((3, 2), np.float32)
    faces = [[] for _ in range(3)]
    active = np.zeros(3, np.int32)
    b = AnalysisResult(3, 30.0, 999, 1080, centers, faces, active, np.array([0], np.int32), "other")
    assert a != b.params_fingerprint
