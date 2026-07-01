from __future__ import annotations

import numpy as np
import pytest

from khunghinh.core.camera_path import (
    AutoZoomParams,
    CameraPathParams,
    SceneCutParams,
    build_camera_path,
    detect_scene_cuts,
    make_center_provider,
    suggest_zoom,
)
from khunghinh.core.geometry import compute_crop_rect
from khunghinh.core.reframe_engine import ReframeParams

W, H, AR = 1920, 1080, 9 / 16
RP = ReframeParams(W, H, AR, 1.0, 1.0)


def test_detect_cuts_basic():
    luma = np.array([10.0] * 30 + [200.0] * 30)
    hist = np.zeros(60)
    hist[30] = 0.9
    assert detect_scene_cuts(luma, hist) == [0, 30]


def test_detect_cuts_min_scene_len_suppresses_adjacent():
    luma = np.full(60, 50.0)
    hist = np.zeros(60)
    hist[30] = 0.9
    hist[34] = 0.95
    cuts = detect_scene_cuts(luma, hist, SceneCutParams(min_scene_len_frames=12))
    assert cuts == [0, 30]


def test_detect_cuts_empty_returns_no_cuts():
    assert detect_scene_cuts([], []) == []  # 0 frame -> không có cắt cảnh nào để báo


def test_detect_cuts_starts_with_zero_when_frames_exist():
    assert detect_scene_cuts(np.full(10, 5.0), np.zeros(10))[0] == 0


def test_deadzone_suppresses_jitter():
    n = 60
    raw = np.zeros((n, 2))
    raw[:, 0] = 960 + 12 * np.sin(np.arange(n))  # jitter ±12 < deadzone
    raw[:, 1] = 540
    path = build_camera_path(raw, [-1] * n, 30.0, RP, CameraPathParams(deadzone_frac_x=0.06))
    assert np.std(path.centers[:, 0]) < 5.0


def test_ramp_is_followed():
    n = 90
    cx = np.concatenate([np.linspace(576, 1344, 60), np.full(30, 1344.0)])
    raw = np.stack([cx, np.full(n, 540.0)], axis=1)
    path = build_camera_path(raw, [-1] * n, 30.0, RP,
                             CameraPathParams(deadzone_frac_x=0.0, deadzone_frac_y=0.0))
    assert abs(path.centers[-1, 0] - 1344) < 60


def test_snap_at_cut():
    n = 60
    cx = np.array([576.0] * 30 + [1344.0] * 30)
    raw = np.stack([cx, np.full(n, 540.0)], axis=1)
    path = build_camera_path(raw, [-1] * n, 30.0, RP, cut_frames=[0, 30])
    c = path.centers[:, 0]
    assert abs(c[30] - 1344) < 5          # snap đúng mục tiêu phải
    assert abs(c[30] - c[29]) > 300       # bước nhảy lớn tại cắt
    assert abs(c[31] - c[30]) < 50        # sau cắt thì êm


def test_settle_delays_speaker_switch():
    n = 30
    chosen = [1] * 10 + [2] * 10 + [1] * 10
    cx = np.array([576.0 if t == 1 else 1344.0 for t in chosen])
    raw = np.stack([cx, np.full(n, 540.0)], axis=1)
    path = build_camera_path(raw, chosen, 30.0, RP,
                             CameraPathParams(deadzone_frac_x=0.0, settle_frames=9))
    # Trong cửa sổ settle (frame 10..18), camera vẫn ở vùng trái.
    assert path.centers[15, 0] < 900


def test_nan_centers_hold_last_finite():
    n = 20
    cx = np.full(n, 960.0)
    cx[5:9] = np.nan
    raw = np.stack([cx, np.full(n, 540.0)], axis=1)
    path = build_camera_path(raw, [-1] * n, 30.0, RP,
                             CameraPathParams(deadzone_frac_x=0.0))
    assert np.isfinite(path.centers).all()
    assert abs(path.centers[6, 0] - 960) < 50


def test_recenter_reduces_trailing_offset_after_sustained_pan():
    # Mục tiêu pan sang phải rồi DỪNG. Không recenter → camera lệch ~dead-zone;
    # có recenter → sau khi pan dừng, camera bám sát tâm hơn hẳn.
    n = 70
    cx = np.concatenate([np.linspace(960, 1300, 40), np.full(30, 1300.0)])
    raw = np.stack([cx, np.full(n, 540.0)], axis=1)
    base = dict(deadzone_frac_x=0.06, deadzone_frac_y=0.0)
    off = build_camera_path(raw, [-1] * n, 30.0, RP,
                            CameraPathParams(recenter_frames=0, **base)).centers[-1, 0]
    on = build_camera_path(raw, [-1] * n, 30.0, RP,
                           CameraPathParams(recenter_frames=6, **base)).centers[-1, 0]
    assert abs(on - 1300) < abs(off - 1300)   # recenter đưa camera gần tâm hơn
    assert abs(on - 1300) < 12                 # gần như đúng tâm sau khi pan dừng


def test_recenter_frozen_during_nan_gap():
    # Pan (mục tiêu ra NGOÀI dead-zone) rồi MẤT MẶT (NaN) kéo dài: camera phải GIỮ
    # nguyên vị trí trailing, KHÔNG được recenter/nhích lên mục tiêu cũ khi 0 mặt.
    n = 40
    cx = np.concatenate([np.linspace(960, 1300, 12), np.full(28, np.nan)])
    raw = np.stack([cx, np.full(n, 540.0)], axis=1)
    path = build_camera_path(raw, [-1] * n, 30.0, RP,
                             CameraPathParams(deadzone_frac_x=0.06, recenter_frames=6))
    c = path.centers[-5:, 0]
    assert np.std(c) < 0.5        # đứng yên ở cuối đoạn gap
    assert c.mean() < 1290        # KHÔNG snap lên ~1300 (recenter sai trong gap)


def test_recenter_does_not_break_jitter_suppression():
    # Jitter trong dead-zone KHÔNG được kích hoạt recenter (giữ camera đứng yên).
    n = 60
    raw = np.zeros((n, 2))
    raw[:, 0] = 960 + 12 * np.sin(np.arange(n))
    raw[:, 1] = 540
    path = build_camera_path(raw, [-1] * n, 30.0, RP,
                             CameraPathParams(deadzone_frac_x=0.06, recenter_frames=6))
    assert np.std(path.centers[:, 0]) < 5.0


def test_every_center_yields_valid_crop():
    n = 40
    raw = np.stack([np.linspace(0, W, n), np.linspace(0, H, n)], axis=1)
    path = build_camera_path(raw, [-1] * n, 30.0, RP)
    for i in range(n):
        cx, cy = path.center_at(i)
        r = compute_crop_rect(W, H, AR, cx, cy, 1.0, 1.0)
        assert r.x >= 0 and r.y >= 0 and r.x + r.width <= W and r.y + r.height <= H


def test_make_center_provider_clamps():
    raw = np.stack([np.linspace(400, 1500, 10), np.full(10, 540.0)], axis=1)
    path = build_camera_path(raw, [-1] * 10, 30.0, RP)
    p = make_center_provider(path)
    assert p(-1) == path.center_at(0)
    assert p(999) == path.center_at(9)
    assert isinstance(p(5)[0], float)


def test_suggest_zoom():
    assert np.allclose(suggest_zoom([-1], [1, 1, 1]), 1.25)
    assert np.allclose(suggest_zoom([-1], [2, 2]), 1.0)
    z = suggest_zoom([-1], [1, 2, 1], zp=AutoZoomParams(manual_override=2.0))
    assert np.allclose(z, 2.0)


def test_suggest_zoom_debounces_flicker():
    fc = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  # nhấp nháy liên tục, không bao giờ ổn định
    z = suggest_zoom([-1] * len(fc), fc, zp=AutoZoomParams(min_stable_frames=5))
    assert np.all(z == z[0])  # giữ nguyên mức zoom ban đầu vì chưa đủ 5 frame liên tiếp


def test_suggest_zoom_switches_after_stable_run():
    fc = [1, 1, 1] + [2] * 10
    z = suggest_zoom([-1] * len(fc), fc, zp=AutoZoomParams(min_stable_frames=5))
    assert z[2] == pytest.approx(1.25)         # vẫn đang 1 mặt
    assert z[3 + 5 - 1] == pytest.approx(1.0)  # đủ 5 frame liên tiếp 2 mặt -> đổi
    assert z[-1] == pytest.approx(1.0)


def test_suggest_zoom_snaps_immediately_at_cut():
    fc = [1, 1, 1, 2, 2]
    z = suggest_zoom([-1] * len(fc), fc, cut_frames=[0, 3], zp=AutoZoomParams(min_stable_frames=5))
    assert z[3] == pytest.approx(1.0)  # snap ngay tại điểm cắt, không chờ debounce
