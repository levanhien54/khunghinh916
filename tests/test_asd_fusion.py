from __future__ import annotations

import numpy as np

from khunghinh.detection.asd_fusion import (
    ASDConfig,
    ActiveSpeakerSelector,
    HeuristicActiveSpeaker,
    LipMotionTracker,
    build_center_provider,
    compute_vad_energy,
)
from khunghinh.detection.base import ActiveSpeakerDetector, FaceBox
from khunghinh.detection.haar_face import mouth_roi


def test_compute_vad_energy_zero_vs_sine():
    assert compute_vad_energy(np.zeros(1600, np.float32), 16000) < 0.04
    sine = (0.5 * np.sin(2 * np.pi * 200 * np.arange(1600) / 16000)).astype(np.float32)
    assert compute_vad_energy(sine, 16000) > 0.2


def test_compute_vad_energy_int16_stereo_matches_float_mono():
    sine = 0.5 * np.sin(2 * np.pi * 200 * np.arange(1600) / 16000)
    mono = sine.astype(np.float32)
    stereo_i16 = np.stack([sine, sine], axis=1) * 32768.0
    stereo_i16 = stereo_i16.astype(np.int16)
    a = compute_vad_energy(mono, 16000)
    b = compute_vad_energy(stereo_i16, 16000)
    assert abs(a - b) < 0.05


def test_compute_vad_energy_empty():
    assert compute_vad_energy(np.array([], np.float32), 16000) == 0.0


def test_lip_motion_static_zero_moving_positive():
    lip = LipMotionTracker()
    static = np.zeros((40, 40, 3), np.uint8)
    assert lip.update(1, static, 0) == 0.0
    assert lip.update(1, static, 1) == 0.0  # không đổi -> 0
    moving0 = np.zeros((40, 40, 3), np.uint8)
    moving1 = np.full((40, 40, 3), 255, np.uint8)
    lip.update(2, moving0, 0)
    s = lip.update(2, moving1, 1)
    assert 0.0 < s < 1.0  # có chuyển động, luôn < 1 nhờ norm floor


def test_heuristic_is_protocol_and_empty():
    h = HeuristicActiveSpeaker()
    assert isinstance(h, ActiveSpeakerDetector)
    assert h.score_speaking([], None, 16000) == []


def test_heuristic_picks_moving_face_when_voiced():
    h = HeuristicActiveSpeaker()
    sine = (0.6 * np.sin(2 * np.pi * 200 * np.arange(1600) / 16000)).astype(np.float32)
    static = np.zeros((40, 40, 3), np.uint8)
    h.score_speaking([static, np.zeros((40, 40, 3), np.uint8)], sine, 16000)  # khởi tạo prev
    scores = h.score_speaking([static, np.full((40, 40, 3), 255, np.uint8)], sine, 16000)
    assert scores[1] > scores[0]


def _frame():
    return np.zeros((100, 200, 3), np.uint8)


def test_selector_no_faces_fallback_center():
    sel = ActiveSpeakerSelector(200, 100)
    d = sel.update(_frame(), [], 0.9, 0)
    assert d.track_id is None and d.is_fallback
    assert d.cx == 100 and d.cy == 50


def test_selector_vad_zero_fallback_picks_larger():
    sel = ActiveSpeakerSelector(200, 100)
    small = FaceBox(10, 10, 30, 30, 0.9, track_id=1)
    large = FaceBox(120, 10, 60, 60, 0.9, track_id=2)
    d = sel.update(_frame(), [small, large], 0.0, 0)
    assert d.is_fallback
    assert d.track_id == 2  # mặt lớn hơn


def test_selector_hysteresis_switch_with_scripted_detector():
    class ScriptedASD:
        scores: list = []

        def score_speaking(self, crops, chunk, sr):
            return list(self.scores)

    asd = ScriptedASD()
    sel = ActiveSpeakerSelector(
        200, 100,
        config=ASDConfig(min_dwell_frames=8, switch_confirm_frames=4, hysteresis_margin=1.25),
        detector=asd, audio_provider=lambda i: (None, 16000),
    )
    faces = [FaceBox(10, 10, 40, 40, 0.9, track_id=1), FaceBox(140, 10, 40, 40, 0.9, track_id=2)]

    # Pha 1: track 1 nói.
    asd.scores = [1.0, 0.0]
    for i in range(10):
        d = sel.update(_frame(), faces, 1.0, i)
    assert d.track_id == 1

    # Một spike của track 2 rồi quay lại 1 -> chưa đủ switch_confirm, giữ 1.
    asd.scores = [0.0, 1.0]
    sel.update(_frame(), faces, 1.0, 10)
    asd.scores = [1.0, 0.0]
    d = sel.update(_frame(), faces, 1.0, 11)
    assert d.track_id == 1

    # Track 2 nói bền vững -> chuyển sang 2.
    asd.scores = [0.0, 1.0]
    for i in range(12, 24):
        d = sel.update(_frame(), faces, 1.0, i)
    assert d.track_id == 2


def test_selector_forces_switch_when_locked_track_disappears():
    class ScriptedASD:
        scores: list = []

        def score_speaking(self, crops, chunk, sr):
            return list(self.scores)

    asd = ScriptedASD()
    sel = ActiveSpeakerSelector(
        200, 100,
        config=ASDConfig(min_dwell_frames=8, switch_confirm_frames=4, hysteresis_margin=1.25),
        detector=asd, audio_provider=lambda i: (None, 16000),
    )
    face1 = FaceBox(10, 10, 40, 40, 0.9, track_id=1)
    face2 = FaceBox(140, 10, 40, 40, 0.9, track_id=2)

    # Khoá vào track 1.
    asd.scores = [1.0, 0.0]
    d = None
    for i in range(10):
        d = sel.update(_frame(), [face1, face2], 1.0, i)
    assert d.track_id == 1

    # Track 1 biến mất khỏi faces (mất bám) — chỉ còn track 2, đang nói.
    asd.scores = [1.0]
    d = sel.update(_frame(), [face2], 1.0, 10)
    # Phải chuyển NGAY sang track 2 (track 1 đã chết, hysteresis không áp dụng được
    # cho 1 id không còn tồn tại) — không rơi vào lựa chọn tùy tiện/giữ id cũ.
    assert d.track_id == 2

    # Track 1 quay lại cùng track 2 — track 2 vẫn đang giữ, không bị xáo trộn.
    asd.scores = [0.0, 1.0]
    d = sel.update(_frame(), [face1, face2], 1.0, 11)
    assert d.track_id == 2


def test_selector_no_score_collision_when_track_ids_are_none():
    cfg = ASDConfig(vad_threshold=0.0, lip_window=1)
    sel = ActiveSpeakerSelector(200, 100, config=cfg)
    face_a = FaceBox(10, 10, 40, 40, 0.9, track_id=None)   # cx=30
    face_b = FaceBox(140, 10, 40, 40, 0.9, track_id=None)  # cx=160

    quiet = np.zeros((100, 200, 3), np.uint8)
    sel.update(quiet, [face_a, face_b], vad=1.0, frame_idx=0)  # khởi tạo lịch sử môi

    moving = quiet.copy()
    bx, by, bw, bh = mouth_roi(face_b, 200, 100, top_frac=cfg.mouth_y0, height_frac=cfg.mouth_y1 - cfg.mouth_y0)
    moving[by:by + bh, bx:bx + bw] = 255  # chỉ vùng miệng face_b đổi mạnh

    d = sel.update(moving, [face_a, face_b], vad=1.0, frame_idx=1)
    # Trước khi sửa: scores[None] bị va chạm, _face_by_id(faces, None) luôn trả về
    # mặt ĐẦU TIÊN (face_a) bất kể mặt nào thực sự có điểm cao hơn -> sai vị trí.
    assert abs(d.cx - face_b.cx) < 1.0


def test_build_center_provider():
    from khunghinh.detection.asd_fusion import SpeakerDecision

    d0 = SpeakerDecision(1, 10.0, 20.0, 0.5, 1.0, False)
    d1 = SpeakerDecision(2, 30.0, 40.0, 0.5, 1.0, False)
    p = build_center_provider([d0, d1], (5.0, 5.0))
    assert p(99) == (30.0, 40.0)
    assert build_center_provider([], (5.0, 5.0))(0) == (5.0, 5.0)
