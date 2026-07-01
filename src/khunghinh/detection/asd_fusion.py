"""Phát hiện người nói (WHO) = VAD âm thanh (KHI NÀO) × chuyển động môi mỗi mặt (AI).

- `ActiveSpeakerSelector`: dùng trong pipeline chính. Nhận điểm VAD MỀM [0,1] đã tính
  sẵn (từ audio_vad), KHÔNG tự giải mã audio. Chọn người nói theo argmax có hysteresis/
  dwell chống nhấp nháy; fallback hình học (mặt lớn nhất, rồi gần tâm) khi không có
  giọng/không có chuyển động. Đây cũng là nơi cắm model sâu (LR-ASD/TalkNet) qua tham số
  `detector` + `audio_provider`.
- `HeuristicActiveSpeaker`: hiện thực Protocol `ActiveSpeakerDetector` (đường stateless,
  dùng compute_vad_energy nội bộ) — chỗ thay model sâu.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import FaceBox
from .haar_face import mouth_roi


@dataclass
class ASDConfig:
    vad_threshold: float = 0.5      # ngưỡng trên ĐIỂM VAD MỀM [0,1]
    lip_window: int = 5
    lip_patch_size: int = 48
    mouth_y0: float = 0.55
    mouth_y1: float = 1.0
    lip_norm_floor: float = 1.0
    hysteresis_margin: float = 1.25
    min_dwell_frames: int = 8
    switch_confirm_frames: int = 4
    track_timeout_frames: int = 15
    energy_vad_threshold: float = 0.04  # CHỈ cho compute_vad_energy (RMS thô)


@dataclass(frozen=True)
class SpeakerDecision:
    track_id: int | None
    cx: float
    cy: float
    fused_score: float
    vad: float
    is_fallback: bool


def compute_vad_energy(audio_chunk, sample_rate: int, prev_smoothed: float = 0.0, alpha: float = 0.3) -> float:
    """Điểm năng lượng giọng nói thô (RMS chuẩn hoá + EMA). CHỈ cho HeuristicActiveSpeaker."""
    if audio_chunk is None:
        return float(prev_smoothed) * (1 - alpha)
    a = np.asarray(audio_chunk)
    if a.size == 0:
        return float(prev_smoothed) * (1 - alpha)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float64) / 32768.0
    else:
        a = a.astype(np.float64)
    if a.ndim > 1:
        a = a.mean(axis=1)
    rms = float(np.sqrt(np.mean(a * a)))
    score = min(1.0, rms / 0.1)
    return float(alpha * score + (1 - alpha) * prev_smoothed)


class LipMotionTracker:
    """Đo chuyển động vùng miệng theo từng track qua thời gian → điểm [0,1)."""

    def __init__(self, config: ASDConfig | None = None):
        self.c = config or ASDConfig()
        self._prev: dict = {}
        self._hist: dict = {}
        self._last_seen: dict = {}

    def reset(self) -> None:
        self._prev.clear()
        self._hist.clear()
        self._last_seen.clear()

    def _prep(self, crop):
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        s = self.c.lip_patch_size
        return cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)

    def update(self, track_id, face_crop, frame_idx: int) -> float:
        patch = self._prep(face_crop)
        prev = self._prev.get(track_id)
        self._prev[track_id] = patch
        self._last_seen[track_id] = frame_idx

        if prev is None or patch is None or prev.shape != patch.shape:
            motion = 0.0
        else:
            motion = float(np.mean(np.abs(patch - prev)))

        dq = self._hist.setdefault(track_id, [])
        dq.append(motion)
        if len(dq) > self.c.lip_window:
            dq.pop(0)
        windowed = float(np.mean(dq)) if dq else 0.0
        return windowed / (windowed + self.c.lip_norm_floor)

    def prune(self, frame_idx: int) -> None:
        dead = [k for k, v in self._last_seen.items() if frame_idx - v > self.c.track_timeout_frames]
        for k in dead:
            self._prev.pop(k, None)
            self._hist.pop(k, None)
            self._last_seen.pop(k, None)


class HeuristicActiveSpeaker:
    """Hiện thực Protocol ActiveSpeakerDetector — chỗ thay model sâu LR-ASD/TalkNet."""

    def __init__(self, config: ASDConfig | None = None):
        self.c = config or ASDConfig()
        self._lip = LipMotionTracker(self.c)
        self._frame = 0

    def score_speaking(self, face_crops, audio_chunk, sample_rate: int) -> list[float]:
        if not face_crops:
            return []
        vad = compute_vad_energy(audio_chunk, sample_rate)
        voiced = vad >= self.c.energy_vad_threshold
        scores = []
        for i, crop in enumerate(face_crops):
            lip = self._lip.update(i, crop, self._frame)
            scores.append(float(vad * lip) if voiced else 0.0)
        self._frame += 1
        return scores


class ActiveSpeakerSelector:
    """Chọn người nói theo từng frame trong pipeline chính (VAD mềm + chuyển động môi)."""

    def __init__(self, src_w: int, src_h: int, config: ASDConfig | None = None, detector=None, audio_provider=None):
        self.src_w = src_w
        self.src_h = src_h
        self.c = config or ASDConfig()
        self.detector = detector            # ActiveSpeakerDetector (đường model sâu, tùy chọn)
        self.audio_provider = audio_provider  # callable(frame_idx)->(chunk, sr) cho đường sâu
        self._lip = LipMotionTracker(self.c)
        self._current: int | None = None
        self._dwell = 0
        self._candidate: int | None = None
        self._candidate_count = 0

    def reset(self) -> None:
        self._lip.reset()
        self._current = None
        self._dwell = 0
        self._candidate = None
        self._candidate_count = 0

    def update(self, frame, faces: list[FaceBox], vad: float, frame_idx: int) -> SpeakerDecision:
        if not faces:
            self._current = None
            self._dwell = 0
            return SpeakerDecision(None, self.src_w / 2.0, self.src_h / 2.0, 0.0, vad, True)

        # `scores` được đánh khoá theo `_face_key` (track_id nếu có, ngược lại id(f))
        # để KHÔNG bao giờ va chạm khi nhiều mặt có track_id=None (đường model sâu
        # chưa qua tracker) — mọi tra cứu bên dưới (best/hysteresis/current) đều
        # dùng cùng hệ khoá này, chỉ quy đổi sang track_id thật khi trả kết quả.
        scores = self._score_faces(frame, faces, vad, frame_idx)
        self._lip.prune(frame_idx)

        best_key = max(scores, key=lambda k: scores[k])
        best_score = scores[best_key]

        is_fallback = best_score <= 0.0
        if is_fallback:
            best_key = self._geometric_pick_key(faces)

        # Mặt đang giữ (self._current) đã biến mất khỏi faces frame này (occlusion,
        # mất bám...) -> hysteresis không còn ý nghĩa (không thể "đợi" 1 khoá đã chết).
        # Buộc chuyển ngay sang best_key thay vì rơi xuống mặt đầu tiên tùy tiện.
        if self._current is not None and self._face_by_key(faces, self._current) is None:
            self._current = None
            self._dwell = 0
            self._candidate = None
            self._candidate_count = 0

        chosen_key = self._apply_hysteresis(best_key, scores)
        face = self._face_by_key(faces, chosen_key)
        if face is None:  # phòng vệ — không nên xảy ra sau khi resync ở trên
            face = max(faces, key=self._geometric_score)
            self._current = self._face_key(face)
        return SpeakerDecision(face.track_id, float(face.cx), float(face.cy),
                               float(scores.get(self._face_key(face), 0.0)), vad, is_fallback)

    # ---- nội bộ ----
    @staticmethod
    def _face_key(f: FaceBox):
        """Khoá định danh trong 1 lần update(): track_id nếu có, ngược lại id(f)."""
        return f.track_id if f.track_id is not None else id(f)

    def _score_faces(self, frame, faces, vad, frame_idx) -> dict:
        scores: dict = {}
        if self.detector is not None and self.audio_provider is not None:
            crops = [self._face_crop(frame, f) for f in faces]
            chunk, sr = self.audio_provider(frame_idx)
            sc = self.detector.score_speaking(crops, chunk, sr)
            for f, s in zip(faces, sc):
                scores[self._face_key(f)] = float(s)
            return scores
        voiced = vad >= self.c.vad_threshold
        for f in faces:
            key = self._face_key(f)
            roi = self._mouth_crop(frame, f)
            lip = self._lip.update(key, roi, frame_idx)
            scores[key] = float(vad * lip) if voiced else 0.0
        return scores

    def _mouth_crop(self, frame, face):
        if frame is None:
            return None
        x, y, w, h = mouth_roi(face, self.src_w, self.src_h, top_frac=self.c.mouth_y0,
                               height_frac=self.c.mouth_y1 - self.c.mouth_y0)
        return frame[y:y + h, x:x + w]

    def _face_crop(self, frame, face):
        if frame is None:
            return None
        x = max(0, face.x)
        y = max(0, face.y)
        return frame[y:y + face.h, x:x + face.w]

    def _geometric_score(self, f: FaceBox):
        cx0, cy0 = self.src_w / 2.0, self.src_h / 2.0
        return (f.w * f.h, -((f.cx - cx0) ** 2 + (f.cy - cy0) ** 2))

    def _geometric_pick_key(self, faces):
        return self._face_key(max(faces, key=self._geometric_score))

    def _face_by_key(self, faces, key):
        for f in faces:
            if self._face_key(f) == key:
                return f
        return None

    def _apply_hysteresis(self, best_key, scores):
        if self._current is None:
            self._current = best_key
            self._dwell = 1
            self._candidate = None
            self._candidate_count = 0
            return self._current
        if best_key == self._current:
            self._dwell += 1
            self._candidate = None
            self._candidate_count = 0
            return self._current
        # best_key khác current.
        if self._dwell < self.c.min_dwell_frames:
            self._dwell += 1
            return self._current
        if best_key == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = best_key
            self._candidate_count = 1
        cur_score = scores.get(self._current, 0.0)
        best_score = scores.get(best_key, 0.0)
        beats = best_score > cur_score * self.c.hysteresis_margin
        if beats and self._candidate_count >= self.c.switch_confirm_frames:
            self._current = best_key
            self._dwell = 1
            self._candidate = None
            self._candidate_count = 0
            return self._current
        self._dwell += 1
        return self._current


def build_center_provider(decisions: list[SpeakerDecision], default_center):
    """Provider tâm thô (chưa làm mượt) — chủ yếu để debug; pipeline dùng camera_path."""

    def provider(i: int):
        if not decisions:
            return default_center
        j = min(max(i, 0), len(decisions) - 1)
        return (decisions[j].cx, decisions[j].cy)

    return provider
