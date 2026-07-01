"""IoU tracker (ByteTrack-lite) — thuần stdlib + FaceBox, dễ kiểm thử (không cần cv2).

Gán track_id nguyên ổn định cho từng khuôn mặt qua các frame để ASD tích lũy được
lịch sử chuyển động môi theo từng người. Ghép 2 tầng (high/low score) như ByteTrack.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..detection.base import FaceBox


def iou(a: FaceBox, b: FaceBox) -> float:
    """IoU hai hộp; 0.0 nếu không chồng lấn hoặc diện tích <= 0."""
    if a.w <= 0 or a.h <= 0 or b.w <= 0 or b.h <= 0:
        return 0.0
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: FaceBox
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative | confirmed | dead

    def predict(self) -> FaceBox:
        return self.box  # seam cho Kalman: hiện là identity

    def mark_matched(self, det: FaceBox) -> None:
        self.box = det
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1


def _greedy_match(tracks: list[Track], dets: list[FaceBox], thresh: float):
    """Ghép tham lam theo IoU. Trả về (matches[(ti,di)], unmatched_track_idx, unmatched_det_idx)."""
    cand = []
    for ti, t in enumerate(tracks):
        pt = t.predict()
        for di, d in enumerate(dets):
            iv = iou(pt, d)
            if iv >= thresh:
                cand.append((iv, ti, di))
    # Tie-break tất định: IoU giảm dần, track_id tăng, det_idx tăng.
    cand.sort(key=lambda c: (-c[0], tracks[c[1]].track_id, c[2]))
    mt: set[int] = set()
    md: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _iv, ti, di in cand:
        if ti in mt or di in md:
            continue
        matches.append((ti, di))
        mt.add(ti)
        md.add(di)
    um_t = [ti for ti in range(len(tracks)) if ti not in mt]
    um_d = [di for di in range(len(dets)) if di not in md]
    return matches, um_t, um_d


class IouTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        high_score_thresh: float = 0.6,
        low_score_thresh: float = 0.1,
        match_low_iou_threshold: float = 0.2,
        return_unconfirmed: bool = False,
    ):
        if not (0 < iou_threshold <= 1):
            raise ValueError("iou_threshold phải thuộc (0, 1]")
        if max_age < 1:
            raise ValueError("max_age phải >= 1")
        if min_hits < 1:
            raise ValueError("min_hits phải >= 1")
        if low_score_thresh > high_score_thresh:
            raise ValueError("low_score_thresh phải <= high_score_thresh")
        if not (0 <= match_low_iou_threshold <= 1):
            raise ValueError("match_low_iou_threshold phải thuộc [0, 1]")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.match_low_iou_threshold = match_low_iou_threshold
        self.return_unconfirmed = return_unconfirmed

        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame_count = 0

    @property
    def tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.state != "dead"]

    def reset(self, hard: bool = False) -> None:
        self._tracks = []
        self._frame_count = 0
        if hard:
            self._next_id = 1

    def update(self, detections: list[FaceBox]) -> list[FaceBox]:
        self._frame_count += 1
        for t in self._tracks:
            t.age += 1

        dets = [d for d in detections if d.w > 0 and d.h > 0]
        high = [d for d in dets if d.score >= self.high_score_thresh]
        # Tầng "low" PHẢI có sàn dưới — dưới low_score_thresh là nhiễu, loại hẳn
        # (không dùng để giải cứu track nào), đúng nguyên lý 3 tầng của ByteTrack.
        low = [d for d in dets if self.low_score_thresh <= d.score < self.high_score_thresh]
        active = [t for t in self._tracks if t.state != "dead"]

        # Tầng 1: track vs detection điểm cao.
        m1, ut1, _ud1_unused = _greedy_match(active, high, self.iou_threshold)
        matched_high_dets: set[int] = set()
        for ti, di in m1:
            active[ti].mark_matched(high[di])
            matched_high_dets.add(di)

        # Tầng 2: track còn lại vs detection điểm thấp (IoU thấp hơn).
        rem_tracks = [active[ti] for ti in ut1]
        m2, _ut2, _ud2 = _greedy_match(rem_tracks, low, self.match_low_iou_threshold)
        matched_rem: set[int] = set()
        for ti, di in m2:
            rem_tracks[ti].mark_matched(low[di])
            matched_rem.add(ti)

        for idx, t in enumerate(rem_tracks):
            if idx not in matched_rem:
                t.mark_missed()

        # Sinh track mới CHỈ từ detection điểm cao chưa khớp (ByteTrack).
        for di, d in enumerate(high):
            if di in matched_high_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks.append(Track(track_id=tid, box=replace(d, track_id=tid)))

        # Chuyển trạng thái + khai tử.
        for t in active:
            if t.time_since_update == 0 and t.state == "tentative" and t.hits >= self.min_hits:
                t.state = "confirmed"
            if t.time_since_update > self.max_age:
                t.state = "dead"
        self._tracks = [t for t in self._tracks if t.state != "dead"]

        out: list[FaceBox] = []
        for t in self._tracks:
            if t.time_since_update != 0:
                continue
            if t.state == "confirmed" or self.return_unconfirmed:
                out.append(replace(t.box, track_id=t.track_id))
        out.sort(key=lambda b: b.track_id if b.track_id is not None else 0)
        return out
