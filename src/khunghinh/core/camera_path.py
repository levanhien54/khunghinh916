"""Pass 2 — dựng quỹ đạo camera ảo mượt từ chuỗi tâm người nói thô.

Sở hữu DUY NHẤT việc làm mượt trong pipeline auto: One Euro (CameraSmoother) +
dead-zone + snap tại cắt cảnh + settle khi đổi người nói. Đầu ra là chuỗi tâm đã
kẹp, dùng làm `center_provider` cho VideoExporter (BẮT BUỘC export với smooth=False).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import base_crop_size, compute_crop_rect
from .reframe_engine import ReframeParams
from .smoothing import CameraSmoother


@dataclass
class SceneCutParams:
    luma_weight: float = 0.4
    hist_weight: float = 0.6
    threshold: float = 0.35
    min_scene_len_frames: int = 12
    adaptive: bool = True
    adaptive_k: float = 3.0


@dataclass
class CameraPathParams:
    deadzone_frac_x: float = 0.06
    deadzone_frac_y: float = 0.08
    min_cutoff: float = 0.6
    beta: float = 0.04
    d_cutoff: float = 1.0
    settle_frames: int = 9
    snap_reset_on_cut: bool = True
    # Recenter: sau khi mục tiêu ở NGOÀI dead-zone liên tục N frame → bỏ offset để về
    # đúng tâm. TẮT mặc định (0) vì đo cho thấy nó tạo răng cưa ~dz mỗi N frame khi pan
    # (giật); lệch tâm ~6% do dead-zone thì nhỏ hơn nhiều. Bật (>0) nếu chấp nhận giật.
    recenter_frames: int = 0


@dataclass
class AutoZoomParams:
    enabled: bool = True
    single_face_zoom: float = 1.25
    multi_face_zoom: float = 1.0
    min_stable_frames: int = 20
    manual_override: float | None = None


@dataclass(frozen=True)
class CameraPath:
    centers: np.ndarray          # (N, 2) float32
    zooms: np.ndarray | None
    cut_frames: list
    n_frames: int

    def center_at(self, i: int) -> tuple[float, float]:
        if self.n_frames == 0:
            return (0.0, 0.0)
        j = min(max(i, 0), self.n_frames - 1)
        return (float(self.centers[j, 0]), float(self.centers[j, 1]))


def detect_scene_cuts(luma, hist_diffs, params: SceneCutParams | None = None) -> list[int]:
    """Trả về danh sách frame là điểm cắt cảnh (gồm 0 khi có ít nhất 1 frame; [] nếu không có frame nào)."""
    p = params or SceneCutParams()
    luma = np.asarray(luma, dtype=np.float64).ravel()
    hist = np.asarray(hist_diffs, dtype=np.float64).ravel()
    n = luma.size
    if n == 0:
        return []

    luma_diff = np.zeros(n)
    if n > 1:
        luma_diff[1:] = np.abs(np.diff(luma)) / 255.0
    combined = p.luma_weight * luma_diff + p.hist_weight * hist
    combined[0] = 0.0

    thr = p.threshold
    if p.adaptive and n > 2:
        body = combined[1:]
        med = float(np.median(body))
        mad = float(np.median(np.abs(body - med)))
        thr = max(p.threshold, med + p.adaptive_k * mad)

    cuts = [0]
    last = 0
    for i in range(1, n):
        if combined[i] > thr and (i - last) >= p.min_scene_len_frames:
            cuts.append(i)
            last = i
    return cuts


def build_camera_path(
    raw_centers,
    chosen_track,
    fps: float,
    params: ReframeParams,
    path_params: CameraPathParams | None = None,
    cut_frames=None,
) -> CameraPath:
    """Làm mượt chuỗi tâm thô thành quỹ đạo camera. Tâm tính với zoom=1.0 (độc lập zoom)."""
    pp = path_params or CameraPathParams()
    raw = np.asarray(raw_centers, dtype=np.float64).reshape(-1, 2)
    n = len(raw)
    cut_set = set(cut_frames or [0])
    cut_set.add(0)
    if n == 0:
        return CameraPath(np.zeros((0, 2), np.float32), None, sorted(cut_set), 0)

    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    base_w, base_h = base_crop_size(params.src_w, params.src_h, params.target_aspect)
    dz_x = pp.deadzone_frac_x * base_w
    dz_y = pp.deadzone_frac_y * base_h

    chosen = list(chosen_track) if chosen_track is not None else [-1] * n
    sm = CameraSmoother(pp.min_cutoff, pp.beta, pp.d_cutoff)

    out = np.zeros((n, 2), dtype=np.float64)
    last_finite = np.array([params.src_w / 2.0, params.src_h / 2.0])
    held = last_finite.copy()
    prev_track = None
    settle_left = 0
    recenter = int(pp.recenter_frames)
    streak_x = streak_y = 0  # số frame liên tiếp mục tiêu ở ngoài dead-zone (mỗi trục)

    for i in range(n):
        tx, ty = raw[i]
        is_gap = not (np.isfinite(tx) and np.isfinite(ty))
        if is_gap:
            tx, ty = last_finite
        else:
            last_finite = np.array([tx, ty])

        cur_track = chosen[i] if i < len(chosen) else -1
        is_cut = i in cut_set

        if is_cut and pp.snap_reset_on_cut:
            sm.reset()
            held = np.array([tx, ty])
            sx, sy = tx, ty
            settle_left = 0
            streak_x = streak_y = 0
        else:
            # Settle: tạm "đứng yên" vài frame khi vừa đổi người nói.
            if (prev_track is not None and cur_track != prev_track
                    and cur_track >= 0 and prev_track >= 0):
                settle_left = pp.settle_frames
            holding = settle_left > 0
            if settle_left > 0:
                settle_left -= 1

            # Frame GAP (không có mặt → tx,ty = last_finite đóng băng): GIỮ nguyên
            # held + streak, KHÔNG chạy dead-zone/recenter — nếu không, sai số dấu
            # phẩy động khiến |tx-held| > dz_x lặp lại mỗi frame gap và recenter sẽ
            # "phát minh" chuyển động camera đúng lúc không có mặt để bám.
            if not holding and not is_gap:
                # Dead-zone + recenter. Mục tiêu ngoài vùng chết LIÊN TỤC >= recenter
                # frame (chuyển động THỰC, không phải jitter) ⇒ BÁM THẲNG tâm (held=tx)
                # và GIỮ streak cao để tiếp tục bám mượt — tránh răng cưa "snap-rồi-
                # reset" gây giật (~dz mỗi N frame). Còn trong "nghi ngờ jitter" thì
                # kẹp dead-zone. Vào lại vùng chết (đứng yên) ⇒ reset streak.
                if abs(tx - held[0]) > dz_x:
                    streak_x += 1
                    if recenter and streak_x >= recenter:
                        held[0] = tx
                    else:
                        held[0] = tx - np.sign(tx - held[0]) * dz_x
                else:
                    streak_x = 0
                if abs(ty - held[1]) > dz_y:
                    streak_y += 1
                    if recenter and streak_y >= recenter:
                        held[1] = ty
                    else:
                        held[1] = ty - np.sign(ty - held[1]) * dz_y
                else:
                    streak_y = 0
            sx, sy = sm.smooth(held[0], held[1], dt)

        prev_track = cur_track
        rect = compute_crop_rect(params.src_w, params.src_h, params.target_aspect,
                                 sx, sy, params.zoom_x, params.zoom_y)
        out[i, 0] = rect.cx
        out[i, 1] = rect.cy

    out = out.astype(np.float32)
    if not np.isfinite(out).all():  # pragma: no cover - phòng vệ
        out = np.nan_to_num(out, nan=float(params.src_w / 2.0))
    return CameraPath(out, None, sorted(cut_set), n)


def make_center_provider(path: CameraPath):
    """Provider tâm cho VideoExporter. DÙNG VỚI exporter smooth=False."""
    centers = path.centers
    n = path.n_frames

    def provider(i: int):
        if n == 0:
            return (0.0, 0.0)
        j = min(max(i, 0), n - 1)
        return (float(centers[j, 0]), float(centers[j, 1]))

    return provider


def suggest_zoom(chosen_track, face_counts, cut_frames=None, zp: AutoZoomParams | None = None) -> np.ndarray:
    """Gợi ý zoom theo frame: sát hơn khi 1 mặt, rộng khi nhiều mặt.

    Có debounce theo `min_stable_frames` (chỉ đổi mức zoom sau khi số mặt giữ
    nguyên đủ lâu, tránh nhấp nháy khi detector thoáng mất/bắt lại 1 mặt), và
    snap tức thì (bỏ qua debounce) tại các điểm cắt cảnh trong `cut_frames`.
    `chosen_track` hiện chưa ảnh hưởng tới quyết định zoom (zoom chỉ phụ thuộc
    số mặt nhìn thấy), được giữ trong chữ ký để tương thích API/dùng trong tương lai.
    """
    zp = zp or AutoZoomParams()
    fc = np.asarray(face_counts)
    n = len(fc)
    if zp.manual_override is not None:
        return np.full(n, float(zp.manual_override), dtype=np.float32)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    cut_set = set(cut_frames or [])
    raw = np.where(fc <= 1, zp.single_face_zoom, zp.multi_face_zoom).astype(np.float32)

    out = np.empty(n, dtype=np.float32)
    current = raw[0]
    candidate = raw[0]
    stable_count = 1
    out[0] = current
    for i in range(1, n):
        if i in cut_set:
            current = candidate = raw[i]
            stable_count = 1
            out[i] = current
            continue
        if raw[i] == candidate:
            stable_count += 1
        else:
            candidate = raw[i]
            stable_count = 1
        if candidate != current and stable_count >= zp.min_stable_frames:
            current = candidate
        out[i] = current
    return out
