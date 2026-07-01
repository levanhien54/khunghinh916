"""Luồng nền phân tích auto-reframe: detect → track → ASD → quỹ đạo camera.

Một vòng giải mã (Pass 1) thu: khuôn mặt/track mỗi frame, tâm người nói thô, đặc
trưng cảnh. Tính VAD toàn clip quanh đó. Pass 2 dựng quỹ đạo camera mượt. Phát
AnalysisResult qua signal (chỉ giao tiếp qua signal, không đụng widget).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..audio_vad import FrameVad, VadParams, extract_audio
from ..config import AppConfig
from ..core.analysis_result import AnalysisResult
from ..core.camera_path import CameraPathParams, SceneCutParams, build_camera_path, detect_scene_cuts
from ..core.reframe_engine import ReframeParams
from ..core.scene_features import compute_features_step
from ..detection.asd_fusion import ASDConfig, ActiveSpeakerSelector
from ..detection.factory import build_face_detector
from ..mediaio.reader import VideoReader
from ..tracking.iou_tracker import IouTracker

log = logging.getLogger(__name__)

PROGRESS_EVERY = 5
_DECODE_BATCH = 32   # số frame giải mã mỗi lô trước khi detect song song


def _resolve_detect_workers(cfg) -> int:  # noqa: ANN001
    """Số luồng detect song song. 0 = auto = min(16, số core - 1)."""
    w = int(getattr(cfg, "analysis_detect_workers", 0) or 0)
    if w > 0:
        return w
    return min(16, max(1, (os.cpu_count() or 4) - 1))


class _Pass1State:
    """Trạng thái + bộ tích luỹ của Pass 1 (giữ tuần tự, dù detect chạy song song)."""

    def __init__(self, tracker, selector, last_reset_at: int):  # noqa: ANN001
        self.tracker = tracker
        self.selector = selector
        self.prev_hist = None
        self.last_faces: list = []
        self.last_reset_at = last_reset_at
        self.raw_centers: list = []
        self.chosen: list = []
        self.faces_pf: list = []
        self.lumas: list = []
        self.diffs: list = []
        self.face_counts: list = []


class AnalysisWorker(QThread):
    progress = pyqtSignal(int, int)
    stage = pyqtSignal(str)
    finished_ok = pyqtSignal(object)   # AnalysisResult
    failed = pyqtSignal(str)

    def __init__(self, video_path: str, target_aspect: float, config: AppConfig, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.target_aspect = target_aspect
        self.config = config
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _consume_frame(self, st: "_Pass1State", frame, dets, luma: float, hist, i: int, cfg, vad) -> None:  # noqa: ANN001
        """Xử lý stateful 1 frame (TUẦN TỰ theo thứ tự): track (nếu có dets mới) →
        diff histogram (cần prev) → cắt-cảnh inline → ASD → tích luỹ. `luma`/`hist`
        đã tính SẴN ở pool (stateless); dets=None ⇒ coast mặt cũ (frame bỏ qua stride)."""
        if dets is not None:
            st.last_faces = st.tracker.update(dets)
        faces = st.last_faces
        # Bhattacharyya so với hist frame trước (phần DUY NHẤT của đặc trưng cảnh cần
        # trạng thái tuần tự — nên nằm ở đây; phần tính hist nặng đã chạy song song).
        diff = 0.0 if st.prev_hist is None else float(cv2.compareHist(st.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        st.prev_hist = hist
        if i > 0 and diff > cfg.cut_threshold and (i - st.last_reset_at) >= cfg.cut_min_scene_len:
            st.selector.reset()
            st.last_reset_at = i
        vad_i = float(vad[min(i, len(vad) - 1)])
        dec = st.selector.update(frame, faces, vad_i, i)
        # Không có mặt để bám → GAP (NaN) để camera_path GIỮ vị trí (không giật về giữa).
        if faces:
            st.raw_centers.append((dec.cx, dec.cy))
        else:
            st.raw_centers.append((float("nan"), float("nan")))
        st.chosen.append(dec.track_id if dec.track_id is not None else -1)
        st.faces_pf.append(faces)
        st.lumas.append(luma)
        st.diffs.append(diff)
        st.face_counts.append(len(faces))

    def run(self) -> None:
        try:
            result = self._analyze()
            self.finished_ok.emit(result)
        except InterruptedError:
            self.failed.emit("Đã hủy.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Phân tích lỗi")
            self.failed.emit(str(exc))

    def _analyze(self) -> AnalysisResult:
        cfg = self.config
        reader = VideoReader(self.video_path)
        info = reader.open()
        try:
            self.stage.emit("Phân tích âm thanh (VAD)…")
            # cv2 CAP_PROP_FRAME_COUNT có thể sai (VFR, container lỗi metadata) — cỡ
            # mảng VAD theo THỜI LƯỢNG AUDIO THỰC TẾ (độc lập với số đếm của cv2) để
            # các frame cuối trên video VFR không bị "đóng băng" vào 1 giá trị VAD cũ.
            audio_track = extract_audio(self.video_path)
            if self._cancel:
                raise InterruptedError()
            if audio_track is not None and info.fps > 0:
                vad_len = max(1, info.frame_count, int(audio_track.duration_sec * info.fps) + 1)
                vad = FrameVad(VadParams()).compute_track(audio_track, info.fps, vad_len)
            else:
                vad = np.ones(max(1, info.frame_count), dtype=np.float32)

            self.stage.emit("Phát hiện & bám khuôn mặt + người nói…")
            tracker = IouTracker(high_score_thresh=cfg.face_score_threshold)
            selector = ActiveSpeakerSelector(
                info.width, info.height,
                ASDConfig(
                    vad_threshold=cfg.asd_vad_threshold,
                    min_dwell_frames=cfg.asd_min_dwell_frames,
                    switch_confirm_frames=cfg.asd_switch_confirm_frames,
                    hysteresis_margin=cfg.asd_hysteresis_margin,
                ),
            )
            detect_stride = max(1, int(cfg.detect_stride))
            workers = _resolve_detect_workers(cfg)

            # SONG SONG HOÁ DETECTION (nút nghẽn ~71%): detector.detect stateless từng
            # frame và Haar/YuNet NHẢ GIL ⇒ thread-pool chạy detect nhiều frame song
            # song (mỗi luồng 1 detector riêng qua thread-local; cv2.setNumThreads(1)
            # tránh nạp chồng luồng nội bộ). CHỈ main thread chạm reader (giải mã theo
            # lô) ⇒ an toàn cv2.VideoCapture. Phần stateful (track/features/ASD/cắt-cảnh)
            # chạy TUẦN TỰ đúng thứ tự frame ⇒ kết quả TƯƠNG ĐƯƠNG bản đơn luồng.
            tls = threading.local()
            fwidth = cfg.analysis_downscale_width

            def _pool_task(args):  # noqa: ANN001, ANN202
                """Chạy SONG SONG mỗi frame: detect (nếu tới mốc stride) + hist đặc
                trưng cảnh — cả hai stateless, nặng, nhả GIL. Trả (dets|None, luma, hist)."""
                frame, do_detect = args
                dets = None
                if do_detect:
                    d = getattr(tls, "det", None)
                    if d is None:
                        d = build_face_detector(cfg)
                        tls.det = d
                    dets = d.detect(frame)
                luma, _diff, hist = compute_features_step(None, frame, fwidth)
                return dets, luma, hist

            st = _Pass1State(tracker, selector, -cfg.cut_min_scene_len)
            reader.rewind()
            i = 0
            total = max(1, info.frame_count)
            prev_cv2_threads = cv2.getNumThreads()
            pool = ThreadPoolExecutor(max_workers=workers) if workers >= 2 else None
            if pool is not None:
                cv2.setNumThreads(1)

            # OVERLAP GIẢI MÃ: 1 producer sở hữu reader, giải mã frame vào hàng đợi
            # song song với main (pool detect+hist → consume). Reader CHỈ producer chạm
            # (an toàn cv2). Hàng đợi FIFO có chặn ⇒ thứ tự frame BẤT BIẾN ⇒ vẫn
            # byte-identical. maxsize giới hạn số frame trong luồng (chặn bộ nhớ).
            frame_q: "queue.Queue" = queue.Queue(maxsize=2 * _DECODE_BATCH)
            stop_evt = threading.Event()
            producer_exc: list = []

            def _produce() -> None:
                try:
                    while not stop_evt.is_set():
                        fr = reader.read_next()
                        if fr is None:
                            break
                        while not stop_evt.is_set():
                            try:
                                frame_q.put(fr, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                except Exception as exc:  # noqa: BLE001
                    producer_exc.append(exc)
                finally:
                    while not stop_evt.is_set():
                        try:
                            frame_q.put(None, timeout=0.1)  # sentinel EOF (chặn để không rớt)
                            break
                        except queue.Full:
                            continue

            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()
            try:
                eof = False
                while not eof:
                    if self._cancel:
                        raise InterruptedError()
                    batch = []
                    while len(batch) < _DECODE_BATCH:
                        fr = frame_q.get()
                        if fr is None:
                            eof = True
                            break
                        batch.append(fr)
                    if not batch:
                        break

                    # detect chỉ ở mốc stride (frame khác coast); hist tính MỖI frame.
                    tasks = [(batch[bi], (i + bi) % detect_stride == 0) for bi in range(len(batch))]
                    if pool is not None:
                        results = list(pool.map(_pool_task, tasks))
                    else:
                        results = [_pool_task(t) for t in tasks]

                    for bi, (dets, luma, hist) in enumerate(results):
                        self._consume_frame(st, batch[bi], dets, luma, hist, i + bi, cfg, vad)
                    i += len(batch)
                    self.progress.emit(min(i, total), max(total, i))
            finally:
                stop_evt.set()
                # rút cạn để producer (nếu đang chặn ở put) thoát được
                try:
                    while True:
                        frame_q.get_nowait()
                except queue.Empty:
                    pass
                # Join KHÔNG timeout: read_next() luôn trả về ⇒ producer chắc chắn kết
                # thúc; phải thoát HẲN trước khi outer-finally gọi reader.release()
                # (cv2.VideoCapture không an toàn đa luồng).
                producer.join()
                if pool is not None:
                    pool.shutdown(wait=True)
                    cv2.setNumThreads(prev_cv2_threads)
            if producer_exc:
                raise producer_exc[0]

            n = i
            raw_centers = st.raw_centers
            chosen = st.chosen
            faces_pf = st.faces_pf
            lumas = st.lumas
            diffs = st.diffs
            self.stage.emit("Tính quỹ đạo camera…")
            cut_frames = detect_scene_cuts(
                np.array(lumas), np.array(diffs),
                SceneCutParams(threshold=cfg.cut_threshold, min_scene_len_frames=cfg.cut_min_scene_len),
            )
            rp = ReframeParams(info.width, info.height, self.target_aspect, 1.0, 1.0)
            if n > 0:
                path = build_camera_path(
                    np.array(raw_centers, dtype=float).reshape(-1, 2),
                    chosen, info.fps, rp,
                    CameraPathParams(
                        min_cutoff=cfg.path_min_cutoff, beta=cfg.path_beta,
                        deadzone_frac_x=cfg.path_deadzone_frac_x, deadzone_frac_y=cfg.path_deadzone_frac_y,
                        settle_frames=cfg.path_settle_frames,
                        recenter_frames=cfg.path_recenter_frames,
                    ),
                    cut_frames=cut_frames,
                )
                centers = path.centers
            else:
                centers = np.zeros((0, 2), np.float32)

            result = AnalysisResult(
                frame_count=n,
                fps=info.fps,
                src_w=info.width,
                src_h=info.height,
                centers_px=centers,
                faces_per_frame=faces_pf,
                active_track_per_frame=np.array(chosen, np.int32),
                scene_cut_frames=np.array(cut_frames, np.int32),
                params_fingerprint=f"{info.width}x{info.height}|{self.target_aspect:.5f}",
            )
            self.progress.emit(n, max(total, n))
            log.info("Phân tích xong: %d frame, %d cắt cảnh.", n, len(cut_frames))
            return result
        finally:
            reader.release()
