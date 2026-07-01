"""Luồng nền phân tích auto-reframe: detect → track → ASD → quỹ đạo camera.

Một vòng giải mã (Pass 1) thu: khuôn mặt/track mỗi frame, tâm người nói thô, đặc
trưng cảnh. Tính VAD toàn clip quanh đó. Pass 2 dựng quỹ đạo camera mượt. Phát
AnalysisResult qua signal (chỉ giao tiếp qua signal, không đụng widget).
"""
from __future__ import annotations

import logging
import queue
import threading

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
            detector = build_face_detector(cfg)
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

            raw_centers: list[tuple[float, float]] = []
            chosen: list[int] = []
            faces_pf: list = []
            lumas: list[float] = []
            diffs: list[float] = []
            face_counts: list[int] = []
            prev_hist = None

            detect_stride = max(1, int(cfg.detect_stride))
            last_faces: list = []

            reader.rewind()
            i = 0
            total = max(1, info.frame_count)
            last_reset_at = -cfg.cut_min_scene_len  # cho phép reset ngay tại frame 0 nếu cần

            # Producer đọc frame (giải mã) song song với consumer (detect/track/ASD),
            # giấu thời gian giải mã sau tính toán. 1 producer + 1 consumer + hàng đợi
            # FIFO có chặn ⇒ thứ tự frame bất biến ⇒ kết quả không đổi. cv2.VideoCapture
            # chỉ được chạm bởi DUY NHẤT producer (không an toàn đa luồng).
            frame_q: "queue.Queue" = queue.Queue(maxsize=16)  # đệm giải mã rộng hơn để overlap mượt
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
                    # Sentinel EOF PHẢI tới được consumer: dùng put có chặn (không
                    # put_nowait) để không bị rớt khi hàng đợi đang đầy → tránh
                    # consumer treo vĩnh viễn ở get(). Thoát sớm nếu đang dừng.
                    while not stop_evt.is_set():
                        try:
                            frame_q.put(None, timeout=0.1)
                            break
                        except queue.Full:
                            continue

            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()
            try:
                while True:
                    if self._cancel:
                        raise InterruptedError()
                    frame = frame_q.get()
                    if frame is None:
                        break

                    # Detect theo stride: quỹ đạo camera đã làm mượt mạnh ở Pass 2 nên
                    # bỏ bớt detect ít ảnh hưởng; giữa các frame "coast" bằng cách tái
                    # dùng khuôn mặt gần nhất. VAD/môi/cắt-cảnh vẫn tính MỖI frame.
                    if i % detect_stride == 0:
                        last_faces = tracker.update(detector.detect(frame))
                    faces = last_faces
                    luma, diff, prev_hist = compute_features_step(prev_hist, frame, cfg.analysis_downscale_width)
                    # Gợi ý cắt cảnh inline (Pass1) → reset lịch sử môi để không lấy
                    # chuyển động "qua cắt cảnh" làm tín hiệu. Có debounce min_scene_len
                    # để tránh reset liên tục mỗi frame trên cảnh quay rung/nhiễu — điểm
                    # cắt CHÍNH XÁC dùng cho camera path vẫn là detect_scene_cuts (Pass2).
                    if i > 0 and diff > cfg.cut_threshold and (i - last_reset_at) >= cfg.cut_min_scene_len:
                        selector.reset()
                        last_reset_at = i

                    vad_i = float(vad[min(i, len(vad) - 1)])
                    dec = selector.update(frame, faces, vad_i, i)

                    # Không có mặt nào để bám → báo GAP (NaN) để camera_path GIỮ vị
                    # trí cũ, thay vì giật khung về giữa trên các dropout detect ngắn.
                    if faces:
                        raw_centers.append((dec.cx, dec.cy))
                    else:
                        raw_centers.append((float("nan"), float("nan")))
                    chosen.append(dec.track_id if dec.track_id is not None else -1)
                    faces_pf.append(faces)
                    lumas.append(luma)
                    diffs.append(diff)
                    face_counts.append(len(faces))

                    i += 1
                    if i % PROGRESS_EVERY == 0:
                        self.progress.emit(i, max(total, i))
            finally:
                stop_evt.set()
                # rút cạn hàng đợi để producer (nếu đang chặn ở put) thoát được
                try:
                    while True:
                        frame_q.get_nowait()
                except queue.Empty:
                    pass
                # Join KHÔNG timeout: read_next() luôn trả về nên producer chắc chắn
                # kết thúc; phải đợi nó thoát HẲN trước khi outer-finally gọi
                # reader.release() (cv2.VideoCapture không an toàn đa luồng — nếu
                # release() chạy khi producer còn trong read_next() có thể crash native).
                producer.join()
            if producer_exc:
                raise producer_exc[0]

            n = i
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
