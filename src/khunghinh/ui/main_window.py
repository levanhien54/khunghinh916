"""Cửa sổ chính: kết nối preview + bảng điều khiển + phân tích auto + luồng xuất."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..core.analysis_result import AnalysisResult
from ..core.geometry import CropRect, compute_crop_rect
from ..core.reframe_engine import ReframeEngine, ReframeParams
from ..core.smoothing import CameraSmoother
from ..mediaio.exporter import ExportSettings, VideoExporter
from ..mediaio.reader import VideoInfo, VideoReader
from .analysis_worker import AnalysisWorker
from .control_panel import ControlPanel
from .preview_view import PreviewView
from .workers import ExportWorker

log = logging.getLogger(__name__)

VIDEO_FILTER = "Video (*.mp4 *.mov *.mkv *.avi *.webm);;Tất cả (*.*)"


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("KhungHinh916 — Reframe video sang dọc 9:16 (TikTok)")
        self.resize(1240, 800)

        self._reader: VideoReader | None = None
        self._info: VideoInfo | None = None
        self._params: ReframeParams | None = None
        self._center_px: tuple[float, float] | None = None
        self._mode = "manual"
        self._cur_frame = 0
        self._blur_bg = False

        self._analysis: AnalysisResult | None = None
        self._analysis_worker: AnalysisWorker | None = None
        self._export_worker: ExportWorker | None = None
        self._progress: QProgressDialog | None = None

        self.preview = PreviewView()
        self.controls = ControlPanel(config.zoom_min, config.zoom_max, config.zoom_default)

        # Khu vực trái: preview + thanh tua frame.
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.preview, 1)
        scrub_row = QHBoxLayout()
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setEnabled(False)
        self.lbl_frame = QLabel("frame 0")
        self.lbl_frame.setMinimumWidth(96)
        scrub_row.addWidget(self.scrub, 1)
        scrub_row.addWidget(self.lbl_frame)
        lv.addLayout(scrub_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(8, 8, 8, 8)
        hl.addWidget(self.controls)
        splitter.addWidget(holder)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([860, 380])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Hãy nhập một video để bắt đầu.")

        self.controls.importRequested.connect(self.on_import)
        self.controls.exportRequested.connect(self.on_export)
        self.controls.zoomXChanged.connect(self.on_zoom_x)
        self.controls.zoomYChanged.connect(self.on_zoom_y)
        self.controls.resetRequested.connect(self.on_reset)
        self.controls.modeChanged.connect(self.on_mode_changed)
        self.controls.analyzeRequested.connect(self.on_analyze)
        self.controls.backgroundModeChanged.connect(self.on_background_mode_changed)
        self.preview.cropCenterChanged.connect(self.on_crop_dragged)
        self.scrub.valueChanged.connect(self.on_scrub)

    # --------------------------- Quản lý worker nền ------------------------
    def _stop_worker(self, worker, label: str, timeout_ms: int = 3000) -> None:
        """Hủy 1 worker nền và chờ nó dừng; nếu hết thời gian, ngắt kết nối signal
        để tránh signal bắn vào widget/khung dữ liệu đã/đang bị thay thế hoặc đóng."""
        if not worker or not worker.isRunning():
            return
        worker.cancel()
        if not worker.wait(timeout_ms):
            log.warning("%s không dừng kịp trong %dms — ngắt kết nối signal.", label, timeout_ms)
            for sig_name in ("progress", "stage", "finished_ok", "failed", "finished"):
                sig = getattr(worker, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except TypeError:
                        pass  # không còn kết nối nào để ngắt

    # ----------------------------- Nhập video -----------------------------
    def on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", VIDEO_FILTER)
        if not path:
            return
        # Nếu đang phân tích video cũ, hủy & chờ trước khi đổi video — tránh kết
        # quả phân tích của video cũ "lọt" vào trạng thái của video mới sau này.
        self._stop_worker(self._analysis_worker, "Phân tích")
        self._analysis_worker = None
        try:
            reader = VideoReader(path)
            info = reader.open()
            frame = reader.read_at(0)
            if frame is None:
                raise IOError("Không đọc được frame đầu tiên.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Mở video lỗi")
            QMessageBox.critical(self, "Lỗi mở video", str(exc))
            return

        if self._reader:
            self._reader.release()
        self._reader, self._info = reader, info
        self._analysis = None  # video mới → bỏ cache phân tích
        self.controls.set_analyzed(False)

        self._params = ReframeParams(
            src_w=info.width,
            src_h=info.height,
            target_aspect=self.config.target_aspect,
            zoom_x=self.controls.sld_zoom_x.value(),
            zoom_y=self.controls.sld_zoom_y.value(),
        )
        self._center_px = self._params.default_center_px()
        self._cur_frame = 0

        self.scrub.setEnabled(True)
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, max(0, info.frame_count - 1))
        self.scrub.setValue(0)
        self.scrub.blockSignals(False)

        self.preview.set_frame(frame)
        self.preview.set_auto_mode(self._mode == "auto")
        self.controls.set_video_info(info)
        self._redraw_for_frame(0)
        self.statusBar().showMessage(f"Đã nhập: {Path(path).name}")

    # --------------------------- Vẽ lại theo frame ------------------------
    def _crop_for_center(self, cx: float, cy: float) -> CropRect:
        assert self._params is not None
        return compute_crop_rect(
            self._params.src_w, self._params.src_h, self._params.target_aspect,
            cx, cy, self._params.zoom_x, self._params.zoom_y,
        )

    def _redraw_for_frame(self, i: int) -> None:
        if not self._reader or not self._params:
            return
        frame = self._reader.read_at(i)
        self.lbl_frame.setText(f"frame {i}")
        if frame is None:
            return

        # Nền mờ chỉ ảnh hưởng bước XUẤT video (xem VideoExporter.run) — preview
        # tương tác luôn hiển thị khung cắt giống Thủ công/Tự động, tránh dựng lại
        # composite tốn CPU mỗi lần tua frame.
        self.preview.set_frame(frame)
        if self._mode == "auto" and self._analysis is not None:
            faces, active = self._analysis.faces_for_frame(i)
            cx, cy = self._analysis.centers_for_frame(i)
            self.preview.set_faces(faces, active)
            self.preview.set_auto_crop_rect(self._crop_for_center(cx, cy))
        else:
            cx, cy = self._center_px or self._params.default_center_px()
            self.preview.set_crop_rect(self._crop_for_center(cx, cy))

    def _refresh_crop(self) -> None:
        if self._mode == "auto" and self._analysis is not None:
            self._redraw_for_frame(self._cur_frame)
            return
        if not self._params or not self._center_px:
            return
        rect = self._crop_for_center(*self._center_px)
        self._center_px = (rect.cx, rect.cy)  # giữ tâm đã kẹp để kéo không trôi
        self.preview.set_crop_rect(rect)

    def on_scrub(self, i: int) -> None:
        self._cur_frame = i
        self._redraw_for_frame(i)

    def on_zoom_x(self, v: float) -> None:
        if self._params:
            self._params.zoom_x = v
            self._refresh_crop()

    def on_zoom_y(self, v: float) -> None:
        if self._params:
            self._params.zoom_y = v
            self._refresh_crop()

    def on_crop_dragged(self, x: float, y: float) -> None:
        if self._params and self._mode == "manual":
            self._center_px = (x, y)
            self._refresh_crop()

    def on_reset(self) -> None:
        if self._params:
            self._center_px = self._params.default_center_px()
            self._refresh_crop()

    def on_background_mode_changed(self, is_blur: bool) -> None:
        self._blur_bg = is_blur
        self.statusBar().showMessage(
            "Nền mờ: bật (lớp nền an toàn khi xuất)." if is_blur else "Nền mờ: tắt."
        )

    def on_mode_changed(self, mode: str) -> None:
        self._mode = mode
        self.preview.set_auto_mode(mode == "auto")
        if mode == "manual":
            self.preview.clear_faces()
        self._redraw_for_frame(self._cur_frame)
        self.statusBar().showMessage(
            "Chế độ Tự động — bấm 'Phân tích video'." if mode == "auto" else "Chế độ Thủ công."
        )

    # ------------------------------ Phân tích -----------------------------
    def on_analyze(self) -> None:
        if not self._info or not self._params:
            return
        if self._analysis_worker and self._analysis_worker.isRunning():
            return
        self.statusBar().showMessage("Đang phân tích…")
        worker = AnalysisWorker(self._info.path, self._params.target_aspect, self.config)
        worker.progress.connect(self.controls.set_analysis_progress)
        worker.stage.connect(self.controls.set_stage)
        worker.finished_ok.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        self._analysis_worker = worker
        worker.start()

    def _on_analysis_done(self, result: object) -> None:
        # Phòng vệ thêm (ngoài việc on_import đã hủy worker cũ): chỉ chấp nhận kết
        # quả khớp đúng video/tỉ lệ đích hiện tại, loại bỏ kết quả "trễ" của video trước.
        if not self._info or not self._params:
            return
        expected_fp = f"{self._info.width}x{self._info.height}|{self._params.target_aspect:.5f}"
        if getattr(result, "params_fingerprint", None) != expected_fp:
            log.warning("Bỏ qua kết quả phân tích không khớp video hiện tại.")
            return
        self._analysis = result  # AnalysisResult
        self.controls.set_analyzed(True)
        self._redraw_for_frame(self._cur_frame)
        n_cuts = len(self._analysis.scene_cut_frames) if self._analysis else 0
        self.statusBar().showMessage(f"Phân tích xong ({n_cuts} cảnh). Sẵn sàng xuất tự động.")

    def _on_analysis_failed(self, msg: str) -> None:
        self.controls.set_analyzed(False)
        if msg != "Đã hủy.":
            QMessageBox.warning(self, "Phân tích thất bại", msg)
        self.statusBar().showMessage(f"Phân tích dừng: {msg}")

    # ------------------------------- Xuất ---------------------------------
    def on_export(self) -> None:
        if not self._info or not self._params or not self._center_px:
            return
        if self._export_worker and self._export_worker.isRunning():
            return
        if self._mode == "auto" and self._analysis is None:
            QMessageBox.information(self, "Cần phân tích", "Hãy bấm 'Phân tích video' trước khi xuất tự động.")
            return

        src = Path(self._info.path)
        default_name = str(src.with_name(src.stem + "_9x16.mp4"))
        out_path, _ = QFileDialog.getSaveFileName(self, "Lưu video 9:16", default_name, "MP4 (*.mp4)")
        if not out_path:
            return

        export_reader = VideoReader(self._info.path)
        export_reader.open()
        engine = ReframeEngine(
            ReframeParams(
                src_w=self._params.src_w,
                src_h=self._params.src_h,
                target_aspect=self._params.target_aspect,
                zoom_x=self._params.zoom_x,
                zoom_y=self._params.zoom_y,
                center_x_norm=self._center_px[0] / self._params.src_w,
                center_y_norm=self._center_px[1] / self._params.src_h,
            ),
            CameraSmoother(self.config.smoothing_min_cutoff, self.config.smoothing_beta),
        )
        settings = ExportSettings(
            out_path=out_path,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            crf=self.config.export_crf,
            codec=self.config.export_codec,
            preset=self.config.export_preset,
            blurred_background=self._blur_bg,
            bg_blur_downscale_divisor=self.config.bg_blur_downscale_divisor,
            bg_blur_dim=self.config.bg_blur_dim,
        )
        exporter = VideoExporter(export_reader, engine, settings)

        if self._mode == "auto" and self._analysis is not None:
            center_provider = self._analysis.make_center_provider()
            smooth = False  # quỹ đạo đã được làm mượt ở Pass 2 — KHÔNG làm mượt lần nữa
        else:
            center_provider = None
            smooth = True

        self._progress = QProgressDialog("Đang xuất video…", "Hủy", 0, 100, self)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setAutoClose(True)
        self._progress.setMinimumDuration(0)

        self._export_worker = ExportWorker(exporter, center_provider=center_provider, smooth=smooth)
        self._export_worker.progress.connect(self._on_progress)
        self._export_worker.finished_ok.connect(self._on_export_done)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.finished.connect(export_reader.release)
        self._progress.canceled.connect(self._export_worker.cancel)
        self._export_worker.start()
        self.statusBar().showMessage("Đang xuất…")

    def _on_progress(self, i: int, n: int) -> None:
        if self._progress and n:
            self._progress.setValue(int(i * 100 / n))

    def _on_export_done(self, path: str) -> None:
        if self._progress:
            self._progress.setValue(100)
        QMessageBox.information(self, "Hoàn tất", f"Đã xuất:\n{path}")
        self.statusBar().showMessage(f"Đã xuất: {path}")

    def _on_export_failed(self, msg: str) -> None:
        if self._progress:
            self._progress.cancel()
        if msg != "Đã hủy.":
            QMessageBox.warning(self, "Xuất thất bại", msg)
        self.statusBar().showMessage(f"Xuất dừng: {msg}")

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._stop_worker(self._analysis_worker, "Phân tích")
        self._stop_worker(self._export_worker, "Xuất video")
        if self._reader:
            self._reader.release()
        super().closeEvent(event)
