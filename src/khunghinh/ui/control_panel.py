"""Bảng điều khiển bên phải: nhập video, thông tin video, zoom X/Y, nút xuất."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class _ZoomSlider(QWidget):
    """Slider zoom dạng float (bước 0.01) kèm nhãn giá trị."""

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, vmin: float, vmax: float, vdef: float):
        super().__init__()
        self._scale = 100.0
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        name = QLabel(label)
        name.setMinimumWidth(54)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(vmin * self._scale), int(vmax * self._scale))
        self._slider.setValue(int(vdef * self._scale))
        self._val = QLabel(f"{vdef:.2f}×")
        self._val.setMinimumWidth(46)
        lay.addWidget(name)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._val)
        self._slider.valueChanged.connect(self._on_change)

    def _on_change(self, raw: int) -> None:
        v = raw / self._scale
        self._val.setText(f"{v:.2f}×")
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self._slider.value() / self._scale

    def set_value(self, v: float) -> None:
        self._slider.setValue(int(v * self._scale))


class ControlPanel(QWidget):
    importRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    zoomXChanged = pyqtSignal(float)
    zoomYChanged = pyqtSignal(float)
    resetRequested = pyqtSignal()
    modeChanged = pyqtSignal(str)        # "manual" | "auto"
    analyzeRequested = pyqtSignal()
    backgroundModeChanged = pyqtSignal(bool)  # True = bật nền mờ (compose thủ công)
    fgScaleChanged = pyqtSignal(float)   # cỡ video A trên nền mờ
    fgResetRequested = pyqtSignal()      # reset cỡ về 1.0

    def __init__(self, zoom_min: float, zoom_max: float, zoom_def: float,
                 fg_min: float = 0.3, fg_max: float = 3.0, fg_def: float = 1.0, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self._has_video = False
        root = QVBoxLayout(self)

        # --- Nhập video ---
        self.btn_import = QPushButton("📂  Nhập video…")
        root.addWidget(self.btn_import)

        # --- Chế độ ---
        mode_box = QGroupBox("Chế độ")
        mlay = QVBoxLayout(mode_box)
        self.rad_manual = QRadioButton("Thủ công (kéo khung)")
        self.rad_auto = QRadioButton("Tự động bám người nói")
        self.rad_manual.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.rad_manual)
        self._mode_group.addButton(self.rad_auto)
        self.btn_analyze = QPushButton("🔍  Phân tích video")
        self.btn_analyze.setEnabled(False)
        self.prog_analyze = QProgressBar()
        self.prog_analyze.setVisible(False)
        self.lbl_stage = QLabel("")
        self.lbl_stage.setStyleSheet("color: #6cf; font-size: 11px;")
        mlay.addWidget(self.rad_manual)
        mlay.addWidget(self.rad_auto)
        mlay.addWidget(self.btn_analyze)
        mlay.addWidget(self.prog_analyze)
        mlay.addWidget(self.lbl_stage)
        root.addWidget(mode_box)

        # --- Thông tin video ---
        info_box = QGroupBox("Thông tin video")
        form = QFormLayout(info_box)
        self.lbl_resolution = QLabel("—")
        self.lbl_aspect = QLabel("—")
        self.lbl_fps = QLabel("—")
        self.lbl_duration = QLabel("—")
        form.addRow("Độ phân giải:", self.lbl_resolution)
        form.addRow("Tỉ lệ gốc:", self.lbl_aspect)
        form.addRow("FPS:", self.lbl_fps)
        form.addRow("Thời lượng:", self.lbl_duration)
        root.addWidget(info_box)

        # --- Khung cắt 9:16 + zoom theo trục (chỉ dùng khi KHÔNG bật Nền mờ) ---
        self.zoom_box = QGroupBox("Khung cắt 9:16 — Zoom theo trục")
        zlay = QVBoxLayout(self.zoom_box)
        self.chk_link = QCheckBox("Khoá Zoom X = Y (giữ tỉ lệ, không méo)")
        self.chk_link.setChecked(True)
        self.sld_zoom_x = _ZoomSlider("Zoom X", zoom_min, zoom_max, zoom_def)
        self.sld_zoom_y = _ZoomSlider("Zoom Y", zoom_min, zoom_max, zoom_def)
        hint = QLabel("Mẹo: kéo khung trong vùng xem trước để chọn tâm.")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        hint.setWordWrap(True)
        self.btn_reset = QPushButton("Đặt lại khung")
        zlay.addWidget(self.chk_link)
        zlay.addWidget(self.sld_zoom_x)
        zlay.addWidget(self.sld_zoom_y)
        zlay.addWidget(hint)
        zlay.addWidget(self.btn_reset)
        root.addWidget(self.zoom_box)

        # --- Nền mờ (compose thủ công: video A trên nền mờ) ---
        bg_box = QGroupBox("Nền")
        blay = QVBoxLayout(bg_box)
        self.chk_blur_bg = QCheckBox("Nền mờ — đặt video A lên nền mờ (kiểu CapCut)")
        bg_hint = QLabel(
            "Video A (nguyên khung) đặt lên nền mờ của chính nó; kéo góc trong ô xem "
            "trước hoặc dùng slider để chỉnh cỡ; vị trí tự bám người nói (chế độ Tự động)."
        )
        bg_hint.setStyleSheet("color: #888; font-size: 11px;")
        bg_hint.setWordWrap(True)
        self.sld_fg_scale = _ZoomSlider("Cỡ video A", fg_min, fg_max, fg_def)
        self.btn_fg_reset = QPushButton("Đặt lại cỡ")
        self.sld_fg_scale.setVisible(False)
        self.btn_fg_reset.setVisible(False)
        blay.addWidget(self.chk_blur_bg)
        blay.addWidget(bg_hint)
        blay.addWidget(self.sld_fg_scale)
        blay.addWidget(self.btn_fg_reset)
        root.addWidget(bg_box)

        root.addStretch(1)

        # --- Xuất ---
        self.btn_export = QPushButton("💾  Xuất video 9:16…")
        self.btn_export.setEnabled(False)
        root.addWidget(self.btn_export)

        # --- Wiring ---
        self.btn_import.clicked.connect(self.importRequested)
        self.btn_export.clicked.connect(self.exportRequested)
        self.btn_reset.clicked.connect(self.resetRequested)
        self.btn_analyze.clicked.connect(self.analyzeRequested)
        self.rad_manual.toggled.connect(self._on_mode_toggle)
        self.sld_zoom_x.valueChanged.connect(self._on_zoom_x)
        self.sld_zoom_y.valueChanged.connect(self._on_zoom_y)
        self.chk_blur_bg.toggled.connect(self._on_blur_bg_toggle)
        self.sld_fg_scale.valueChanged.connect(self.fgScaleChanged)
        self.btn_fg_reset.clicked.connect(self.fgResetRequested)

    # --- Nền mờ ---
    def _on_blur_bg_toggle(self, checked: bool) -> None:
        is_blur = bool(checked)
        # Nền mờ = compose (video A lên nền mờ) → KHÔNG dùng khung cắt 9:16/Zoom X-Y.
        # Ẩn cả nhóm crop, chỉ hiện điều khiển cỡ video A. Chế độ Thủ công/Tự động vẫn
        # giữ (quyết định vị trí compose: tĩnh giữa hay bám người nói).
        self.zoom_box.setVisible(not is_blur)
        self.sld_fg_scale.setVisible(is_blur)
        self.btn_fg_reset.setVisible(is_blur)
        self.backgroundModeChanged.emit(is_blur)

    def set_fg_scale(self, v: float) -> None:
        """Đồng bộ slider khi cỡ đổi từ nơi khác (gizmo) — không phát lại vòng lặp."""
        self.sld_fg_scale.blockSignals(True)
        self.sld_fg_scale.set_value(v)
        self.sld_fg_scale.blockSignals(False)

    def is_blur_background(self) -> bool:
        return self.chk_blur_bg.isChecked()

    # --- Chế độ / phân tích ---
    def _on_mode_toggle(self, _checked: bool) -> None:
        self._refresh_analyze_enabled()
        self.modeChanged.emit("manual" if self.rad_manual.isChecked() else "auto")

    def _refresh_analyze_enabled(self) -> None:
        self.btn_analyze.setEnabled(self._has_video and self.rad_auto.isChecked())

    def set_mode(self, mode: str) -> None:
        (self.rad_manual if mode == "manual" else self.rad_auto).setChecked(True)

    def set_analysis_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.prog_analyze.setRange(0, total)
            self.prog_analyze.setValue(done)
            self.prog_analyze.setVisible(0 < done < total)

    def set_stage(self, text: str) -> None:
        self.lbl_stage.setText(text)

    def set_analyzed(self, ok: bool) -> None:
        self.prog_analyze.setVisible(False)
        self.lbl_stage.setText("✓ Đã phân tích — sẵn sàng xuất." if ok else "")

    def _on_zoom_x(self, v: float) -> None:
        if self.chk_link.isChecked():
            self.sld_zoom_y.blockSignals(True)
            self.sld_zoom_y.set_value(v)
            self.sld_zoom_y.blockSignals(False)
            self.zoomYChanged.emit(v)
        self.zoomXChanged.emit(v)

    def _on_zoom_y(self, v: float) -> None:
        if self.chk_link.isChecked():
            self.sld_zoom_x.blockSignals(True)
            self.sld_zoom_x.set_value(v)
            self.sld_zoom_x.blockSignals(False)
            self.zoomXChanged.emit(v)
        self.zoomYChanged.emit(v)

    def set_video_info(self, info) -> None:  # noqa: ANN001
        self.lbl_resolution.setText(f"{info.width} × {info.height}")
        self.lbl_aspect.setText(info.aspect_label())
        self.lbl_fps.setText(f"{info.fps:.2f}")
        self.lbl_duration.setText(f"{info.duration_sec:.1f} s")
        self.btn_export.setEnabled(True)
        self._has_video = True
        self._refresh_analyze_enabled()
