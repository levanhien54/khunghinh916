"""Luồng nền (QThread) để xuất video — không treo GUI, có tiến độ + hủy."""
from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal

from ..mediaio.exporter import VideoExporter

log = logging.getLogger(__name__)


class ExportWorker(QThread):
    progress = pyqtSignal(int, int)     # (đã xử lý, tổng)
    finished_ok = pyqtSignal(str)       # đường dẫn file xuất
    failed = pyqtSignal(str)            # thông báo lỗi / "Đã hủy."

    def __init__(self, exporter: VideoExporter, center_provider=None, smooth: bool = True, parent=None):
        super().__init__(parent)
        self._exporter = exporter
        self._center_provider = center_provider
        self._smooth = smooth
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            out = self._exporter.run(
                center_provider=self._center_provider,
                smooth=self._smooth,
                progress_cb=lambda i, n: self.progress.emit(i, n),
                cancel_cb=lambda: self._cancel,
            )
            self.finished_ok.emit(out)
        except InterruptedError:
            self.failed.emit("Đã hủy.")
        except Exception as exc:  # noqa: BLE001
            log.exception("Xuất video lỗi")
            self.failed.emit(str(exc))
