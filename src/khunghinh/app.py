"""Bootstrap ứng dụng: cấu hình logging, nạp config, mở cửa sổ chính."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import AppConfig
from .logging_setup import setup_logging


def main(argv: list[str] | None = None) -> int:
    setup_logging(level=logging.INFO)
    log = logging.getLogger("khunghinh")

    config = AppConfig.load(Path.cwd() / "config.json")

    # Import Qt sau khi logging sẵn sàng để bắt lỗi import rõ ràng.
    from PyQt6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("KhungHinh916")

    window = MainWindow(config)
    window.show()
    log.info("Ứng dụng đã khởi động (%dx%d đích).", config.target_width, config.target_height)
    return app.exec()
