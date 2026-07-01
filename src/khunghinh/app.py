"""Bootstrap ứng dụng: cấu hình logging, nạp config, mở cửa sổ chính."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import AppConfig
from .logging_setup import setup_logging


def _icon_path() -> str:
    """Tìm icon.ico (dev ở gốc dự án, hoặc trong gói PyInstaller qua _MEIPASS)."""
    bases = [getattr(sys, "_MEIPASS", None), str(Path.cwd()), str(Path(__file__).resolve().parents[2])]
    for base in bases:
        if base:
            p = Path(base) / "icon.ico"
            if p.is_file():
                return str(p)
    return ""


def main(argv: list[str] | None = None) -> int:
    setup_logging(level=logging.INFO)
    log = logging.getLogger("khunghinh")

    config = AppConfig.load(Path.cwd() / "config.json")

    # Import Qt sau khi logging sẵn sàng để bắt lỗi import rõ ràng.
    from PyQt6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("KhungHinh916")
    icon_path = _icon_path()
    if icon_path:
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))
        log.info("Đã nạp icon: %s", icon_path)

    window = MainWindow(config)
    window.show()
    log.info("Ứng dụng đã khởi động (%dx%d đích).", config.target_width, config.target_height)
    return app.exec()
