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


def _install_crash_logging() -> None:
    """Ghi MỌI exception chưa bắt (main + luồng nền) ra log — nếu không, ở bản .exe
    windowed một lỗi trong slot Qt sẽ đóng cửa sổ mà KHÔNG để lại dấu vết."""
    import threading
    import traceback

    clog = logging.getLogger("khunghinh")

    def _hook(exc_type, exc, tb) -> None:  # noqa: ANN001
        clog.critical("Lỗi CHƯA BẮT (main):\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = _hook

    def _thook(args) -> None:  # noqa: ANN001
        name = getattr(args.thread, "name", "?")
        clog.critical("Lỗi CHƯA BẮT (luồng %s):\n%s", name,
                      "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

    threading.excepthook = _thook


def _selftest(vid: str, log: logging.Logger) -> int:
    """Chạy ĐÚNG ĐƯỜNG GUI (QApplication + MainWindow hiện + nhập video + phân tích
    QThread thật + vẽ preview auto & nền mờ) để tái hiện crash TRONG bản .exe.
    Dùng: `KhungHinh916.exe --selftest <video>`. Kết quả ghi ra log + in stdout."""
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication

    from .core.reframe_engine import ReframeParams
    from .mediaio.reader import VideoReader
    from .ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    log.info("SELFTEST (GUI) bắt đầu: %s", vid)
    try:
        cfg = AppConfig()
        w = MainWindow(cfg)
        w.show()
        app.processEvents()

        reader = VideoReader(vid)
        info = reader.open()
        frame = reader.read_at(0)
        w._reader, w._info = reader, info
        w._params = ReframeParams(info.width, info.height, cfg.target_aspect, 1.0, 1.0)
        w._center_px = w._params.default_center_px()
        w.scrub.setRange(0, max(0, info.frame_count - 1))
        w.preview.set_frame(frame)
        w.controls.set_video_info(info)
        w._mode = "auto"
        w._redraw_for_frame(0)
        app.processEvents()
        log.info("SELFTEST: nhập + preview OK")

        w.on_analyze()
        loop = QEventLoop()
        if w._analysis_worker is not None:
            w._analysis_worker.finished_ok.connect(lambda _x: loop.quit())
            w._analysis_worker.failed.connect(lambda _m: loop.quit())
        QTimer.singleShot(120000, loop.quit)
        loop.exec()
        log.info("SELFTEST: phân tích xong (analysis=%s)", w._analysis is not None)

        w._redraw_for_frame(min(3, info.frame_count - 1))       # vẽ auto overlay
        w.on_background_mode_changed(True)                        # nền mờ compose
        w._redraw_for_frame(min(3, info.frame_count - 1))
        app.processEvents()
        reader.release()
        log.info("SELFTEST GUI OK")
        print("SELFTEST OK", flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001
        log.critical("SELFTEST CRASH: %s", exc, exc_info=True)
        print(f"SELFTEST CRASH: {exc}", flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    setup_logging(level=logging.INFO)
    _install_crash_logging()
    log = logging.getLogger("khunghinh")

    args = list(argv if argv is not None else sys.argv)
    if "--selftest" in args:
        idx = args.index("--selftest")
        vid = args[idx + 1] if idx + 1 < len(args) else ""
        return _selftest(vid, log)

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
