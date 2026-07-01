"""Cấu hình logging tập trung: console (gọn) + file xoay vòng (chi tiết để debug)."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> Path:
    """Khởi tạo logging toàn ứng dụng. Idempotent (gọi nhiều lần an toàn).

    - Console: theo `level` (mặc định INFO).
    - File `logs/khunghinh.log`: luôn DEBUG, xoay vòng 2MB x 3 file.
    Trả về đường dẫn file log.
    """
    global _CONFIGURED
    log_dir = log_dir or (Path.cwd() / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "khunghinh.log"

    if _CONFIGURED:
        return log_file

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Bảo đảm console in được tiếng Việt (Windows mặc định cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(level)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging đã khởi tạo → %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
