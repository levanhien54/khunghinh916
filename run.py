#!/usr/bin/env python3
"""Điểm khởi chạy đơn giản cho KhungHinh916.

Cho phép chạy trực tiếp `python run.py` mà không cần `pip install -e .`.
"""
from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

# Bản .exe windowed: sys.stdout/stderr = None → tránh crash nếu 1 thư viện lỡ ghi ra.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# faulthandler: bắt crash NATIVE (segfault từ cv2/YuNet/DNN — Python excepthook KHÔNG
# bắt được) → ghi C-stack ra logs/crash_native.log. Giúp truy nguyên "tự đóng không log".
try:
    _logdir = Path.cwd() / "logs"
    _logdir.mkdir(parents=True, exist_ok=True)
    _fh = open(_logdir / "crash_native.log", "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=_fh, all_threads=True)
except Exception:  # noqa: BLE001
    pass

# Cho phép import package `khunghinh` từ thư mục src/ khi chạy trực tiếp.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from khunghinh.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
