#!/usr/bin/env python3
"""Điểm khởi chạy đơn giản cho KhungHinh916.

Cho phép chạy trực tiếp `python run.py` mà không cần `pip install -e .`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Cho phép import package `khunghinh` từ thư mục src/ khi chạy trực tiếp.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from khunghinh.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
