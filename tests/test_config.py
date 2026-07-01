from __future__ import annotations

from khunghinh.config import AppConfig


def test_detect_stride_default_is_one():
    assert AppConfig().detect_stride == 1


def test_export_preset_default():
    assert AppConfig().export_preset == "veryfast"
