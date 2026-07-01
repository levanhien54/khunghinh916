from __future__ import annotations

import math

from khunghinh.core.geometry import base_crop_size, compute_crop_rect

TARGET = 9 / 16


def test_base_crop_landscape_limited_by_height():
    w, h = base_crop_size(1920, 1080, TARGET)
    assert math.isclose(h, 1080, rel_tol=1e-6)
    assert math.isclose(w / h, TARGET, rel_tol=1e-6)
    assert w <= 1920


def test_base_crop_portrait_limited_by_width():
    w, h = base_crop_size(720, 1280, TARGET)
    assert w <= 720
    assert math.isclose(w / h, TARGET, rel_tol=1e-6)


def test_crop_within_bounds_when_centered():
    r = compute_crop_rect(1920, 1080, TARGET, 960, 540, 1.0, 1.0)
    assert r.x >= 0 and r.y >= 0
    assert r.x + r.width <= 1920
    assert r.y + r.height <= 1080


def test_crop_clamped_at_top_left_edge():
    r = compute_crop_rect(1920, 1080, TARGET, 0, 0, 1.0, 1.0)
    assert r.x == 0 and r.y == 0


def test_crop_clamped_at_bottom_right_edge():
    r = compute_crop_rect(1920, 1080, TARGET, 5000, 5000, 1.0, 1.0)
    assert r.x + r.width <= 1920
    assert r.y + r.height <= 1080


def test_zoom_in_shrinks_crop():
    r1 = compute_crop_rect(1920, 1080, TARGET, 960, 540, 1.0, 1.0)
    r2 = compute_crop_rect(1920, 1080, TARGET, 960, 540, 2.0, 2.0)
    assert r2.width < r1.width
    assert r2.height < r1.height


def test_independent_axis_zoom():
    base_w, base_h = base_crop_size(1920, 1080, TARGET)
    r = compute_crop_rect(1920, 1080, TARGET, 960, 540, 2.0, 1.0)
    assert math.isclose(r.width, base_w / 2, abs_tol=2)
    assert math.isclose(r.height, min(base_h, 1080), abs_tol=2)


def test_invalid_source_raises():
    import pytest

    with pytest.raises(ValueError):
        base_crop_size(0, 100, TARGET)
