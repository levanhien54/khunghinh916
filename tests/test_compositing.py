from __future__ import annotations

import numpy as np
import pytest

from khunghinh.core.compositing import (
    composite_crop_on_blurred_background,
    crop_and_resize,
    fit_dimensions,
    make_blurred_background,
)
from khunghinh.core.geometry import CropRect


def test_fit_dimensions_landscape_into_portrait():
    fw, fh, xo, yo = fit_dimensions(1920, 1080, 1080, 1920)
    assert fw == 1080
    assert abs(fh - 608) <= 2
    assert xo == 0
    assert yo == (1920 - fh) // 2


def test_fit_dimensions_matching_aspect_fills_exactly():
    fw, fh, xo, yo = fit_dimensions(1080, 1920, 1080, 1920)
    assert fw == 1080 and fh == 1920
    assert xo == 0 and yo == 0


def test_fit_dimensions_centered_within_bounds():
    fw, fh, xo, yo = fit_dimensions(1920, 1080, 1080, 1920)
    assert xo >= 0 and yo >= 0
    assert xo + fw <= 1080
    assert yo + fh <= 1920
    assert yo == pytest.approx((1920 - fh) / 2, abs=1)


def test_fit_dimensions_invalid_raises():
    with pytest.raises(ValueError):
        fit_dimensions(0, 100, 100, 100)
    with pytest.raises(ValueError):
        fit_dimensions(100, 100, 0, 100)
    with pytest.raises(ValueError):
        fit_dimensions(100, 100, 100, -5)


def test_blurred_background_shape():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bg = make_blurred_background(frame, 1080, 1920)
    assert bg.shape == (1920, 1080, 3)
    assert bg.dtype == np.uint8


def test_blurred_background_reduces_variance():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8).astype(np.uint8)
    bg = make_blurred_background(noisy, 1080, 1920, downscale_divisor=32, dim=1.0)
    assert bg.astype(np.float32).std() < noisy.astype(np.float32).std() * 0.5


def test_blurred_background_dim_reduces_brightness():
    frame = np.full((200, 300, 3), 200, dtype=np.uint8)
    bright = make_blurred_background(frame, 100, 100, dim=1.0)
    dim = make_blurred_background(frame, 100, 100, dim=0.5)
    assert dim.mean() < bright.mean() * 0.6


def test_blurred_background_small_divisor_still_works():
    frame = np.full((40, 40, 3), 128, dtype=np.uint8)
    bg = make_blurred_background(frame, 64, 64, downscale_divisor=64)
    assert bg.shape == (64, 64, 3)


def test_crop_and_resize_matches_target_size():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    rect = CropRect(x=200, y=100, width=800, height=600)
    out = crop_and_resize(frame, rect, 400, 300)
    assert out.shape == (300, 400, 3)
    assert out.dtype == np.uint8


def test_crop_and_resize_same_size_rect_is_identity():
    frame = np.full((100, 100, 3), 77, dtype=np.uint8)
    rect = CropRect(x=0, y=0, width=100, height=100)
    out = crop_and_resize(frame, rect, 100, 100)
    assert np.array_equal(out, frame)


def test_crop_and_resize_empty_rect_falls_back_to_full_frame():
    frame = np.full((50, 50, 3), 42, dtype=np.uint8)
    rect = CropRect(x=200, y=200, width=10, height=10)  # hoàn toàn ngoài khung
    out = crop_and_resize(frame, rect, 50, 50)
    assert np.array_equal(out, frame)


def test_composite_crop_on_blurred_background_matches_plain_crop():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    rect = CropRect(x=300, y=0, width=1080, height=1080)  # vùng zoom lệch tâm
    expected = crop_and_resize(frame, rect, 1080, 1920)
    out = composite_crop_on_blurred_background(frame, rect, 1080, 1920)
    assert np.array_equal(out, expected)


def test_composite_crop_on_blurred_background_shape_and_dtype():
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    rect = CropRect(x=0, y=0, width=720, height=720)
    out = composite_crop_on_blurred_background(
        frame, rect, 1080, 1920, downscale_divisor=16, dim=0.4
    )
    assert out.shape == (1920, 1080, 3)
    assert out.dtype == np.uint8


def test_crop_and_resize_uses_cubic_when_upscaling(monkeypatch):
    import khunghinh.core.compositing as comp
    seen = {}
    real = comp.cv2.resize

    def spy(src, dsize, interpolation, **kw):
        seen["interp"] = interpolation
        return real(src, dsize, interpolation=interpolation, **kw)

    monkeypatch.setattr(comp.cv2, "resize", spy)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    rect = CropRect(x=656, y=0, width=608, height=1080)  # crop 608x1080 -> 1080x1920 = PHÓNG TO
    comp.crop_and_resize(frame, rect, 1080, 1920)
    assert seen["interp"] == comp.cv2.INTER_CUBIC


def test_crop_and_resize_uses_area_when_downscaling(monkeypatch):
    import khunghinh.core.compositing as comp
    seen = {}
    real = comp.cv2.resize

    def spy(src, dsize, interpolation, **kw):
        seen["interp"] = interpolation
        return real(src, dsize, interpolation=interpolation, **kw)

    monkeypatch.setattr(comp.cv2, "resize", spy)
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)  # 4K nguồn
    rect = CropRect(x=0, y=0, width=2160, height=2160)  # crop 2160 -> 1080 = THU NHỎ
    comp.crop_and_resize(frame, rect, 1080, 1080)
    assert seen["interp"] == comp.cv2.INTER_AREA


def test_composite_crop_skips_blur_compute(monkeypatch):
    # Nền mờ luôn bị crop đè kín → không được tốn CPU dựng nền mờ.
    import khunghinh.core.compositing as comp

    def _boom(*a, **k):
        raise AssertionError("make_blurred_background KHÔNG được gọi (đã short-circuit)")

    monkeypatch.setattr(comp, "make_blurred_background", _boom)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rect = CropRect(x=50, y=0, width=270, height=480)
    out = comp.composite_crop_on_blurred_background(frame, rect, 1080, 1920)
    assert np.array_equal(out, comp.crop_and_resize(frame, rect, 1080, 1920))
