"""Ghép khung hình lên NỀN MỜ — lớp nền an toàn phía dưới khung hình đã cắt.

Kỹ thuật làm mờ rẻ trên CPU: thu nhỏ ảnh xuống rất nhỏ (~1/32) rồi phóng to lại.
Việc nội suy khi phóng to từ ảnh cực nhỏ tạo hiệu ứng mờ mịn, nhanh hơn nhiều so
với Gaussian blur trực tiếp trên ảnh lớn — phù hợp xử lý offline trên CPU-only.

Pipeline: NỀN = bản sao toàn khung hình gốc, cắt theo tỉ lệ đích rồi phóng to phủ
kín 1080×1920 (qua bước thu nhỏ-mờ ở trên). FOREGROUND = khung hình đã được
reframe engine cắt theo tâm/zoom (thủ công hoặc bám người nói), resize phủ KÍN
khung đích — luôn đè hoàn toàn lên nền mờ. Nền mờ vì vậy gần như không bao giờ
hiển thị được trong kết quả cuối; nó tồn tại như một lớp nền an toàn/kiến trúc,
không phải để tạo viền mờ nhìn thấy (đây là lựa chọn thiết kế đã xác nhận, xem
docs/superpowers/specs/2026-07-01-blur-background-reframe-design.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import CropRect, compute_crop_rect


def fit_dimensions(src_w: int, src_h: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Co ảnh để vừa TRỌN (contain) trong khung đích, giữ tỉ lệ, căn giữa.

    Trả về (fit_w, fit_h, x_offset, y_offset).
    """
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        raise ValueError("Kích thước phải > 0")
    scale = min(target_w / src_w, target_h / src_h)
    fit_w = min(target_w, max(1, round(src_w * scale)))
    fit_h = min(target_h, max(1, round(src_h * scale)))
    x_off = (target_w - fit_w) // 2
    y_off = (target_h - fit_h) // 2
    return fit_w, fit_h, x_off, y_off


def make_blurred_background(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    downscale_divisor: int = 32,
    dim: float = 0.55,
) -> np.ndarray:
    """Nền mờ phủ KÍN khung đích: thu nhỏ ~1/divisor rồi phóng to lại (mờ rẻ trên CPU).

    `dim` (0..1) làm tối nền để khung hình chính nổi bật hơn khi đặt lên trên.
    """
    h, w = frame.shape[:2]
    divisor = max(1, int(downscale_divisor))
    small_w = max(1, round(w / divisor))
    small_h = max(1, round(h / divisor))
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Cắt theo tỉ lệ đích trên ảnh đã thu nhỏ trước khi phóng to để PHỦ KÍN
    # khung đích mà không bị méo hình (cùng logic crop-to-fill của core/geometry).
    target_aspect = target_w / target_h
    rect = compute_crop_rect(small_w, small_h, target_aspect, small_w / 2.0, small_h / 2.0)
    cropped = small[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]
    if cropped.size == 0:
        cropped = small

    # Làm tối (dim) TRÊN ẢNH NHỎ trước khi phóng to: rẻ hơn nhiều (vài trăm px thay
    # vì ~2M px của canvas đầy) và convertScaleAbs chạy C++ 1 lượt, không cấp phát
    # buffer float32. resize tuyến tính là phép tuyến tính nên
    # resize(cropped*dim) ≡ resize(cropped)*dim (chỉ khác làm tròn — vô hình vì mờ).
    if dim != 1.0:
        cropped = cv2.convertScaleAbs(cropped, alpha=dim)
    bg = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return bg


def crop_and_resize(frame: np.ndarray, rect: CropRect, target_w: int, target_h: int) -> np.ndarray:
    """Cắt `frame` theo `rect` rồi resize phủ kín (target_w, target_h).

    Chọn nội suy theo HƯỚNG: thu nhỏ → INTER_AREA (khử răng cưa tốt nhất); phóng to
    → INTER_CUBIC (nét hơn INTER_AREA vốn chỉ tối ưu cho thu nhỏ). Nguồn landscape
    vào khung 9:16 hầu như luôn là PHÓNG TO (vd. crop 608×1080 → 1080×1920 ~1.78×),
    nên đây là điểm cải thiện độ nét ở gần như mọi frame xuất.
    """
    h, w = frame.shape[:2]
    x1, y1 = max(0, rect.x), max(0, rect.y)
    x2, y2 = min(rect.x + rect.width, w), min(rect.y + rect.height, h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame
    ch, cw = cropped.shape[:2]
    upscaling = target_w > cw or target_h > ch
    interp = cv2.INTER_CUBIC if upscaling else cv2.INTER_AREA
    return cv2.resize(cropped, (target_w, target_h), interpolation=interp)


def composite_crop_on_blurred_background(
    frame: np.ndarray,
    rect: CropRect,
    target_w: int,
    target_h: int,
    downscale_divisor: int = 32,
    dim: float = 0.55,
) -> np.ndarray:
    """Trả về khung hình đã cắt phủ kín canvas.

    Vì `crop_and_resize` luôn resize đúng (target_w, target_h) — phủ kín 100% canvas
    ở tỉ lệ 9:16 — nên lớp nền mờ (nếu dựng) sẽ bị đè hoàn toàn, không bao giờ hiển
    thị. Bỏ hẳn việc dựng nền mờ để tiết kiệm CPU mỗi frame (tối ưu tốc độ đã xác
    nhận). Giữ tham số downscale_divisor/dim để tương thích chữ ký với nơi gọi.
    """
    return crop_and_resize(frame, rect, target_w, target_h)


@dataclass(frozen=True)
class ForegroundPlacement:
    """Rect đặt foreground trên canvas (px). x,y = góc trên-trái, CÓ THỂ âm khi tràn."""

    fg_w: int
    fg_h: int
    x: int
    y: int


def place_foreground(
    src_w: int, src_h: int, canvas_w: int, canvas_h: int,
    fg_scale: float, person_cx_norm: float, person_cy_norm: float,
) -> ForegroundPlacement:
    """Tính cỡ + vị trí foreground (video A nguyên khung) trên canvas nền mờ.

    Baseline (fg_scale=1) = contain-fit. Đặt tâm người (norm [0,1]) vào giữa canvas,
    rồi kẹp theo trục: trục fg >= canvas -> pan bám người trong biên (2 rìa cắt);
    trục fg < canvas -> căn giữa (letterbox mờ). Xem spec.
    """
    if src_w <= 0 or src_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("Kích thước phải > 0")
    if fg_scale <= 0:
        raise ValueError("fg_scale phải > 0")

    contain = min(canvas_w / src_w, canvas_h / src_h)
    fg_w = max(1, round(src_w * contain * fg_scale))
    fg_h = max(1, round(src_h * contain * fg_scale))

    x = round(canvas_w / 2 - person_cx_norm * fg_w)
    y = round(canvas_h / 2 - person_cy_norm * fg_h)

    if fg_w >= canvas_w:
        x = max(canvas_w - fg_w, min(x, 0))   # kẹp [canvas_w-fg_w, 0]
    else:
        x = round((canvas_w - fg_w) / 2)
    if fg_h >= canvas_h:
        y = max(canvas_h - fg_h, min(y, 0))
    else:
        y = round((canvas_h - fg_h) / 2)

    return ForegroundPlacement(int(fg_w), int(fg_h), int(x), int(y))


def composite_manual_on_blurred_background(
    frame: np.ndarray, canvas_w: int, canvas_h: int, fg_scale: float,
    person_cx_norm: float, person_cy_norm: float,
    downscale_divisor: int = 32, dim: float = 0.55,
) -> np.ndarray:
    """Ghép video A (nguyên khung, cỡ fg_scale, bám người) lên nền mờ phủ kín canvas.

    Nền mờ vẽ trước; foreground resize theo scale (CUBIC khi phóng to) rồi dán phần
    chồng lấn canvas lên trên (phần tràn bị cắt). Xem spec compose thủ công.
    """
    bg = make_blurred_background(frame, canvas_w, canvas_h, downscale_divisor, dim)
    h, w = frame.shape[:2]
    p = place_foreground(w, h, canvas_w, canvas_h, fg_scale, person_cx_norm, person_cy_norm)

    interp = cv2.INTER_CUBIC if (p.fg_w > w or p.fg_h > h) else cv2.INTER_AREA
    fg = cv2.resize(frame, (p.fg_w, p.fg_h), interpolation=interp)

    x0, y0 = max(0, p.x), max(0, p.y)
    x1, y1 = min(canvas_w, p.x + p.fg_w), min(canvas_h, p.y + p.fg_h)
    if x1 <= x0 or y1 <= y0:
        return bg
    fx0, fy0 = x0 - p.x, y0 - p.y
    bg[y0:y1, x0:x1] = fg[fy0:fy0 + (y1 - y0), fx0:fx0 + (x1 - x0)]
    return bg
