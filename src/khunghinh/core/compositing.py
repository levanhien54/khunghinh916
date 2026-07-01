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
    bg = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if dim != 1.0:
        bg = np.clip(bg.astype(np.float32) * dim, 0, 255).astype(np.uint8)
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
