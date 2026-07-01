"""Toán học cắt khung (crop) — hàm thuần, không phụ thuộc GUI, dễ kiểm thử.

Quy ước toạ độ:
- Gốc (0,0) ở góc trên-trái, x sang phải, y xuống dưới (giống ảnh OpenCV/Qt).
- ``center_x``, ``center_y``: tâm vùng cắt, tính bằng pixel của ảnh gốc.
- ``zoom_x``, ``zoom_y`` >= 1.0: càng lớn càng cắt sát (vùng nguồn nhỏ lại → "phóng to").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropRect:
    """Hình chữ nhật cắt theo pixel ảnh gốc (số nguyên)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def base_crop_size(src_w: int, src_h: int, target_aspect: float) -> tuple[float, float]:
    """Kích thước vùng cắt lớn nhất có tỉ lệ ``target_aspect`` (= w/h) vừa khít trong ảnh gốc."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Kích thước ảnh gốc phải > 0")
    if target_aspect <= 0:
        raise ValueError("target_aspect phải > 0")

    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        # Ảnh gốc rộng hơn khung đích → giới hạn bởi chiều cao.
        h = float(src_h)
        w = h * target_aspect
    else:
        # Ảnh gốc cao/hẹp hơn → giới hạn bởi chiều rộng.
        w = float(src_w)
        h = w / target_aspect
    return w, h


def compute_crop_rect(
    src_w: int,
    src_h: int,
    target_aspect: float,
    center_x: float,
    center_y: float,
    zoom_x: float = 1.0,
    zoom_y: float = 1.0,
) -> CropRect:
    """Tính vùng cắt từ tâm + hệ số zoom theo từng trục, kẹp (clamp) trong ảnh gốc.

    Lưu ý: cho phép zoom X và Y độc lập (theo yêu cầu chỉnh tỉ lệ theo trục X/Y).
    Nếu zoom_x != zoom_y, ảnh xuất sẽ bị kéo giãn phi tỉ lệ khi resize về khung đích —
    đó là chủ ý điều khiển của người dùng (UI có tuỳ chọn khoá X=Y để tránh méo).
    """
    zoom_x = max(zoom_x, 1e-6)
    zoom_y = max(zoom_y, 1e-6)

    base_w, base_h = base_crop_size(src_w, src_h, target_aspect)
    w = min(base_w / zoom_x, float(src_w))
    h = min(base_h / zoom_y, float(src_h))

    # Đặt tâm rồi kẹp top-left để vùng cắt nằm trọn trong ảnh.
    x = _clamp(center_x - w / 2, 0.0, src_w - w)
    y = _clamp(center_y - h / 2, 0.0, src_h - h)

    return CropRect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))


def _clamp(v: float, lo: float, hi: float) -> float:
    if hi < lo:  # vùng cắt lớn bằng cả ảnh → top-left = 0
        return lo
    return max(lo, min(v, hi))
