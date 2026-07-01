"""Engine reframe: biến chuỗi 'tâm mục tiêu theo thời gian' thành vùng cắt mượt.

Đây là điểm cắm (plug point) cho auto-speaker-detection về sau:
- Chế độ THỦ CÔNG (hiện tại): tâm cố định do người dùng đặt bằng cách kéo khung preview.
- Chế độ TỰ ĐỘNG (tương lai): ASD/face-detector cấp tâm người nói mỗi frame →
  engine làm mượt (One Euro) rồi cắt. Chỉ cần thay `center_provider`, không sửa engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import CropRect, compute_crop_rect
from .smoothing import CameraSmoother


@dataclass
class ReframeParams:
    """Tham số reframe cho một video."""

    src_w: int
    src_h: int
    target_aspect: float
    zoom_x: float = 1.0
    zoom_y: float = 1.0
    # Tâm mặc định (chuẩn hoá 0..1) khi chưa có tín hiệu tự động.
    center_x_norm: float = 0.5
    center_y_norm: float = 0.5

    def default_center_px(self) -> tuple[float, float]:
        return self.center_x_norm * self.src_w, self.center_y_norm * self.src_h


class ReframeEngine:
    """Sinh CropRect cho từng frame, có làm mượt camera ảo."""

    def __init__(self, params: ReframeParams, smoother: CameraSmoother | None = None):
        self.params = params
        self.smoother = smoother or CameraSmoother()

    def reset(self) -> None:
        self.smoother.reset()

    def crop_for_center(
        self, center_x: float, center_y: float, dt: float, smooth: bool = True
    ) -> CropRect:
        if smooth:
            center_x, center_y = self.smoother.smooth(center_x, center_y, dt)
        return compute_crop_rect(
            self.params.src_w,
            self.params.src_h,
            self.params.target_aspect,
            center_x,
            center_y,
            self.params.zoom_x,
            self.params.zoom_y,
        )

    def static_crop(self) -> CropRect:
        """Vùng cắt tĩnh (không làm mượt) — dùng cho preview tức thời / chế độ thủ công."""
        cx, cy = self.params.default_center_px()
        return compute_crop_rect(
            self.params.src_w,
            self.params.src_h,
            self.params.target_aspect,
            cx,
            cy,
            self.params.zoom_x,
            self.params.zoom_y,
        )
