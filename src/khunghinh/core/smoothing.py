"""Bộ lọc làm mượt chuyển động camera ảo — lớp thuần, dễ kiểm thử.

Triển khai One Euro filter (Casiez, Roussel & Vogel, 2012): bám nhanh khi tín hiệu
thay đổi mạnh, rất mượt khi tín hiệu gần đứng yên → cảm giác "tripod pan" tự nhiên,
không giật. Đây là kỹ thuật được các dự án auto-reframe mã nguồn mở dùng phổ biến.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter 1 chiều.

    Tham số:
    - ``min_cutoff``: tần số cắt tối thiểu — nhỏ hơn = mượt hơn nhưng trễ hơn.
    - ``beta``: hệ số thích ứng theo tốc độ — lớn hơn = bám chuyển động nhanh tốt hơn.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3
        if self._x_prev is None:
            self._x_prev = x
            return x

        # Đạo hàm + làm mượt đạo hàm.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Cutoff thích ứng theo tốc độ ước lượng.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


@dataclass
class CameraSmoother:
    """Làm mượt tâm camera (cx, cy) bằng hai bộ One Euro độc lập."""

    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0

    def __post_init__(self) -> None:
        self._fx = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
        self._fy = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def smooth(self, cx: float, cy: float, dt: float) -> tuple[float, float]:
        return self._fx(cx, dt), self._fy(cy, dt)


class EmaFilter:
    """Trung bình trượt mũ (EMA) — đơn giản, dùng làm easing cuối hoặc phương án dự phòng."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)
        self._y: float | None = None

    def reset(self) -> None:
        self._y = None

    def __call__(self, x: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = self.alpha * x + (1 - self.alpha) * self._y
        return self._y
