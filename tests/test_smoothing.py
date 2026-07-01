from __future__ import annotations

import random

from khunghinh.core.smoothing import CameraSmoother, EmaFilter, OneEuroFilter


def _variance(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def test_one_euro_constant_input_is_stable():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = [f(5.0, 1 / 30) for _ in range(50)]
    assert abs(out[-1] - 5.0) < 1e-6


def test_one_euro_reduces_noise_variance():
    rng = random.Random(0)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    raw = [100 + rng.uniform(-10, 10) for _ in range(200)]
    sm = [f(x, 1 / 30) for x in raw]
    assert _variance(sm[50:]) < _variance(raw[50:]) * 0.5


def test_one_euro_tracks_step():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    for _ in range(30):
        f(0.0, 1 / 30)
    vals = [f(100.0, 1 / 30) for _ in range(60)]
    assert vals[-1] > 90  # hội tụ gần giá trị mục tiêu


def test_camera_smoother_returns_pair():
    s = CameraSmoother()
    x, y = s.smooth(10.0, 20.0, 1 / 30)
    assert isinstance(x, float) and isinstance(y, float)


def test_ema_basic():
    e = EmaFilter(alpha=0.5)
    assert e(10.0) == 10.0
    assert e(20.0) == 15.0
