# Hợp nhất "Nền mờ" với Reframe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "Nền mờ" (blurred background) an independent toggle that layers a
blurred safety-backdrop *underneath* the existing reframe crop (manual zoom/center
or auto speaker-tracking), instead of the current mutually-exclusive checkbox that
disables reframe entirely and shows the untouched full frame.

**Architecture:** The export/preview pipeline always computes a `CropRect` via
`ReframeEngine`/`compute_crop_rect` (exactly as today's non-blur path does). A new
`blurred_background: bool` flag only changes the final compositing step: draw
`make_blurred_background(frame, ...)` first, then overwrite it entirely with the
resized crop. Because the crop always resizes to exactly fill the target canvas,
the blurred layer is drawn but essentially never visible in the final output —
this is an explicitly confirmed trade-off (see spec), not a bug to fix later.

**Tech Stack:** Python 3.9+, OpenCV (`cv2`), NumPy, PyQt6, pytest. No new
dependencies.

## Global Constraints

- CPU-only: no GPU-dependent APIs anywhere in this change.
- Spec: `docs/superpowers/specs/2026-07-01-blur-background-reframe-design.md` — follow it exactly; do not reintroduce the old "blur_fit = full frame, no crop" mode in any form.
- Match existing code conventions: Vietnamese-language docstrings/comments throughout `src/khunghinh/`, module-level docstring at top of each touched file, `from __future__ import annotations` at top of core/mediaio modules that already have it — preserve this style, don't switch to English or strip existing comments.
- `pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]` — run tests with `python -m pytest` from the repo root (`C:\Users\sonson\Desktop\KhungHinh916`), no extra path setup needed.
- This is not a git repository (`git status` fails with "not a git repository"). Skip all `git add`/`git commit` steps unless the user has since run `git init` — if so, commit after each task exactly as written below.
- No existing unit tests target `exporter.py`, `control_panel.py`, or `main_window.py` (PyQt6/ffmpeg/cv2 I/O layers are validated manually per README, not unit-tested) — do not invent new test harnesses for them in this plan; verify those tasks with a plain `python -c "import ..."` syntax/import check plus the final full-suite run.

---

### Task 1: `core/compositing.py` — replace full-frame composite with crop-based composite

**Files:**
- Modify: `src/khunghinh/core/compositing.py`
- Modify: `tests/test_compositing.py`

**Interfaces:**
- Consumes: `CropRect` from `src/khunghinh/core/geometry.py` (fields `x`, `y`, `width`, `height`, already defined — no changes needed there).
- Produces (used by Task 3):
  - `crop_and_resize(frame: np.ndarray, rect: CropRect, target_w: int, target_h: int) -> np.ndarray`
  - `composite_crop_on_blurred_background(frame: np.ndarray, rect: CropRect, target_w: int, target_h: int, downscale_divisor: int = 32, dim: float = 0.55) -> np.ndarray`
  - `make_blurred_background(...)` and `fit_dimensions(...)` are unchanged (still exported, still used by tests).
  - `composite_fit_on_blurred_background` is **removed** — Task 5 removes its last caller.

- [ ] **Step 1: Update the test file — remove old full-frame tests, add new crop-based tests**

Replace the entire contents of `tests/test_compositing.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail (missing functions)**

Run: `python -m pytest tests/test_compositing.py -v`
Expected: FAIL — `ImportError: cannot import name 'crop_and_resize' from 'khunghinh.core.compositing'` (and/or `composite_crop_on_blurred_background`).

- [ ] **Step 3: Replace `src/khunghinh/core/compositing.py`**

Replace the entire file contents with:

```python
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
    """Cắt `frame` theo `rect` rồi resize phủ kín (target_w, target_h)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, rect.x), max(0, rect.y)
    x2, y2 = min(rect.x + rect.width, w), min(rect.y + rect.height, h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def composite_crop_on_blurred_background(
    frame: np.ndarray,
    rect: CropRect,
    target_w: int,
    target_h: int,
    downscale_divisor: int = 32,
    dim: float = 0.55,
) -> np.ndarray:
    """Vẽ nền mờ trước, sau đó đè khung hình đã cắt (`rect`) phủ kín toàn canvas.

    Vì `crop_and_resize` luôn trả về đúng kích thước (target_w, target_h), nền mờ
    bị ghi đè hoàn toàn — xem docstring module ở đầu file.
    """
    canvas = make_blurred_background(frame, target_w, target_h, downscale_divisor, dim)
    canvas[:] = crop_and_resize(frame, rect, target_w, target_h)
    return canvas
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compositing.py -v`
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit** (skip if not a git repo — see Global Constraints)

```bash
git add src/khunghinh/core/compositing.py tests/test_compositing.py
git commit -m "refactor(compositing): crop-based composite replaces full-frame fit"
```

---

### Task 2: `config.py` — replace `background_mode` string with `blurred_background` bool

**Files:**
- Modify: `src/khunghinh/config.py:31-34`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Task 5): `AppConfig.blurred_background: bool` (default `False`), replacing `AppConfig.background_mode: str`. `bg_blur_downscale_divisor: int` and `bg_blur_dim: float` are unchanged.

- [ ] **Step 1: Edit the dataclass field**

In `src/khunghinh/config.py`, replace:

```python
    # --- Nền mờ (giữ trọn khung hình, không cắt) ---
    background_mode: str = "crop_fill"   # "crop_fill" | "blur_fit"
    bg_blur_downscale_divisor: int = 32
    bg_blur_dim: float = 0.55
```

with:

```python
    # --- Nền mờ (lớp nền an toàn phía dưới khung hình đã cắt) ---
    blurred_background: bool = False
    bg_blur_downscale_divisor: int = 32
    bg_blur_dim: float = 0.55
```

- [ ] **Step 2: Sanity-check import**

Run: `python -c "from khunghinh.config import AppConfig; c = AppConfig(); print(c.blurred_background, c.bg_blur_downscale_divisor)"`
Expected output: `False 32`

- [ ] **Step 3: Run full test suite to confirm no regression**

Run: `python -m pytest -q`
Expected: all tests pass (no test references `background_mode`).

- [ ] **Step 4: Commit** (skip if not a git repo)

```bash
git add src/khunghinh/config.py
git commit -m "refactor(config): blurred_background bool replaces background_mode enum"
```

---

### Task 3: `mediaio/exporter.py` — always compute crop, branch only on compositing

**Files:**
- Modify: `src/khunghinh/mediaio/exporter.py`

**Interfaces:**
- Consumes: `crop_and_resize`, `composite_crop_on_blurred_background` from `src/khunghinh/core/compositing.py` (Task 1). `AppConfig.blurred_background` naming convention from Task 2 (this file defines its own `ExportSettings`, not `AppConfig`, but mirrors the same field name for consistency).
- Produces (used by Task 5): `ExportSettings.blurred_background: bool = False` replacing `ExportSettings.background_mode: str`. `VideoExporter.run()` behavior: always resolves `rect` via `center_provider`/`engine.static_crop()` regardless of `blurred_background`.

- [ ] **Step 1: Edit imports**

`CropRect` and `numpy` in this file are only ever used by `_crop_and_resize`
(Step 4 deletes it), so both become unused once that function is gone — drop
them along with swapping the compositing import.

In `src/khunghinh/mediaio/exporter.py`, replace:

```python
import cv2
import numpy as np

from ..core.compositing import composite_fit_on_blurred_background
from ..core.geometry import CropRect
from ..core.reframe_engine import ReframeEngine
from .reader import VideoReader
```

with:

```python
import cv2

from ..core.compositing import composite_crop_on_blurred_background, crop_and_resize
from ..core.reframe_engine import ReframeEngine
from .reader import VideoReader
```

- [ ] **Step 2: Edit `ExportSettings`**

Replace:

```python
    # "crop_fill": cắt+lấp đầy khung đích (mặc định, hành vi hiện có).
    # "blur_fit": giữ TRỌN khung hình gốc (không cắt), đặt lên nền mờ phủ kín.
    background_mode: str = "crop_fill"
    bg_blur_downscale_divisor: int = 32
    bg_blur_dim: float = 0.55
```

with:

```python
    # Nền mờ: lớp nền an toàn vẽ trước rồi bị khung hình đã cắt đè kín lên trên
    # (xem docs/superpowers/specs/2026-07-01-blur-background-reframe-design.md).
    blurred_background: bool = False
    bg_blur_downscale_divisor: int = 32
    bg_blur_dim: float = 0.55
```

- [ ] **Step 3: Edit `VideoExporter.run()`**

Replace:

```python
        blur_fit = self.settings.background_mode == "blur_fit"
        self.engine.reset()
        self.reader.rewind()
        total = info.frame_count
        idx = 0
        try:
            while True:
                if cancel_cb and cancel_cb():
                    raise InterruptedError("Đã hủy xuất.")
                frame = self.reader.read_next()
                if frame is None:
                    break
                if blur_fit:
                    out_frame = composite_fit_on_blurred_background(
                        frame, tw, th,
                        downscale_divisor=self.settings.bg_blur_downscale_divisor,
                        dim=self.settings.bg_blur_dim,
                    )
                else:
                    if center_provider is not None:
                        cx, cy = center_provider(idx)
                        rect = self.engine.crop_for_center(cx, cy, dt, smooth=smooth)
                    else:
                        rect = self.engine.static_crop()
                    out_frame = _crop_and_resize(frame, rect, tw, th)
                writer.write(out_frame)
                idx += 1
                if progress_cb and total:
                    progress_cb(idx, total)
        finally:
            writer.release()
```

with:

```python
        self.engine.reset()
        self.reader.rewind()
        total = info.frame_count
        idx = 0
        try:
            while True:
                if cancel_cb and cancel_cb():
                    raise InterruptedError("Đã hủy xuất.")
                frame = self.reader.read_next()
                if frame is None:
                    break
                if center_provider is not None:
                    cx, cy = center_provider(idx)
                    rect = self.engine.crop_for_center(cx, cy, dt, smooth=smooth)
                else:
                    rect = self.engine.static_crop()
                if self.settings.blurred_background:
                    out_frame = composite_crop_on_blurred_background(
                        frame, rect, tw, th,
                        downscale_divisor=self.settings.bg_blur_downscale_divisor,
                        dim=self.settings.bg_blur_dim,
                    )
                else:
                    out_frame = crop_and_resize(frame, rect, tw, th)
                writer.write(out_frame)
                idx += 1
                if progress_cb and total:
                    progress_cb(idx, total)
        finally:
            writer.release()
```

- [ ] **Step 4: Remove the now-unused local `_crop_and_resize` helper**

Delete this function from the bottom of the file (it has been replaced by
`crop_and_resize` in `core/compositing.py`, imported in Step 1):

```python
def _crop_and_resize(frame: np.ndarray, rect: CropRect, tw: int, th: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = max(0, rect.x), max(0, rect.y)
    x2, y2 = min(rect.x + rect.width, w), min(rect.y + rect.height, h)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = frame
    return cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)
```

Leave `_safe_rmtree` and the rest of the file untouched. Neither `numpy` nor
`CropRect` are referenced anywhere else in this file (verify with a quick
search before moving on) — Step 1 already dropped both imports.

- [ ] **Step 5: Sanity-check import**

Run: `python -c "from khunghinh.mediaio.exporter import ExportSettings, VideoExporter; s = ExportSettings(out_path='x.mp4'); print(s.blurred_background)"`
Expected output: `False`

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest -q`
Expected: all tests pass (no test imports `_crop_and_resize` or `background_mode`).

- [ ] **Step 7: Commit** (skip if not a git repo)

```bash
git add src/khunghinh/mediaio/exporter.py
git commit -m "refactor(exporter): always reframe-crop, blur is a compositing-only flag"
```

---

### Task 4: `ui/control_panel.py` — stop disabling reframe controls when blur is on

**Files:**
- Modify: `src/khunghinh/ui/control_panel.py:63,128-180`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Task 5): `ControlPanel.backgroundModeChanged` signal keeps the same signature (`pyqtSignal(bool)`), `ControlPanel.is_blur_background()` keeps the same signature — only internal enable/disable logic changes, no public interface changes.

- [ ] **Step 1: Update the signal comment**

Replace:

```python
    backgroundModeChanged = pyqtSignal(bool)  # True = nền mờ (blur_fit)
```

with:

```python
    backgroundModeChanged = pyqtSignal(bool)  # True = bật nền mờ (lớp nền an toàn)
```

- [ ] **Step 2: Update the "Nền" group box hint text**

Replace:

```python
        # --- Nền mờ (giữ trọn khung hình) ---
        bg_box = QGroupBox("Nền")
        blay = QVBoxLayout(bg_box)
        self.chk_blur_bg = QCheckBox("Nền mờ — giữ trọn khung hình (không cắt)")
        bg_hint = QLabel(
            "Thu nhỏ khung hình gốc rồi phóng to làm nền mờ phủ kín; đặt khung hình "
            "gốc (vừa trọn, không cắt) lên trên. Khi bật, khung cắt thủ công bị tắt."
        )
```

with:

```python
        # --- Nền mờ (lớp nền an toàn phía dưới khung hình đã cắt) ---
        bg_box = QGroupBox("Nền")
        blay = QVBoxLayout(bg_box)
        self.chk_blur_bg = QCheckBox("Nền mờ (lớp nền an toàn phía dưới khung đã cắt)")
        bg_hint = QLabel(
            "Thu nhỏ khung hình gốc rồi phóng to làm nền mờ; khung hình đã cắt theo "
            "Thủ công/Tự động vẫn hoạt động bình thường và phủ kín lên trên nền mờ."
        )
```

- [ ] **Step 3: Stop disabling reframe controls in `_on_blur_bg_toggle`**

Replace:

```python
    # --- Nền mờ ---
    def _on_blur_bg_toggle(self, checked: bool) -> None:
        is_blur = bool(checked)
        # Nền mờ giữ TRỌN khung hình (không cắt) -> khung cắt thủ công/tự động vô nghĩa.
        for w in (self.rad_manual, self.rad_auto, self.sld_zoom_x, self.sld_zoom_y,
                  self.chk_link, self.btn_reset):
            w.setEnabled(not is_blur)
        self._refresh_analyze_enabled()
        self.backgroundModeChanged.emit(is_blur)
```

with:

```python
    # --- Nền mờ ---
    def _on_blur_bg_toggle(self, checked: bool) -> None:
        is_blur = bool(checked)
        self.backgroundModeChanged.emit(is_blur)
```

- [ ] **Step 4: Stop gating "Phân tích video" on the blur checkbox**

Replace:

```python
    def _refresh_analyze_enabled(self) -> None:
        self.btn_analyze.setEnabled(
            self._has_video and self.rad_auto.isChecked() and not self.chk_blur_bg.isChecked()
        )
```

with:

```python
    def _refresh_analyze_enabled(self) -> None:
        self.btn_analyze.setEnabled(self._has_video and self.rad_auto.isChecked())
```

- [ ] **Step 5: Sanity-check import**

Run: `python -c "import khunghinh.ui.control_panel"`
Expected: no error (PyQt6 import succeeds, no syntax errors).

- [ ] **Step 6: Commit** (skip if not a git repo)

```bash
git add src/khunghinh/ui/control_panel.py
git commit -m "feat(ui): blur toggle no longer disables reframe controls"
```

---

### Task 5: `ui/main_window.py` — wire the combined blur+reframe export/preview

**Files:**
- Modify: `src/khunghinh/ui/main_window.py:23,181-188,234-240,294,318-337`

**Interfaces:**
- Consumes: `ExportSettings.blurred_background` (Task 3), `AppConfig.blurred_background` naming (Task 2 — note: `MainWindow` reads `self.config.bg_blur_downscale_divisor`/`bg_blur_dim` only, it does not read `AppConfig.blurred_background` directly since the UI state comes from `self._blur_bg`, already tracked).
- Produces: no new public interface — this is the top of the call chain.

Note (out of scope, no action needed): after Step 2, `PreviewView.set_overlay_enabled()`
(`src/khunghinh/ui/preview_view.py:179-182`) loses its only caller. It's a
pre-existing public widget method unrelated to this spec — leave it as-is; don't
delete it as part of this plan.

- [ ] **Step 1: Drop the now-unused `composite_fit_on_blurred_background` import**

Replace:

```python
from ..core.compositing import composite_fit_on_blurred_background
from ..core.geometry import CropRect, compute_crop_rect
```

with:

```python
from ..core.geometry import CropRect, compute_crop_rect
```

- [ ] **Step 2: Simplify `_redraw_for_frame`— preview always shows the crop-rect overlay**

Replace:

```python
    def _redraw_for_frame(self, i: int) -> None:
        if not self._reader or not self._params:
            return
        frame = self._reader.read_at(i)
        self.lbl_frame.setText(f"frame {i}")
        if frame is None:
            return

        if self._blur_bg:
            composite = composite_fit_on_blurred_background(
                frame, self.config.target_width, self.config.target_height,
                downscale_divisor=self.config.bg_blur_downscale_divisor,
                dim=self.config.bg_blur_dim,
            )
            self.preview.set_frame(composite)
            return

        self.preview.set_frame(frame)
        if self._mode == "auto" and self._analysis is not None:
            faces, active = self._analysis.faces_for_frame(i)
            cx, cy = self._analysis.centers_for_frame(i)
            self.preview.set_faces(faces, active)
            self.preview.set_auto_crop_rect(self._crop_for_center(cx, cy))
        else:
            cx, cy = self._center_px or self._params.default_center_px()
            self.preview.set_crop_rect(self._crop_for_center(cx, cy))
```

with:

```python
    def _redraw_for_frame(self, i: int) -> None:
        if not self._reader or not self._params:
            return
        frame = self._reader.read_at(i)
        self.lbl_frame.setText(f"frame {i}")
        if frame is None:
            return

        # Nền mờ chỉ ảnh hưởng bước XUẤT video (xem VideoExporter.run) — preview
        # tương tác luôn hiển thị khung cắt giống Thủ công/Tự động, tránh dựng lại
        # composite tốn CPU mỗi lần tua frame.
        self.preview.set_frame(frame)
        if self._mode == "auto" and self._analysis is not None:
            faces, active = self._analysis.faces_for_frame(i)
            cx, cy = self._analysis.centers_for_frame(i)
            self.preview.set_faces(faces, active)
            self.preview.set_auto_crop_rect(self._crop_for_center(cx, cy))
        else:
            cx, cy = self._center_px or self._params.default_center_px()
            self.preview.set_crop_rect(self._crop_for_center(cx, cy))
```

- [ ] **Step 3: Simplify `on_background_mode_changed`**

Replace:

```python
    def on_background_mode_changed(self, is_blur: bool) -> None:
        self._blur_bg = is_blur
        self.preview.set_overlay_enabled(not is_blur)
        self._redraw_for_frame(self._cur_frame)
        self.statusBar().showMessage(
            "Nền mờ: giữ trọn khung hình, không cắt." if is_blur else "Đã tắt nền mờ."
        )
```

with:

```python
    def on_background_mode_changed(self, is_blur: bool) -> None:
        self._blur_bg = is_blur
        self.statusBar().showMessage(
            "Nền mờ: bật (lớp nền an toàn khi xuất)." if is_blur else "Nền mờ: tắt."
        )
```

- [ ] **Step 4: Drop the blur guard on the "cần phân tích" warning in `on_export`**

Replace:

```python
        if not self._blur_bg and self._mode == "auto" and self._analysis is None:
            QMessageBox.information(self, "Cần phân tích", "Hãy bấm 'Phân tích video' trước khi xuất tự động.")
            return
```

with:

```python
        if self._mode == "auto" and self._analysis is None:
            QMessageBox.information(self, "Cần phân tích", "Hãy bấm 'Phân tích video' trước khi xuất tự động.")
            return
```

- [ ] **Step 5: Wire `ExportSettings` and unify `center_provider`/`smooth` resolution**

Replace:

```python
        settings = ExportSettings(
            out_path=out_path,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            crf=self.config.export_crf,
            codec=self.config.export_codec,
            background_mode="blur_fit" if self._blur_bg else "crop_fill",
            bg_blur_downscale_divisor=self.config.bg_blur_downscale_divisor,
            bg_blur_dim=self.config.bg_blur_dim,
        )
        exporter = VideoExporter(export_reader, engine, settings)

        if self._blur_bg:
            center_provider, smooth = None, True  # bỏ qua — exporter dùng nhánh blur_fit
        elif self._mode == "auto" and self._analysis is not None:
            center_provider = self._analysis.make_center_provider()
            smooth = False  # quỹ đạo đã được làm mượt ở Pass 2 — KHÔNG làm mượt lần nữa
        else:
            center_provider = None
            smooth = True
```

with:

```python
        settings = ExportSettings(
            out_path=out_path,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            crf=self.config.export_crf,
            codec=self.config.export_codec,
            blurred_background=self._blur_bg,
            bg_blur_downscale_divisor=self.config.bg_blur_downscale_divisor,
            bg_blur_dim=self.config.bg_blur_dim,
        )
        exporter = VideoExporter(export_reader, engine, settings)

        if self._mode == "auto" and self._analysis is not None:
            center_provider = self._analysis.make_center_provider()
            smooth = False  # quỹ đạo đã được làm mượt ở Pass 2 — KHÔNG làm mượt lần nữa
        else:
            center_provider = None
            smooth = True
```

- [ ] **Step 6: Sanity-check import**

Run: `python -c "import khunghinh.ui.main_window"`
Expected: no error.

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Commit** (skip if not a git repo)

```bash
git add src/khunghinh/ui/main_window.py
git commit -m "feat(ui): combine blurred background with reframe crop on export"
```

---

### Task 6: Update README and verify final test count

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (documentation-only + final verification).

- [ ] **Step 1: Run the full suite and record the test count**

Run: `python -m pytest -q`
Expected: `88 passed` (was 86; Task 1's rewrite of `tests/test_compositing.py`
removed 3 tests and added 5, net +2 → 88). If the number differs, use whatever
`python -m pytest -q` actually reports as `N` in Step 3 below instead of 88.

- [ ] **Step 2: Update the "Chế độ Nền mờ" bullet in `README.md`**

Replace:

```
- 🌫️ **Chế độ Nền mờ** — giữ **trọn** khung hình gốc (không cắt mất nội dung 2 bên):
  thu nhỏ ảnh gốc ~32 lần rồi phóng to lại (mờ rẻ trên CPU) làm nền phủ kín 9:16,
  đặt khung hình gốc (vừa trọn, căn giữa) lên trên.
```

with:

```
- 🌫️ **Nền mờ** — bật kèm với Thủ công/Tự động (không còn loại trừ lẫn nhau):
  thu nhỏ ảnh gốc ~32 lần rồi phóng to lại (mờ rẻ trên CPU) vẽ làm lớp nền phủ
  kín 9:16, sau đó khung hình đã cắt theo tâm/zoom (thủ công hoặc bám người nói)
  vẫn phủ kín lên trên như bình thường — nền mờ là lớp nền an toàn phía dưới.
```

- [ ] **Step 3: Update the test-count line in `README.md`**

Replace:

```
- 🧱 Kiến trúc phân lớp, **86 unit test** cho phần logic, **logging** ra file để debug.
```

with the actual count `N` recorded in Step 1, e.g. if `N` is 88:

```
- 🧱 Kiến trúc phân lớp, **88 unit test** cho phần logic, **logging** ra file để debug.
```

Also update the matching count in the `## Kiểm thử` section:

```
python -m pytest          # 86 unit test: geometry, smoothing, tracking, ASD, camera path,
                           # audio VAD, face detection, compositing, analysis result...
```

to use the same `N`.

- [ ] **Step 4: Commit** (skip if not a git repo)

```bash
git add README.md
git commit -m "docs: describe combined blur+reframe background mode"
```

---

## Final Verification

- [ ] Run `python -m pytest -q` one more time from repo root — full suite passes.
- [ ] Run `python run.py`, import `demo_video.mp4`, verify: (a) checking "Nền mờ" no longer disables the Thủ công/Tự động radio buttons or zoom sliders, (b) "Phân tích video" is clickable in Tự động mode regardless of the blur checkbox, (c) exporting with blur ON in Tự động mode (after analyzing) produces a video visually identical to exporting with blur OFF (same crop, since blur is now a no-op-visually safety layer) — this is the confirmed intended behavior, not a bug.
