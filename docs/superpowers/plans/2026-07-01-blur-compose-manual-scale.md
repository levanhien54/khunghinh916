# Nền mờ Compose thủ công (cỡ tay + bám người) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chế độ Nền mờ đặt video A (nguyên khung, giữ tỉ lệ) lên nền mờ với cỡ do người dùng chỉnh tay (`fg_scale`, tĩnh) và vị trí tự bám người nói (pan ngang + letterbox mờ trên/dưới cho nguồn landscape).

**Architecture:** Một hàm thuần `place_foreground` tính rect đặt foreground (contain × scale, đặt tâm người vào giữa canvas, kẹp theo trục). `composite_manual_on_blurred_background` dùng nó để dán video A đã resize lên nền mờ. Exporter + preview dùng chung hàm compose này (WYSIWYG). Preview hiển thị canvas đã ghép + gizmo kéo góc để scale; control panel thêm slider "Cỡ video A".

**Tech Stack:** Python 3.9+, OpenCV (`cv2`), NumPy, PyQt6, pytest.

## Global Constraints

- CPU-only; không GPU. Không thêm dependency mới.
- Comment/docstring tiếng Việt, `from __future__ import annotations` ở đầu module core/mediaio (theo style hiện có).
- Chạy test từ gốc repo (`C:\Users\sonson\Desktop\KhungHinh916`) bằng `python -m pytest` (pyproject đặt `pythonpath=["src"]`). Cho lệnh import thủ công cần `PYTHONPATH=src` (pytest tự lo, `python -c` thì không).
- Repo là git (đã init). Commit sau mỗi task với dòng cuối: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Lớp Qt/ffmpeg (control_panel, preview_view, main_window, exporter runtime) KHÔNG unit-test (theo tiền lệ dự án) — verify bằng `python -c "import ..."` + smoke-test headless + build exe.
- Spec: `docs/superpowers/specs/2026-07-01-blur-compose-manual-scale-design.md`.

---

### Task 1: `place_foreground` — hàm thuần tính rect đặt foreground

**Files:**
- Modify: `src/khunghinh/core/compositing.py`
- Modify: `tests/test_compositing.py`

**Interfaces:**
- Consumes: nothing.
- Produces (dùng ở Task 2, 6):
  - `ForegroundPlacement` dataclass (frozen): fields `fg_w: int, fg_h: int, x: int, y: int` (x,y = góc trên-trái trên canvas, có thể âm).
  - `place_foreground(src_w:int, src_h:int, canvas_w:int, canvas_h:int, fg_scale:float, person_cx_norm:float, person_cy_norm:float) -> ForegroundPlacement`

- [ ] **Step 1: Viết test thất bại** — thêm vào cuối `tests/test_compositing.py`:

```python
def test_place_foreground_landscape_fit_matches_image():
    from khunghinh.core.compositing import place_foreground
    p = place_foreground(1920, 1080, 1080, 1920, 1.0, 0.5, 0.5)
    assert p.fg_w == 1080          # khít chiều rộng
    assert abs(p.fg_h - 608) <= 1  # thấp hơn canvas
    assert p.x == 0                # phủ ngang
    assert p.y == (1920 - p.fg_h) // 2  # căn giữa dọc (letterbox mờ trên/dưới)


def test_place_foreground_zoomed_pans_to_person():
    from khunghinh.core.compositing import place_foreground
    center = place_foreground(1920, 1080, 1080, 1920, 2.0, 0.5, 0.5)
    right = place_foreground(1920, 1080, 1080, 1920, 2.0, 0.85, 0.5)
    assert center.fg_w > 1080 and center.fg_h < 1920  # rộng hơn canvas, vẫn thấp hơn
    assert -center.fg_w + 1080 <= center.x <= 0        # kẹp trong biên ngang
    assert right.x < center.x                          # người lệch phải -> fg dịch trái (pan)


def test_place_foreground_person_at_edge_clamps_no_blur_gap():
    from khunghinh.core.compositing import place_foreground
    p = place_foreground(1920, 1080, 1080, 1920, 2.0, 1.0, 0.5)
    assert p.x == 1080 - p.fg_w    # kẹp sát biên phải, không lộ viền ngang


def test_place_foreground_small_scale_centers_both_axes():
    from khunghinh.core.compositing import place_foreground
    p = place_foreground(1920, 1080, 1080, 1920, 0.5, 0.9, 0.1)  # người lệch nhưng fg<canvas
    assert p.fg_w < 1080 and p.fg_h < 1920
    assert p.x == round((1080 - p.fg_w) / 2)   # căn giữa, KHÔNG theo người
    assert p.y == round((1920 - p.fg_h) / 2)


def test_place_foreground_invalid_raises():
    from khunghinh.core.compositing import place_foreground
    with pytest.raises(ValueError):
        place_foreground(0, 100, 100, 100, 1.0, 0.5, 0.5)
    with pytest.raises(ValueError):
        place_foreground(100, 100, 100, 100, 0.0, 0.5, 0.5)
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `python -m pytest tests/test_compositing.py -q -k place_foreground`
Expected: FAIL — `ImportError: cannot import name 'place_foreground'`.

- [ ] **Step 3: Thêm code vào `src/khunghinh/core/compositing.py`**

Thêm import dataclass ở đầu file (sau `from __future__ import annotations`):

```python
from dataclasses import dataclass
```

Thêm cuối file:

```python
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
```

- [ ] **Step 4: Chạy để xác nhận PASS**

Run: `python -m pytest tests/test_compositing.py -q -k place_foreground`
Expected: PASS — 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/khunghinh/core/compositing.py tests/test_compositing.py
git commit -m "feat(compositing): place_foreground — rect đặt foreground bám người trên canvas"
```

---

### Task 2: `composite_manual_on_blurred_background` — ghép compose

**Files:**
- Modify: `src/khunghinh/core/compositing.py`
- Modify: `tests/test_compositing.py`

**Interfaces:**
- Consumes: `place_foreground`, `ForegroundPlacement` (Task 1), `make_blurred_background` (đã có).
- Produces (dùng ở Task 3, 6): `composite_manual_on_blurred_background(frame:np.ndarray, canvas_w:int, canvas_h:int, fg_scale:float, person_cx_norm:float, person_cy_norm:float, downscale_divisor:int=32, dim:float=0.55) -> np.ndarray` — trả canvas (canvas_h, canvas_w, 3) uint8.

- [ ] **Step 1: Viết test thất bại** — thêm vào `tests/test_compositing.py`:

```python
def test_composite_manual_shape_dtype():
    from khunghinh.core.compositing import composite_manual_on_blurred_background
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    out = composite_manual_on_blurred_background(frame, 1080, 1920, 1.0, 0.5, 0.5)
    assert out.shape == (1920, 1080, 3)
    assert out.dtype == np.uint8


def test_composite_manual_foreground_band_differs_from_blur_band():
    from khunghinh.core.compositing import composite_manual_on_blurred_background
    # foreground sáng đều, nền mờ bị dim -> band giữa (fg) sáng hơn band trên (blur)
    frame = np.full((1080, 1920, 3), 240, dtype=np.uint8)
    out = composite_manual_on_blurred_background(frame, 1080, 1920, 1.0, 0.5, 0.5, dim=0.4)
    top_band = out[50].mean()       # y=50 nằm vùng letterbox mờ (fg bắt đầu ~y=656)
    mid_band = out[960].mean()      # y=960 nằm giữa foreground
    assert mid_band > top_band + 20


def test_composite_manual_small_scale_corners_are_blur():
    from khunghinh.core.compositing import composite_manual_on_blurred_background, place_foreground
    frame = np.full((1080, 1920, 3), 240, dtype=np.uint8)
    out = composite_manual_on_blurred_background(frame, 1080, 1920, 0.5, 0.5, 0.5, dim=0.3)
    p = place_foreground(1920, 1080, 1080, 1920, 0.5, 0.5, 0.5)
    assert p.x > 0 and p.y > 0                 # fg nhỏ hơn canvas, có viền
    assert out[0, 0].mean() < 200              # góc canvas là nền mờ (bị dim), không phải fg 240


def test_composite_manual_odd_sizes_no_crash():
    from khunghinh.core.compositing import composite_manual_on_blurred_background
    frame = np.random.randint(0, 255, (137, 251, 3), dtype=np.uint8)
    out = composite_manual_on_blurred_background(frame, 1080, 1920, 1.3, 0.6, 0.4)
    assert out.shape == (1920, 1080, 3) and out.dtype == np.uint8
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `python -m pytest tests/test_compositing.py -q -k composite_manual`
Expected: FAIL — `ImportError: cannot import name 'composite_manual_on_blurred_background'`.

- [ ] **Step 3: Thêm code vào `src/khunghinh/core/compositing.py`** (cuối file):

```python
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
```

- [ ] **Step 4: Chạy để xác nhận PASS**

Run: `python -m pytest tests/test_compositing.py -q`
Expected: PASS — toàn bộ file compositing xanh (bao gồm test cũ + 9 test mới của Task 1+2).

- [ ] **Step 5: Commit**

```bash
git add src/khunghinh/core/compositing.py tests/test_compositing.py
git commit -m "feat(compositing): composite_manual_on_blurred_background — video A trên nền mờ"
```

---

### Task 3: Config + Exporter dùng compose ở nhánh nền mờ

**Files:**
- Modify: `src/khunghinh/config.py`
- Modify: `src/khunghinh/mediaio/exporter.py`
- Modify: `tests/test_exporter.py`

**Interfaces:**
- Consumes: `composite_manual_on_blurred_background` (Task 2).
- Produces (dùng ở Task 4, 6): `AppConfig.fg_scale_default/min/max`; `ExportSettings.fg_scale: float = 1.0`; nhánh blur của `VideoExporter._iter_output_frames` dùng compose với tâm người từ `center_provider` (hoặc tâm khung nếu None).

- [ ] **Step 1: Thêm test cho ExportSettings.fg_scale** — thêm vào `tests/test_exporter.py`:

```python
def test_export_settings_has_fg_scale_default():
    from khunghinh.mediaio.exporter import ExportSettings
    s = ExportSettings(out_path="out.mp4")
    assert s.fg_scale == 1.0
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `python -m pytest tests/test_exporter.py -q -k fg_scale`
Expected: FAIL — `AttributeError: 'ExportSettings' object has no attribute 'fg_scale'`.

- [ ] **Step 3a: `config.py`** — thêm sau cụm `bg_blur_dim` (khối "Nền mờ"):

```python
    fg_scale_default: float = 1.0   # 1.0 = vừa khít (contain); >1 phóng to; <1 thu nhỏ
    fg_scale_min: float = 0.3
    fg_scale_max: float = 3.0
```

- [ ] **Step 3b: `exporter.py` ExportSettings** — thêm field sau `bg_blur_dim`:

```python
    fg_scale: float = 1.0   # cỡ video A trên nền mờ (chế độ compose thủ công)
```

- [ ] **Step 3c: `exporter.py` import** — đổi dòng import compositing thành:

```python
from ..core.compositing import composite_manual_on_blurred_background, crop_and_resize
```

(bỏ `composite_crop_on_blurred_background` — không còn dùng ở exporter.)

- [ ] **Step 3d: `exporter.py` `_iter_output_frames`** — thay thân vòng lặp (khối `if center_provider ... else crop_and_resize`) bằng:

```python
            if self.settings.blurred_background:
                h, w = frame.shape[:2]
                if center_provider is not None:
                    cx, cy = center_provider(idx)
                else:
                    cx, cy = w / 2.0, h / 2.0
                out_frame = composite_manual_on_blurred_background(
                    frame, tw, th, self.settings.fg_scale,
                    cx / w, cy / h,
                    downscale_divisor=self.settings.bg_blur_downscale_divisor,
                    dim=self.settings.bg_blur_dim,
                )
            else:
                if center_provider is not None:
                    cx, cy = center_provider(idx)
                    rect = self.engine.crop_for_center(cx, cy, dt, smooth=smooth)
                else:
                    rect = self.engine.static_crop()
                out_frame = crop_and_resize(frame, rect, tw, th)
```

- [ ] **Step 4: Chạy test + import**

Run: `python -m pytest tests/test_exporter.py -q`
Expected: PASS.
Run: `PYTHONPATH=src python -c "from khunghinh.mediaio.exporter import VideoExporter, ExportSettings; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Smoke-test xuất nền mờ compose** (frame center, scale 1.3):

```bash
PYTHONPATH=src python -c "
from khunghinh.mediaio.reader import VideoReader
from khunghinh.core.reframe_engine import ReframeEngine, ReframeParams
from khunghinh.core.smoothing import CameraSmoother
from khunghinh.mediaio.exporter import VideoExporter, ExportSettings
r=VideoReader('demo_video.mp4'); info=r.open()
eng=ReframeEngine(ReframeParams(info.width,info.height,1080/1920,1.0,1.0,0.5,0.5), CameraSmoother(1.0,0.05))
VideoExporter(r,eng,ExportSettings(out_path='scratch_blur.mp4', blurred_background=True, fg_scale=1.3)).run(); r.release()
import cv2,os; c=cv2.VideoCapture('scratch_blur.mp4'); print('OK', c.isOpened(), int(c.get(3)), int(c.get(4)), int(c.get(7))); c.release(); os.remove('scratch_blur.mp4')
"
```
Expected: `OK True 1080 1920 90`.

- [ ] **Step 6: Commit**

```bash
git add src/khunghinh/config.py src/khunghinh/mediaio/exporter.py tests/test_exporter.py
git commit -m "feat(exporter): nhánh nền mờ dùng compose thủ công (fg_scale + bám người)"
```

---

### Task 4: Control panel — slider "Cỡ video A" + reset

**Files:**
- Modify: `src/khunghinh/ui/control_panel.py`

**Interfaces:**
- Consumes: `_ZoomSlider` (đã có trong file).
- Produces (dùng ở Task 6): signals `fgScaleChanged = pyqtSignal(float)`, `fgResetRequested = pyqtSignal()`; widget `self.sld_fg_scale` (_ZoomSlider); method `set_fg_scale(v: float)` (đồng bộ slider khi gizmo đổi); slider + nút ẩn/hiện theo checkbox nền mờ.

- [ ] **Step 1: Thêm signals** — trong `ControlPanel`, cạnh `backgroundModeChanged`:

```python
    fgScaleChanged = pyqtSignal(float)   # cỡ video A trên nền mờ
    fgResetRequested = pyqtSignal()      # reset cỡ về 1.0
```

- [ ] **Step 2: Thêm widget vào nhóm "Nền"** — sau `blay.addWidget(bg_hint)` (nhận `fg_min/fg_max/fg_def` qua `__init__`; xem Step 4):

```python
        self.sld_fg_scale = _ZoomSlider("Cỡ video A", fg_min, fg_max, fg_def)
        self.btn_fg_reset = QPushButton("Đặt lại cỡ")
        self.sld_fg_scale.setVisible(False)
        self.btn_fg_reset.setVisible(False)
        blay.addWidget(self.sld_fg_scale)
        blay.addWidget(self.btn_fg_reset)
```

- [ ] **Step 3: Nối tín hiệu** — trong phần `# --- Wiring ---`, thêm:

```python
        self.sld_fg_scale.valueChanged.connect(self.fgScaleChanged)
        self.btn_fg_reset.clicked.connect(self.fgResetRequested)
```

- [ ] **Step 4: Hiện/ẩn theo nền mờ + method đồng bộ** — sửa `_on_blur_bg_toggle` và thêm `set_fg_scale`:

```python
    def _on_blur_bg_toggle(self, checked: bool) -> None:
        is_blur = bool(checked)
        self.sld_fg_scale.setVisible(is_blur)
        self.btn_fg_reset.setVisible(is_blur)
        self.backgroundModeChanged.emit(is_blur)

    def set_fg_scale(self, v: float) -> None:
        """Đồng bộ slider khi cỡ đổi từ nơi khác (gizmo) — không phát lại vòng lặp."""
        self.sld_fg_scale.blockSignals(True)
        self.sld_fg_scale.set_value(v)
        self.sld_fg_scale.blockSignals(False)
```

- [ ] **Step 5: Nhận tham số fg_* trong `__init__`** — đổi chữ ký và truyền từ caller. Sửa dòng `def __init__(self, zoom_min, zoom_max, zoom_def, parent=None):` thành:

```python
    def __init__(self, zoom_min: float, zoom_max: float, zoom_def: float,
                 fg_min: float = 0.3, fg_max: float = 3.0, fg_def: float = 1.0, parent=None):
```

- [ ] **Step 6: Kiểm tra import**

Run: `PYTHONPATH=src python -c "import khunghinh.ui.control_panel; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add src/khunghinh/ui/control_panel.py
git commit -m "feat(ui): control panel thêm slider Cỡ video A + reset cho nền mờ"
```

---

### Task 5: Preview — hiển thị canvas compose + gizmo kéo góc để scale

**Files:**
- Modify: `src/khunghinh/ui/preview_view.py`

**Interfaces:**
- Consumes: `ndarray_to_qpixmap` (đã có).
- Produces (dùng ở Task 6): `PreviewView.set_compose(canvas_bgr: np.ndarray, fg_x:int, fg_y:int, fg_w:int, fg_h:int, scale:float)`; signals `fgScaleDragged = pyqtSignal(float)` (scale mới khi kéo góc) + `fgResetClicked = pyqtSignal()`; method `set_compose_mode(on: bool)` để bật/tắt gizmo compose (ẩn overlay crop cũ).

- [ ] **Step 1: Thêm item gizmo + signals + method.** Sau class `_CropOverlay`, thêm helper khoảng cách + class `_ComposeGizmo(QGraphicsItem)`:

```python
def _pt_dist(a: QPointF, b: QPointF) -> float:
    """Khoảng cách Euclid giữa 2 QPointF (không dùng manhattanLength — không có ở mọi bản Qt)."""
    return ((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5


class _ComposeGizmo(QGraphicsItem):
    """Gizmo compose: viền quanh foreground + 4 tay nắm góc (kéo = scale) + nút reset.

    Toạ độ theo pixel CANVAS. Kéo góc đổi cỡ theo tỉ lệ khoảng cách góc→tâm foreground,
    phát fgScaleDragged(scale_mới). Bấm nút tròn dưới -> fgResetClicked.
    """

    HANDLE = 9  # bán kính tay nắm (px canvas)

    def __init__(self, view: "PreviewView"):
        super().__init__()
        self._view = view
        self._cw, self._ch = 1, 1
        self._fg = QRectF(0, 0, 1, 1)   # rect foreground trên canvas (có thể tràn)
        self._cur_scale = 1.0
        self._drag = False
        self._half0 = 1.0               # nửa đường chéo fg lúc bắt đầu kéo
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def set_canvas(self, cw: int, ch: int) -> None:
        self.prepareGeometryChange()
        self._cw, self._ch = max(1, cw), max(1, ch)

    def set_fg(self, x: int, y: int, w: int, h: int, scale: float) -> None:
        self._fg = QRectF(x, y, w, h)
        self._cur_scale = scale
        self.update()

    def _reset_center(self) -> QPointF:
        return QPointF(self._cw / 2, self._ch + self.HANDLE * 3)

    def boundingRect(self) -> QRectF:
        r = self._fg.adjusted(-self.HANDLE, -self.HANDLE, self.HANDLE, self.HANDLE)
        return r.united(QRectF(0, 0, self._cw, self._ch)).united(
            QRectF(self._reset_center() - QPointF(self.HANDLE, self.HANDLE),
                   self._reset_center() + QPointF(self.HANDLE, self.HANDLE)))

    def paint(self, painter, option, widget=None) -> None:  # noqa: ANN001
        pen = QPen(QColor(255, 255, 255), max(2, int(self._cw / 400)))
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(self._fg)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        for c in (self._fg.topLeft(), self._fg.topRight(),
                  self._fg.bottomLeft(), self._fg.bottomRight()):
            painter.drawEllipse(c, self.HANDLE, self.HANDLE)
        rc = self._reset_center()
        painter.setBrush(QBrush(QColor(30, 32, 40)))
        painter.drawEllipse(rc, self.HANDLE + 2, self.HANDLE + 2)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawArc(QRectF(rc.x() - self.HANDLE, rc.y() - self.HANDLE,
                               self.HANDLE * 2, self.HANDLE * 2), 30 * 16, 300 * 16)

    def _near(self, a: QPointF, b: QPointF, r: float) -> bool:
        return _pt_dist(a, b) <= r * 1.6

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        pos = event.pos()
        if self._near(pos, self._reset_center(), self.HANDLE + 2):
            self._view.emit_fg_reset()
            event.accept()
            return
        corners = (self._fg.topLeft(), self._fg.topRight(),
                   self._fg.bottomLeft(), self._fg.bottomRight())
        if any(self._near(pos, c, self.HANDLE + 3) for c in corners):
            self._drag = True
            self._half0 = max(1.0, _pt_dist(self._fg.center(), pos))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._drag:
            half = max(1.0, _pt_dist(self._fg.center(), event.pos()))
            new_scale = self._cur_scale * (half / self._half0)
            self._view.emit_fg_scale(new_scale)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self._drag = False
        super().mouseReleaseEvent(event)
```

- [ ] **Step 2: Cập nhật `PreviewView`** — thêm signals sau `cropCenterChanged`:

```python
    fgScaleDragged = pyqtSignal(float)
    fgResetClicked = pyqtSignal()
```

Trong `__init__`, sau khi tạo `self._overlay`, thêm gizmo:

```python
        self._gizmo = _ComposeGizmo(self)
        self._scene.addItem(self._gizmo)
        self._gizmo.setVisible(False)
        self._compose_mode = False
```

- [ ] **Step 3: Thêm method compose + emit + đổi set_frame để tôn trọng compose_mode.**

```python
    def set_compose_mode(self, on: bool) -> None:
        self._compose_mode = on
        self._overlay.setVisible(self._overlay_wanted and not on)
        self._gizmo.setVisible(on)

    def set_compose(self, canvas_bgr, fg_x, fg_y, fg_w, fg_h, scale) -> None:  # noqa: ANN001
        h, w = canvas_bgr.shape[:2]
        self._pixmap_item.setPixmap(ndarray_to_qpixmap(canvas_bgr))
        self._gizmo.set_canvas(w, h)
        self._gizmo.set_fg(fg_x, fg_y, fg_w, fg_h, scale)
        rect = self._gizmo.boundingRect()
        self._scene.setSceneRect(rect)     # mở rộng để thấy tay nắm tràn ra ngoài canvas
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def emit_fg_scale(self, scale: float) -> None:
        self.fgScaleDragged.emit(float(scale))

    def emit_fg_reset(self) -> None:
        self.fgResetClicked.emit()
```

Sửa `set_frame` để KHÔNG bật lại overlay khi đang compose: đổi dòng `self._overlay.setVisible(self._overlay_wanted)` thành:

```python
        self._overlay.setVisible(self._overlay_wanted and not self._compose_mode)
```

- [ ] **Step 4: Kiểm tra import**

Run: `PYTHONPATH=src python -c "import khunghinh.ui.preview_view; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/khunghinh/ui/preview_view.py
git commit -m "feat(ui): preview hiển thị canvas compose + gizmo kéo góc để scale video A"
```

---

### Task 6: Main window — nối state, preview compose, export

**Files:**
- Modify: `src/khunghinh/ui/main_window.py`

**Interfaces:**
- Consumes: `composite_manual_on_blurred_background`, `place_foreground` (Task 1,2); `ExportSettings.fg_scale` (Task 3); `ControlPanel.fgScaleChanged/fgResetRequested/set_fg_scale` (Task 4); `PreviewView.set_compose/set_compose_mode/fgScaleDragged/fgResetClicked` (Task 5).
- Produces: nối chuỗi hoàn chỉnh.

- [ ] **Step 1: Import + state.** Sửa import compositing:

```python
from ..core.compositing import composite_manual_on_blurred_background, place_foreground
```

Trong `__init__`, cạnh `self._blur_bg = False`, thêm:

```python
        self._fg_scale = self.config.fg_scale_default
```

Truyền dải fg vào ControlPanel — sửa dòng tạo `self.controls = ControlPanel(...)`:

```python
        self.controls = ControlPanel(
            config.zoom_min, config.zoom_max, config.zoom_default,
            config.fg_scale_min, config.fg_scale_max, config.fg_scale_default,
        )
```

- [ ] **Step 2: Nối tín hiệu** — cạnh `self.controls.backgroundModeChanged.connect(...)`:

```python
        self.controls.fgScaleChanged.connect(self.on_fg_scale_changed)
        self.controls.fgResetRequested.connect(self.on_fg_reset)
        self.preview.fgScaleDragged.connect(self.on_fg_scale_dragged)
        self.preview.fgResetClicked.connect(self.on_fg_reset)
```

- [ ] **Step 3: Handler scale + reset + bật compose mode.** Thêm methods; và sửa `on_background_mode_changed`:

```python
    def on_background_mode_changed(self, is_blur: bool) -> None:
        self._blur_bg = is_blur
        self.preview.set_compose_mode(is_blur)
        self._redraw_for_frame(self._cur_frame)
        self.statusBar().showMessage(
            "Nền mờ: video A đặt lên nền mờ (kéo góc để chỉnh cỡ)." if is_blur else "Nền mờ: tắt."
        )

    def on_fg_scale_changed(self, v: float) -> None:
        self._fg_scale = float(v)
        self._redraw_for_frame(self._cur_frame)

    def on_fg_scale_dragged(self, v: float) -> None:
        v = max(self.config.fg_scale_min, min(float(v), self.config.fg_scale_max))
        self._fg_scale = v
        self.controls.set_fg_scale(v)      # đồng bộ slider
        self._redraw_for_frame(self._cur_frame)

    def on_fg_reset(self) -> None:
        self._fg_scale = self.config.fg_scale_default
        self.controls.set_fg_scale(self._fg_scale)
        self._redraw_for_frame(self._cur_frame)
```

- [ ] **Step 4: Preview compose trong `_redraw_for_frame`.** Thêm nhánh blur ở đầu (sau khi `frame` đã đọc, trước nhánh auto/manual hiện có):

```python
        if self._blur_bg:
            h, w = frame.shape[:2]
            tw, th = self.config.target_width, self.config.target_height
            if self._mode == "auto" and self._analysis is not None:
                cx, cy = self._analysis.centers_for_frame(i)
            else:
                cx, cy = w / 2.0, h / 2.0
            canvas = composite_manual_on_blurred_background(
                frame, tw, th, self._fg_scale, cx / w, cy / h,
                downscale_divisor=self.config.bg_blur_downscale_divisor,
                dim=self.config.bg_blur_dim,
            )
            p = place_foreground(w, h, tw, th, self._fg_scale, cx / w, cy / h)
            self.preview.set_compose(canvas, p.x, p.y, p.fg_w, p.fg_h, self._fg_scale)
            return
```

(Đặt khối này NGAY sau `if frame is None: return` trong `_redraw_for_frame`, trước phần vẽ crop hiện tại. Đồng thời XOÁ comment cũ "Nền mờ chỉ ảnh hưởng bước XUẤT video…" ở dưới — không còn đúng vì giờ nền mờ có preview compose riêng. `centers_for_frame(i)` đã có sẵn trên `AnalysisResult`.)

- [ ] **Step 5: Truyền fg_scale khi export.** Trong `on_export`, thêm `fg_scale=self._fg_scale` vào `ExportSettings(...)` (cạnh `blurred_background=self._blur_bg`).

- [ ] **Step 6: Kiểm tra import + full suite**

Run: `PYTHONPATH=src python -c "import khunghinh.ui.main_window; print('ok')"`
Expected: `ok`.
Run: `python -m pytest -q`
Expected: toàn bộ pass.

- [ ] **Step 7: Commit**

```bash
git add src/khunghinh/ui/main_window.py
git commit -m "feat(ui): nối compose nền mờ — preview canvas + gizmo + export fg_scale"
```

---

### Task 7: Cập nhật README + build lại exe + verify

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Full suite + đếm test**

Run: `python -m pytest -q`
Expected: pass (105 cũ + 9 test compositing mới + 1 exporter = 115). Ghi lại số N thực tế.

- [ ] **Step 2: Cập nhật mô tả Nền mờ trong `README.md`** — thay bullet "🌫️ Nền mờ" hiện tại bằng:

```
- 🌫️ **Nền mờ (compose kiểu CapCut)** — đặt video A (nguyên khung) lên nền mờ của
  chính nó: cỡ chỉnh tay (kéo góc / slider "Cỡ video A"), vị trí tự bám người nói
  (pan ngang giữ người trong khung, cắt 2 rìa, letterbox mờ trên/dưới).
```

- [ ] **Step 3: Cập nhật số test** — đổi mọi chỗ "105 unit test"/"105 test" trong README thành N (từ Step 1).

- [ ] **Step 4: Build lại exe**

Run:
```bash
python -m PyInstaller --noconfirm --clean --name KhungHinh916 --onedir --console \
  --paths src --collect-data cv2 --collect-submodules khunghinh \
  --exclude-module onnxruntime --exclude-module matplotlib --exclude-module scipy --exclude-module pandas \
  run.py
```
Expected: `Build complete!`. Exe tại `dist/KhungHinh916/KhungHinh916.exe`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: mô tả chế độ nền mờ compose (CapCut) + cập nhật số test"
```

---

## Final Verification

- [ ] `python -m pytest -q` từ gốc repo — toàn bộ pass.
- [ ] Chạy exe `dist/KhungHinh916/KhungHinh916.exe`: nhập `demo_video.mp4`, bật **Nền mờ** → preview hiện canvas 9:16 với video A trên nền mờ + gizmo; kéo góc / slider "Cỡ video A" đổi cỡ; nút reset về vừa khít; bật **Tự động** + Phân tích → foreground bám người khi phóng to; xuất video → mở kiểm tra 1080×1920, video A trên nền mờ đúng cỡ, bám người.
