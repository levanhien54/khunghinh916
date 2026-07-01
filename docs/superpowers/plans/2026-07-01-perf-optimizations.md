# Tối ưu tốc độ xử lý (gói #1/#5/#2/#4/#6) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Giảm thời gian xử lý (xuất video + phân tích auto) mà không đổi kết quả nhìn thấy được, dựa trên bản đánh giá hiệu năng đã xác minh đối kháng.

**Architecture:** Hai khâu nặng, mỗi khâu 1 lượt giải mã. XUẤT: bỏ encode 2 lần (mp4v tạm → re-encode) bằng cách pipe frame BGR thô thẳng vào một tiến trình ffmpeg (rawvideo → libx264), thêm `-preset veryfast`, và bỏ bước dựng nền mờ luôn-bị-đè. PHÂN TÍCH: detect khuôn mặt theo stride (config, mặc định giữ nguyên) + overlap giải mã với tính toán bằng producer/consumer.

**Tech Stack:** Python 3.9+, OpenCV (`cv2`), NumPy, ffmpeg (subprocess/pipe), PyQt6 (QThread), pytest. Không thêm dependency.

## Global Constraints

- CPU-only, không GPU. Không thêm dependency (ffmpeg/opencv/numpy/stdlib đã có).
- Giữ đúng 88 test hiện có xanh; thêm test mới cho phần logic thuần (cmd-building, compositing, config).
- Giữ phong cách code hiện có: docstring/comment tiếng Việt, `from __future__ import annotations`.
- Đây KHÔNG phải git repo (`git status` fail) → bỏ mọi bước `git add/commit` trừ khi người dùng đã `git init`.
- ffmpeg CÓ trên PATH ở máy này (`C:\checkvideo_tools\ffmpeg.exe`) → chạy được smoke-test xuất thật với `demo_video.mp4`.
- Chạy test từ repo root: `python -m pytest -q`. Với import ngoài pytest: `$env:PYTHONPATH = "src"` trước.
- Mọi thay đổi hành vi có đánh đổi (preset → file to hơn; stride>1 → chất lượng giảm nhẹ) phải để MẶC ĐỊNH giữ nguyên chất lượng hiện tại và cho tinh chỉnh qua config.

## Thứ tự & phụ thuộc

1. Task 1 (#1 preset) tạo helper `_video_encode_flags()` mà Task 3 (#2 pipe) dùng lại → làm trước.
2. Task 2 (#5 short-circuit blur) độc lập.
3. Task 3 (#2 pipe ffmpeg) refactor lớn exporter, dùng helper của Task 1.
4. Task 4 (#4 detect_stride) và Task 5 (#6 producer/consumer) đều sửa `analysis_worker._analyze` — làm #4 trước, #6 sau (compose được).

---

### Task 1: #1 — Thêm knob `preset` + `-preset veryfast` cho encode ffmpeg

**Files:**
- Modify: `src/khunghinh/config.py` (thêm field)
- Modify: `src/khunghinh/mediaio/exporter.py` (ExportSettings field + helper + dùng trong `_mux_audio`)
- Modify: `src/khunghinh/ui/main_window.py` (wire config → ExportSettings)
- Test: `tests/test_exporter.py` (mới)

**Interfaces:**
- Produces (Task 3 dùng): `VideoExporter._video_encode_flags() -> list[str]` trả về `["-c:v", codec, ("-preset", preset nếu x264/x265), "-crf", str(crf), "-pix_fmt", "yuv420p"]`.
- `AppConfig.export_preset: str = "veryfast"`; `ExportSettings.preset: str = "veryfast"`.

- [ ] **Step 1: Viết test thất bại** — tạo `tests/test_exporter.py`:

```python
from __future__ import annotations

from khunghinh.mediaio.exporter import ExportSettings, VideoExporter


def _exporter(**kw) -> VideoExporter:
    settings = ExportSettings(out_path="out.mp4", **kw)
    # reader/engine không cần cho việc dựng cmd — truyền None, chỉ gọi helper thuần.
    return VideoExporter(reader=None, engine=None, settings=settings)


def test_encode_flags_include_preset_for_libx264():
    flags = _exporter(codec="libx264", preset="veryfast", crf=18)._video_encode_flags()
    assert "-preset" in flags
    assert flags[flags.index("-preset") + 1] == "veryfast"
    assert "-c:v" in flags and flags[flags.index("-c:v") + 1] == "libx264"
    assert "-crf" in flags and flags[flags.index("-crf") + 1] == "18"
    assert flags[flags.index("-pix_fmt") + 1] == "yuv420p"


def test_encode_flags_include_preset_for_libx265():
    flags = _exporter(codec="libx265", preset="faster")._video_encode_flags()
    assert "-preset" in flags and flags[flags.index("-preset") + 1] == "faster"


def test_encode_flags_omit_preset_for_non_x26x_codec():
    flags = _exporter(codec="mpeg4", preset="veryfast")._video_encode_flags()
    assert "-preset" not in flags  # -preset là cờ riêng của x264/x265
    assert flags[flags.index("-c:v") + 1] == "mpeg4"
```

- [ ] **Step 2: Chạy test → FAIL**

Run: `python -m pytest tests/test_exporter.py -q`
Expected: FAIL — `AttributeError: 'VideoExporter' object has no attribute '_video_encode_flags'` (và `ExportSettings` chưa có `preset`).

- [ ] **Step 3: Thêm `export_preset` vào `config.py`**

Trong `src/khunghinh/config.py`, ngay dưới `export_codec`:

```python
    export_crf: int = 18          # H.264: thấp hơn = nét hơn (18 ~ "thị giác không mất mát")
    export_codec: str = "libx264"
    export_preset: str = "veryfast"   # x264/x265: nhanh hơn 'medium' 3-5x, size to hơn chút
```

- [ ] **Step 4: Thêm `preset` field + helper vào `exporter.py`**

Trong `ExportSettings`, thêm sau `codec`:

```python
    codec: str = "libx264"
    preset: str = "veryfast"
    copy_audio: bool = True
```

Thêm method vào `VideoExporter` (đặt ngay trên `run`):

```python
    def _video_encode_flags(self) -> list[str]:
        """Cờ encode video cho ffmpeg. -preset chỉ hợp lệ với x264/x265."""
        flags = ["-c:v", self.settings.codec]
        if self.settings.codec.startswith("libx26"):
            flags += ["-preset", self.settings.preset]
        flags += ["-crf", str(self.settings.crf), "-pix_fmt", "yuv420p"]
        return flags
```

- [ ] **Step 5: Dùng helper trong `_mux_audio`**

Thay khối `cmd = [...]` trong `_mux_audio`:

```python
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_video),
            "-i", src_video,
            "-map", "0:v:0", "-map", "1:a:0?",
            *self._video_encode_flags(),
            "-c:a", "aac", "-shortest",
            out_path,
        ]
```

- [ ] **Step 6: Chạy test → PASS**

Run: `python -m pytest tests/test_exporter.py -q`
Expected: PASS (3 passed).

- [ ] **Step 7: Wire config → ExportSettings trong `main_window.py`**

Trong `on_export`, thêm `preset=self.config.export_preset,` vào `ExportSettings(...)` ngay sau `codec=self.config.export_codec,`.

- [ ] **Step 8: Chạy toàn bộ suite**

Run: `python -m pytest -q`
Expected: tất cả pass (88 cũ + 3 mới = 91).

---

### Task 2: #5 — Short-circuit nền mờ (bỏ compute luôn bị đè)

**Files:**
- Modify: `src/khunghinh/core/compositing.py` (`composite_crop_on_blurred_background`)
- Test: `tests/test_compositing.py` (thêm 1 test)

**Interfaces:**
- `composite_crop_on_blurred_background(frame, rect, target_w, target_h, downscale_divisor=32, dim=0.55)` giữ nguyên chữ ký; nội bộ KHÔNG còn gọi `make_blurred_background`. Output byte-identical với `crop_and_resize`.

- [ ] **Step 1: Viết test thất bại** — thêm vào cuối `tests/test_compositing.py`:

```python
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
```

- [ ] **Step 2: Chạy test → FAIL**

Run: `python -m pytest tests/test_compositing.py::test_composite_crop_skips_blur_compute -q`
Expected: FAIL — `AssertionError: make_blurred_background KHÔNG được gọi` (hàm hiện vẫn gọi nó).

- [ ] **Step 3: Short-circuit hàm**

Thay thân `composite_crop_on_blurred_background` trong `src/khunghinh/core/compositing.py`:

```python
def composite_crop_on_blurred_background(
    frame: np.ndarray,
    rect: CropRect,
    target_w: int,
    target_h: int,
    downscale_divisor: int = 32,
    dim: float = 0.55,
) -> np.ndarray:
    """Trả về khung hình đã cắt phủ kín canvas.

    Vì crop luôn resize đúng (target_w, target_h) — phủ kín 100% canvas ở tỉ lệ 9:16 —
    nên lớp nền mờ (nếu dựng) sẽ bị đè hoàn toàn, không bao giờ hiển thị. Bỏ hẳn việc
    dựng nền mờ để tiết kiệm CPU mỗi frame (tối ưu tốc độ đã xác nhận). Giữ tham số
    downscale_divisor/dim để tương thích chữ ký với nơi gọi (exporter).
    """
    return crop_and_resize(frame, rect, target_w, target_h)
```

- [ ] **Step 4: Chạy test → PASS + toàn bộ compositing**

Run: `python -m pytest tests/test_compositing.py -q`
Expected: PASS (14 passed — 13 cũ + 1 mới; test byte-equality cũ vẫn xanh).

---

### Task 3: #2 — Pipe frame BGR thô vào ffmpeg (bỏ encode 2 lần)

**Files:**
- Modify: `src/khunghinh/mediaio/exporter.py` (refactor `run`, thêm `_iter_output_frames`, `_build_pipe_cmd`, `_run_ffmpeg_pipe`, `_run_mp4v_fallback`; bỏ `_mux_audio`, `_safe_rmtree`, `tempfile`)
- Test: `tests/test_exporter.py` (thêm test cmd)

**Interfaces:**
- Consumes: `VideoExporter._video_encode_flags()` (Task 1).
- Produces: `VideoExporter._build_pipe_cmd(tw, th, fps, src_video) -> list[str]`. `run(...)` signature không đổi (vẫn `center_provider, smooth, progress_cb, cancel_cb`).

- [ ] **Step 1: Viết test thất bại** — thêm vào `tests/test_exporter.py`:

```python
def test_build_pipe_cmd_rawvideo_and_audio_map():
    cmd = _exporter(target_width=1080, target_height=1920, codec="libx264",
                    preset="veryfast", crf=20)._build_pipe_cmd(1080, 1920, 30.0, "in.mp4")
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"          # input pix_fmt
    assert "-s" in cmd and cmd[cmd.index("-s") + 1] == "1080x1920"
    assert cmd.count("-i") == 2                                # stdin '-' + src
    assert "-map" in cmd                                       # có map audio nguồn
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "veryfast"
    assert cmd[-1] == "out.mp4"                                # out_path cuối cùng
```

- [ ] **Step 2: Chạy test → FAIL**

Run: `python -m pytest tests/test_exporter.py::test_build_pipe_cmd_rawvideo_and_audio_map -q`
Expected: FAIL — `AttributeError: ... '_build_pipe_cmd'`.

- [ ] **Step 3: Refactor `exporter.py`** — thay imports đầu file:

```python
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from ..core.compositing import composite_crop_on_blurred_background, crop_and_resize
from ..core.reframe_engine import ReframeEngine
from .reader import VideoReader
```

- [ ] **Step 4: Thay toàn bộ `run` + `_mux_audio` + `_safe_rmtree`** bằng:

```python
    def run(
        self,
        center_provider: CenterProvider | None = None,
        smooth: bool = True,
        progress_cb: ProgressCb | None = None,
        cancel_cb: CancelCb | None = None,
    ) -> str:
        info = self.reader.info or self.reader.open()
        tw, th = self.settings.target_width, self.settings.target_height
        fps = info.fps or 30.0
        dt = 1.0 / fps
        self.engine.reset()
        self.reader.rewind()
        frames = self._iter_output_frames(tw, th, dt, center_provider, smooth,
                                          progress_cb, cancel_cb, info.frame_count)
        if self.settings.copy_audio and ffmpeg_available():
            return self._run_ffmpeg_pipe(frames, tw, th, fps, info.path)
        return self._run_mp4v_fallback(frames, tw, th, fps)

    def _iter_output_frames(
        self, tw: int, th: int, dt: float,
        center_provider: CenterProvider | None, smooth: bool,
        progress_cb: ProgressCb | None, cancel_cb: CancelCb | None, total: int,
    ) -> Iterator[np.ndarray]:
        idx = 0
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
            yield out_frame
            idx += 1
            if progress_cb and total:
                progress_cb(idx, total)

    def _build_pipe_cmd(self, tw: int, th: int, fps: float, src_video: str) -> list[str]:
        """ffmpeg đọc frame BGR thô từ stdin, ghép audio nguồn, encode 1 lần sang H.264."""
        return [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{tw}x{th}", "-r", str(fps),
            "-i", "-",
            "-i", src_video,
            "-map", "0:v:0", "-map", "1:a:0?",
            *self._video_encode_flags(),
            "-c:a", "aac", "-shortest",
            self.settings.out_path,
        ]

    def _run_ffmpeg_pipe(self, frames: Iterator[np.ndarray], tw: int, th: int,
                         fps: float, src_video: str) -> str:
        out_path = self.settings.out_path
        cmd = self._build_pipe_cmd(tw, th, fps, src_video)
        log.debug("ffmpeg pipe: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stderr_tail: list[bytes] = []

        def _drain() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_tail.append(line)
                del stderr_tail[:-40]  # giữ ~40 dòng cuối để báo lỗi

        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

        n = 0
        try:
            assert proc.stdin is not None
            for out_frame in frames:
                proc.stdin.write(np.ascontiguousarray(out_frame).tobytes())
                n += 1
            proc.stdin.close()
        except InterruptedError:
            proc.kill()
            proc.wait()
            drainer.join(timeout=2.0)
            _safe_remove(out_path)
            raise
        except BrokenPipeError:
            proc.wait()
            drainer.join(timeout=2.0)
            raise RuntimeError("ffmpeg đóng pipe sớm — xem logs/khunghinh.log.")
        ret = proc.wait()
        drainer.join(timeout=2.0)
        if ret != 0:
            log.error("ffmpeg lỗi:\n%s", b"".join(stderr_tail).decode("utf-8", "replace")[-1500:])
            _safe_remove(out_path)
            raise RuntimeError("ffmpeg thất bại — xem logs/khunghinh.log.")
        log.info("Xuất xong (pipe): %s (%d frame).", out_path, n)
        return out_path

    def _run_mp4v_fallback(self, frames: Iterator[np.ndarray], tw: int, th: int,
                           fps: float) -> str:
        """Không có ffmpeg → ghi mp4v thẳng ra out_path (KHÔNG audio)."""
        out_path = self.settings.out_path
        log.warning("Không thấy ffmpeg — xuất mp4v KHÔNG audio: %s", out_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (tw, th))
        if not writer.isOpened():
            raise IOError("Không khởi tạo được VideoWriter (kiểm tra codec mp4v).")
        n = 0
        try:
            for out_frame in frames:
                writer.write(out_frame)
                n += 1
        finally:
            writer.release()
        log.info("Xuất xong (mp4v): %s (%d frame).", out_path, n)
        return out_path


def _safe_remove(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
```

Lưu ý: xoá hẳn `_mux_audio`, `_safe_rmtree`, và import `tempfile`. Hàm module-level `_safe_remove` thay `_safe_rmtree`.

- [ ] **Step 5: Chạy test cmd → PASS**

Run: `python -m pytest tests/test_exporter.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Smoke-test xuất thật** (ffmpeg có trên máy) — tạo `scratchpad/smoke_export.py` trong scratch dir và chạy:

Run:
```
python -c "import sys; sys.path.insert(0,'src'); \
from khunghinh.mediaio.reader import VideoReader; \
from khunghinh.core.reframe_engine import ReframeEngine, ReframeParams; \
from khunghinh.core.smoothing import CameraSmoother; \
from khunghinh.mediaio.exporter import VideoExporter, ExportSettings; \
r=VideoReader('demo_video.mp4'); info=r.open(); \
eng=ReframeEngine(ReframeParams(info.width,info.height,1080/1920,1.0,1.0,0.5,0.5), CameraSmoother(1.0,0.05)); \
exp=VideoExporter(r, eng, ExportSettings(out_path='scratch_out.mp4')); \
p=exp.run(); r.release(); \
import os; print('OUT bytes:', os.path.getsize(p))"
```
Expected: in ra `OUT bytes: <số > 0>`, không lỗi. Sau đó mở lại kiểm tra:
```
python -c "import cv2; c=cv2.VideoCapture('scratch_out.mp4'); \
print('opened', c.isOpened(), 'W', int(c.get(3)), 'H', int(c.get(4)), 'N', int(c.get(7)))"
```
Expected: `opened True W 1080 H 1920 N <>0>`. Xoá file: `python -c "import os; os.remove('scratch_out.mp4')"`.

- [ ] **Step 7: Chạy toàn bộ suite**

Run: `python -m pytest -q`
Expected: tất cả pass (91).

---

### Task 4: #4 — Detect khuôn mặt theo stride (config, mặc định = 1)

**Files:**
- Modify: `src/khunghinh/config.py` (thêm `detect_stride: int = 1`)
- Modify: `src/khunghinh/ui/analysis_worker.py` (vòng lặp `_analyze`)
- Test: `tests/test_config.py` (mới, nhẹ)

**Interfaces:**
- `AppConfig.detect_stride: int = 1` (1 = detect mỗi frame, giữ nguyên hành vi hiện tại).

- [ ] **Step 1: Viết test thất bại** — tạo `tests/test_config.py`:

```python
from __future__ import annotations

from khunghinh.config import AppConfig


def test_detect_stride_default_is_one():
    assert AppConfig().detect_stride == 1


def test_export_preset_default():
    assert AppConfig().export_preset == "veryfast"
```

- [ ] **Step 2: Chạy test → FAIL**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'detect_stride'`.

- [ ] **Step 3: Thêm field vào `config.py`**

Trong khối `# --- Phát hiện khuôn mặt ---`, thêm:

```python
    face_detect_width: int = 640        # downscale trước khi detect (tốc độ/độ chính xác)
    detect_stride: int = 1              # detect mỗi N frame (1=mỗi frame); >1 = nhanh hơn, coast track giữa các frame
```

- [ ] **Step 4: Sửa vòng lặp trong `analysis_worker._analyze`**

Ngay trước `reader.rewind()` (khoảng dòng 95), thêm khởi tạo:

```python
            detect_stride = max(1, int(cfg.detect_stride))
            last_faces: list = []
```

Trong vòng `while True`, thay dòng:

```python
                faces = tracker.update(detector.detect(frame))
```

bằng:

```python
                # Detect theo stride: quỹ đạo camera đã làm mượt mạnh ở Pass 2 nên bỏ
                # bớt detect ít ảnh hưởng; giữa các frame "coast" bằng cách tái dùng
                # danh sách khuôn mặt gần nhất. VAD/môi/cắt-cảnh vẫn tính MỖI frame.
                if i % detect_stride == 0:
                    last_faces = tracker.update(detector.detect(frame))
                faces = last_faces
```

(compute_features_step, scene-cut reset, selector.update giữ nguyên MỖI frame.)

- [ ] **Step 5: Chạy test config → PASS + import worker**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS (2 passed).

Run: `$env:PYTHONPATH = "src"; python -c "import khunghinh.ui.analysis_worker; print('worker OK')"`
Expected: `worker OK`.

- [ ] **Step 6: Chạy toàn bộ suite**

Run: `python -m pytest -q`
Expected: tất cả pass (93 = 91 + 2 config). Với `detect_stride=1` mặc định, hành vi loop identical.

---

### Task 5: #6 — Overlap giải mã với tính toán (producer/consumer)

**Files:**
- Modify: `src/khunghinh/ui/analysis_worker.py` (`_analyze`: producer thread + hàng đợi có chặn)

**Interfaces:**
- Không đổi interface công khai. Nội bộ: một `threading.Thread` producer đọc `reader.read_next()` đẩy vào `queue.Queue(maxsize=8)`; consumer (vòng chính) lấy ra xử lý. FIFO 1-producer-1-consumer → thứ tự frame bất biến → kết quả bit-identical.

- [ ] **Step 1: Thêm import** vào đầu `analysis_worker.py`:

```python
import logging
import queue
import threading
```

- [ ] **Step 2: Bọc vòng consumer bằng producer thread**

Thay khối từ `reader.rewind()` đến hết vòng `while True` (phần đọc frame) bằng cấu trúc producer/consumer. Cụ thể, thay:

```python
            detect_stride = max(1, int(cfg.detect_stride))
            last_faces: list = []

            reader.rewind()
            i = 0
            total = max(1, info.frame_count)
            last_reset_at = -cfg.cut_min_scene_len  # cho phép reset ngay tại frame 0 nếu cần
            while True:
                if self._cancel:
                    raise InterruptedError()
                frame = reader.read_next()
                if frame is None:
                    break
```

bằng:

```python
            detect_stride = max(1, int(cfg.detect_stride))
            last_faces: list = []

            reader.rewind()
            i = 0
            total = max(1, info.frame_count)
            last_reset_at = -cfg.cut_min_scene_len  # cho phép reset ngay tại frame 0 nếu cần

            # Producer đọc frame (giải mã) song song với consumer (detect/track/ASD),
            # giấu thời gian giải mã sau tính toán. 1 producer + 1 consumer + hàng đợi
            # FIFO có chặn ⇒ thứ tự frame bất biến ⇒ kết quả không đổi. cv2.VideoCapture
            # chỉ được chạm bởi DUY NHẤT producer (không an toàn đa luồng).
            frame_q: "queue.Queue" = queue.Queue(maxsize=8)
            stop_evt = threading.Event()
            producer_exc: list = []

            def _produce() -> None:
                try:
                    while not stop_evt.is_set():
                        fr = reader.read_next()
                        if fr is None:
                            break
                        while not stop_evt.is_set():
                            try:
                                frame_q.put(fr, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                except Exception as exc:  # noqa: BLE001
                    producer_exc.append(exc)
                finally:
                    try:
                        frame_q.put_nowait(None)  # sentinel EOF
                    except queue.Full:
                        pass

            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()
            try:
                while True:
                    if self._cancel:
                        raise InterruptedError()
                    frame = frame_q.get()
                    if frame is None:
                        break
```

- [ ] **Step 3: Đóng producer an toàn sau vòng lặp**

Vòng `while` xử lý frame giữ NGUYÊN thân (detect stride, features, ASD, append...) nhưng lùi thêm 1 cấp thụt lề (vì nay nằm trong `try`). Ngay sau khi vòng `while` kết thúc (dòng `n = i`), đóng producer. Sửa đoạn:

```python
            n = i
            self.stage.emit("Tính quỹ đạo camera…")
```

thành:

```python
            finally:
                stop_evt.set()
                # rút cạn hàng đợi để producer (nếu đang chặn ở put) thoát được
                try:
                    while True:
                        frame_q.get_nowait()
                except queue.Empty:
                    pass
                producer.join(timeout=3.0)
            if producer_exc:
                raise producer_exc[0]

            n = i
            self.stage.emit("Tính quỹ đạo camera…")
```

Lưu ý thụt lề: toàn bộ thân vòng `while True` (từ `if self._cancel` cũ tới `i += 1`/progress) phải nằm trong `try:` ở Step 2; `finally:` ở Step 3 khớp với `try:` đó. `reader.release()` vẫn ở `finally` ngoài cùng của `_analyze` — chỉ chạy SAU khi producer đã join, nên không release cap dưới chân producer.

- [ ] **Step 4: Kiểm tra import + đọc lại logic thụt lề**

Run: `$env:PYTHONPATH = "src"; python -c "import khunghinh.ui.analysis_worker; print('worker OK')"`
Expected: `worker OK` (không SyntaxError/IndentationError).

- [ ] **Step 5: Smoke-test phân tích thật trên `demo_video.mp4`**

Tạo và chạy script (cần QApplication offscreen cho QThread/QObject):

Run:
```
$env:PYTHONPATH = "src"; $env:QT_QPA_PLATFORM = "offscreen"; \
python -c "import sys; from PyQt6.QtWidgets import QApplication; app=QApplication(sys.argv); \
from khunghinh.config import AppConfig; from khunghinh.ui.analysis_worker import AnalysisWorker; \
w=AnalysisWorker('demo_video.mp4', 1080/1920, AppConfig()); res=w._analyze(); \
print('frames', res.frame_count, 'cuts', len(res.scene_cut_frames), 'centers', res.centers_px.shape)"
```
Expected: in ra `frames <>0>`, `centers` khớp `(frames, 2)`, không lỗi/không treo.

- [ ] **Step 6: Smoke-test lại với `detect_stride=3`** (kiểm chứng #4 + #6 cùng hoạt động):

Run:
```
$env:PYTHONPATH = "src"; $env:QT_QPA_PLATFORM = "offscreen"; \
python -c "import sys; from PyQt6.QtWidgets import QApplication; app=QApplication(sys.argv); \
from khunghinh.config import AppConfig; from khunghinh.ui.analysis_worker import AnalysisWorker; \
cfg=AppConfig(); cfg.detect_stride=3; \
w=AnalysisWorker('demo_video.mp4', 1080/1920, cfg); res=w._analyze(); \
print('stride3 frames', res.frame_count)"
```
Expected: `stride3 frames <bằng số frame ở Step 5>` (stride chỉ đổi tần suất detect, không đổi số frame xử lý).

- [ ] **Step 7: Chạy toàn bộ suite**

Run: `python -m pytest -q`
Expected: tất cả pass (93). (Không có test worker nên suite không đổi; smoke-test là bằng chứng chạy thật.)

---

## Final Verification

- [ ] `python -m pytest -q` — toàn bộ pass (93).
- [ ] Smoke export (Task 3 Step 6) tạo mp4 1080×1920 hợp lệ, có audio (mở bằng ffprobe/cv2).
- [ ] Smoke analysis (Task 5 Step 5-6) trả AnalysisResult hợp lệ với stride 1 và 3.
- [ ] `python run.py` (nếu có màn hình): xuất 1 video ngắn, xác nhận nhanh hơn rõ và ảnh không lỗi.
- [ ] Rà `grep -rn "_mux_audio\|_safe_rmtree\|tempfile" src` → không còn tham chiếu cũ.
