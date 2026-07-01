"""Xuất video dọc 9:16: cắt từng frame → ghi video → ghép lại audio gốc bằng ffmpeg.

Phiên bản nền tảng dùng chế độ cắt TĨNH (tâm + zoom cố định do người dùng đặt).
Khi tích hợp ASD, chỉ cần truyền `center_provider(frame_idx) -> (cx, cy)` để cấp
tâm người nói theo từng frame — engine sẽ tự làm mượt.
"""
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

log = logging.getLogger(__name__)

ProgressCb = Callable[[int, int], None]      # (frame_đã_xử_lý, tổng_frame)
CancelCb = Callable[[], bool]                # trả True để hủy
CenterProvider = Callable[[int], "tuple[float, float]"]


@dataclass
class ExportSettings:
    out_path: str
    target_width: int = 1080
    target_height: int = 1920
    crf: int = 18
    codec: str = "libx264"
    preset: str = "veryfast"
    copy_audio: bool = True
    # Nền mờ: lớp nền an toàn vẽ trước rồi bị khung hình đã cắt đè kín lên trên
    # (xem docs/superpowers/specs/2026-07-01-blur-background-reframe-design.md).
    blurred_background: bool = False
    bg_blur_downscale_divisor: int = 32
    bg_blur_dim: float = 0.55


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _probe_audio_codec(src_video: str) -> str:
    """Trả về codec audio của stream đầu tiên (vd. 'aac'), '' nếu không có/không dò được.

    Dùng ffprobe (đi kèm ffmpeg). Lỗi/thiếu ffprobe → '' → nhánh re-encode an toàn.
    """
    if shutil.which("ffprobe") is None:
        return ""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "default=nw=1:nokey=1", src_video],
            capture_output=True, text=True, timeout=15,
        )
        return proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


class VideoExporter:
    def __init__(self, reader: VideoReader, engine: ReframeEngine, settings: ExportSettings):
        self.reader = reader
        self.engine = engine
        self.settings = settings

    def _video_encode_flags(self) -> list[str]:
        """Cờ encode video cho ffmpeg. -preset chỉ hợp lệ với x264/x265.

        Frame vào là bgr24 thô (không mang metadata màu). Gắn signalling bt709 +
        limited range để player hiển thị đúng ma trận màu HD (tránh lệch màu do
        rơi về bt601 mặc định); yuv420p là chuẩn tương thích rộng nhất.
        """
        flags = ["-c:v", self.settings.codec]
        if self.settings.codec.startswith("libx26"):
            flags += ["-preset", self.settings.preset]
        flags += [
            "-crf", str(self.settings.crf),
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
        ]
        return flags

    @staticmethod
    def _audio_flags_for_codec(codec: str) -> list[str]:
        """Chọn cách xử lý audio theo codec NGUỒN.

        Nếu codec nguồn mux thẳng vào MP4 được (aac/mp3/ac3/eac3/alac) → `-c:a copy`
        (lossless + nhanh, tránh re-encode thế hệ 2). Ngược lại (hoặc không rõ) →
        re-encode AAC 192k (trong suốt với thoại, gần trong suốt với nhạc).
        """
        if codec.lower() in ("aac", "mp3", "ac3", "eac3", "alac"):
            return ["-c:a", "copy"]
        return ["-c:a", "aac", "-b:a", "192k"]

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
        """Sinh từng frame đầu ra đã cắt/ghép; hủy giữa chừng qua cancel_cb."""
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
        """ffmpeg đọc frame BGR thô từ stdin, ghép audio nguồn, encode 1 lần sang H.264.

        Audio: copy lossless nếu codec nguồn muxable, ngược lại AAC 192k. `+faststart`
        đưa moov atom lên đầu để phát/stream (upload TikTok) mượt.
        """
        return [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{tw}x{th}", "-r", str(fps),
            "-i", "-",
            "-i", src_video,
            "-map", "0:v:0", "-map", "1:a:0?",
            *self._video_encode_flags(),
            *self._audio_flags_for_codec(_probe_audio_codec(src_video)),
            "-shortest",
            "-movflags", "+faststart",
            self.settings.out_path,
        ]

    def _run_ffmpeg_pipe(self, frames: Iterator[np.ndarray], tw: int, th: int,
                         fps: float, src_video: str) -> str:
        """Pipe frame thô thẳng vào MỘT tiến trình ffmpeg — không encode trung gian."""
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
                # Ghi thẳng buffer (memoryview) — tránh copy 1 bản bytes/frame của
                # .tobytes(); cv2.resize đã trả mảng C-contiguous nên .data là raw BGR.
                proc.stdin.write(np.ascontiguousarray(out_frame).data)
                n += 1
            proc.stdin.close()
        except InterruptedError:
            # Người dùng hủy → giết ffmpeg, dọn thread + file dở.
            _abort_ffmpeg(proc, drainer, out_path)
            raise
        except Exception as exc:  # noqa: BLE001
            # Mọi lỗi khác khi sinh/ghi frame: cv2.error frame hỏng, engine lỗi, hoặc
            # broken pipe (BrokenPipeError HAY bare OSError/WinError trên Windows khi
            # ffmpeg thoát sớm). PHẢI dọn sạch để không rò tiến trình/thread/file.
            _abort_ffmpeg(proc, drainer, out_path)
            tail = b"".join(stderr_tail).decode("utf-8", "replace")[-1500:]
            if isinstance(exc, OSError):  # broken pipe: ffmpeg thoát sớm
                log.error("ffmpeg đóng pipe sớm:\n%s", tail)
                raise RuntimeError("ffmpeg đóng pipe sớm — xem logs/khunghinh.log.") from exc
            log.error("Lỗi khi xuất qua ffmpeg pipe:\n%s", tail)
            raise
        ret = proc.wait()
        drainer.join(timeout=2.0)
        if ret != 0:
            log.error("ffmpeg lỗi:\n%s",
                      b"".join(stderr_tail).decode("utf-8", "replace")[-1500:])
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


def _abort_ffmpeg(proc: "subprocess.Popen", drainer: threading.Thread, out_path: str) -> None:
    """Dọn sạch khi xuất qua pipe thất bại/hủy: đóng stdin, giết + reap ffmpeg,
    join thread đọc stderr, xoá file đầu ra dở dang. Nuốt mọi lỗi phụ khi dọn."""
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass
    drainer.join(timeout=2.0)
    _safe_remove(out_path)


def _safe_remove(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
