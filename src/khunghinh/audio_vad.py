"""Voice Activity Detection theo từng frame video — nguồn VAD duy nhất cho pipeline auto.

Trích audio bằng ffmpeg → 16kHz mono → đọc bằng `wave` + numpy (KHÔNG cần librosa),
tính điểm "có người nói" MỀM [0,1] cho mỗi frame video, căn theo timestamp.
Không có audio / không có ffmpeg → trả về toàn 1.0 (vô hiệu hoá yếu tố "khi nào",
để yếu tố "ai" theo chuyển động môi tự quyết).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass(frozen=True)
class AudioTrack:
    signal: np.ndarray  # float32 mono [-1, 1]
    sample_rate: int

    @property
    def duration_sec(self) -> float:
        return len(self.signal) / self.sample_rate if self.sample_rate else 0.0


@dataclass
class VadParams:
    win_ms: float = 25.0
    hop_ms: float = 10.0
    noise_percentile: float = 20.0
    margin_db: float = 6.0
    floor_min_db: float = -60.0
    soft_db_range: float = 10.0
    use_zcr: bool = True
    zcr_max: float = 0.35
    vad_on: float = 0.5   # Schmitt: bật khi điểm mềm vượt
    vad_off: float = 0.35  # Schmitt: tắt khi điểm mềm tụt dưới


def _read_wav(path: Path) -> AudioTrack:
    with wave.open(str(path), "rb") as wf:
        n, sr, sw, ch = wf.getnframes(), wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:  # pragma: no cover
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return AudioTrack(signal=np.ascontiguousarray(data, dtype=np.float32), sample_rate=sr)


def extract_audio(
    video_path: str, work_dir: str | None = None, target_sr: int = 16000, timeout_sec: float = 120.0
) -> AudioTrack | None:
    """Trích audio → AudioTrack, hoặc None nếu không có ffmpeg / không có audio stream / quá hạn."""
    if not ffmpeg_available():
        log.info("Không có ffmpeg — bỏ qua VAD audio.")
        return None
    tmp_made = work_dir is None
    wd = Path(work_dir or tempfile.mkdtemp(prefix="kh_audio_"))
    wav = wd / "audio_16k_mono.wav"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(target_sr),
           "-acodec", "pcm_s16le", str(wav)]
    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg trích audio quá %.0fs — bỏ qua VAD audio cho video này.", timeout_sec)
            return None
        if proc.returncode != 0 or not wav.is_file():
            stderr = (proc.stderr or "").lower()
            if any(m in stderr for m in ("invalid data found", "moov atom not found", "error opening input")):
                raise RuntimeError("File video lỗi/hỏng — không trích được audio.")
            log.info("Không trích được audio (có thể video không có audio) → VAD = all active.")
            return None
        return _read_wav(wav)
    finally:
        try:
            if wav.is_file():
                wav.unlink()
            if tmp_made:
                shutil.rmtree(wd, ignore_errors=True)
        except Exception:  # noqa: BLE001 # pragma: no cover
            pass


def _schmitt_gate(soft: np.ndarray, on_t: float, off_t: float) -> np.ndarray:
    out = np.zeros_like(soft)
    state = False
    for i, v in enumerate(soft):
        if state:
            if v < off_t:
                state = False
        elif v > on_t:
            state = True
        out[i] = v if state else 0.0
    return out


class FrameVad:
    def __init__(self, params: VadParams | None = None):
        self.p = params or VadParams()

    def compute(self, signal: np.ndarray, sample_rate: int, fps: float, frame_count: int) -> np.ndarray:
        if fps <= 0 or frame_count <= 0:
            raise ValueError("fps và frame_count phải > 0")
        p = self.p
        sig = np.asarray(signal, dtype=np.float64).ravel()
        n = sig.size
        if n == 0:
            return np.ones(frame_count, dtype=np.float32)

        win = max(1, int(sample_rate * p.win_ms / 1000.0))
        hop = max(1, int(sample_rate * p.hop_ms / 1000.0))
        n_hops = 1 + max(0, (n - win)) // hop if n >= win else 1

        sq = np.concatenate([[0.0], np.cumsum(sig * sig)])
        starts = np.arange(n_hops) * hop
        ends = np.minimum(starts + win, n)
        counts = np.maximum(1, ends - starts)
        energy = (sq[ends] - sq[starts]) / counts
        rms = np.sqrt(np.maximum(energy, 1e-12))
        db = 20.0 * np.log10(np.maximum(rms, 1e-7))

        floor = max(float(np.percentile(db, p.noise_percentile)), p.floor_min_db)
        soft = np.clip((db - (floor + p.margin_db)) / max(1e-6, p.soft_db_range), 0.0, 1.0)

        if p.use_zcr:
            sign = np.sign(sig)
            sign[sign == 0] = 1.0
            changes = (np.abs(np.diff(sign)) > 0).astype(np.float64)
            csum = np.concatenate([[0.0], np.cumsum(changes)])
            zc_ends = np.minimum(ends, csum.size - 1)
            zcr = (csum[zc_ends] - csum[np.minimum(starts, csum.size - 1)]) / counts
            soft = np.where(zcr > p.zcr_max, soft * 0.5, soft)

        soft = _schmitt_gate(soft, p.vad_on, p.vad_off)

        # Căn tâm-hop sang tâm-frame video; forward-fill các frame cuối.
        hop_centers = (starts + win / 2.0) / sample_rate
        frame_times = (np.arange(frame_count) + 0.5) / fps
        idx = np.searchsorted(hop_centers, frame_times)
        idx = np.clip(idx, 0, soft.size - 1)
        return soft[idx].astype(np.float32)

    def compute_track(self, track: AudioTrack, fps: float, frame_count: int) -> np.ndarray:
        return self.compute(track.signal, track.sample_rate, fps, frame_count)


def frame_vad_for_video(
    video_path: str,
    fps: float,
    frame_count: int,
    params: VadParams | None = None,
    work_dir: str | None = None,
) -> np.ndarray:
    """VAD mềm [0,1] cho mỗi frame. Không audio/không ffmpeg → np.ones(frame_count)."""
    track = extract_audio(video_path, work_dir=work_dir)
    if track is None:
        return np.ones(frame_count, dtype=np.float32)
    return FrameVad(params).compute_track(track, fps, frame_count)
