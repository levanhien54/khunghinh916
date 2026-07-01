from __future__ import annotations

import numpy as np
import pytest

import subprocess

from khunghinh import audio_vad
from khunghinh.audio_vad import FrameVad, VadParams, extract_audio, frame_vad_for_video


def test_zeros_all_low():
    out = FrameVad().compute(np.zeros(16000 * 2, np.float32), 16000, 30, 60)
    assert out.shape == (60,)
    assert out.dtype == np.float32
    assert np.all(out < 0.5)


def test_tone_then_silence():
    sr = 16000
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    sig = np.concatenate([tone, np.zeros(sr, np.float32)])
    out = FrameVad().compute(sig, sr, 30, 60)
    assert out[:30].mean() > 0.6
    assert out[30:].mean() < 0.2


def test_shape_for_various_fps():
    sig = (0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    for fps in (24, 25, 30, 60):
        out = FrameVad().compute(sig, 16000, fps, fps)
        assert out.shape == (fps,) and out.dtype == np.float32


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        FrameVad().compute(np.zeros(100, np.float32), 16000, 0, 10)
    with pytest.raises(ValueError):
        FrameVad().compute(np.zeros(100, np.float32), 16000, 30, 0)


def test_no_audio_returns_ones(monkeypatch):
    monkeypatch.setattr(audio_vad, "extract_audio", lambda *a, **k: None)
    out = frame_vad_for_video("nope.mp4", 30, 45)
    assert out.shape == (45,)
    assert np.allclose(out, 1.0)


def test_extract_audio_ffmpeg_timeout_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_vad, "ffmpeg_available", lambda: True)

    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(audio_vad.subprocess, "run", fake_run)
    result = extract_audio("video.mp4", work_dir=str(tmp_path), timeout_sec=1.0)
    assert result is None


def test_zcr_damping_reduces_noise_vad():
    sr = 16000
    rng = np.random.default_rng(0)
    noise = (0.4 * rng.standard_normal(sr)).astype(np.float32)
    sig = np.concatenate([noise, np.zeros(sr, np.float32)])  # silence để có floor
    with_zcr = FrameVad(VadParams(use_zcr=True)).compute(sig, sr, 30, 60)[:30].mean()
    without_zcr = FrameVad(VadParams(use_zcr=False)).compute(sig, sr, 30, 60)[:30].mean()
    assert with_zcr <= without_zcr
