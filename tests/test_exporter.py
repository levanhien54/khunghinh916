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


def test_encode_flags_include_bt709_color_signalling():
    flags = _exporter(codec="libx264")._video_encode_flags()
    assert flags[flags.index("-colorspace") + 1] == "bt709"
    assert flags[flags.index("-color_range") + 1] == "tv"


def test_build_pipe_cmd_has_faststart():
    cmd = _exporter()._build_pipe_cmd(1080, 1920, 30.0, "in.mp4")
    assert "-movflags" in cmd and cmd[cmd.index("-movflags") + 1] == "+faststart"


def test_audio_flags_copy_for_aac_source():
    # Nguồn đã là AAC (muxable vào MP4) → copy lossless, không re-encode.
    assert _exporter()._audio_flags_for_codec("aac") == ["-c:a", "copy"]


def test_audio_flags_reencode_for_incompatible_source():
    # Nguồn không muxable trực tiếp (vd. vorbis/opus-in-mp4 rủi ro) → re-encode AAC 192k.
    flags = _exporter()._audio_flags_for_codec("vorbis")
    assert flags[:2] == ["-c:a", "aac"]
    assert "-b:a" in flags and flags[flags.index("-b:a") + 1] == "192k"


def test_audio_flags_reencode_when_codec_unknown():
    flags = _exporter()._audio_flags_for_codec("")
    assert flags[:2] == ["-c:a", "aac"]


def test_export_settings_has_fg_scale_default():
    from khunghinh.mediaio.exporter import ExportSettings
    s = ExportSettings(out_path="out.mp4")
    assert s.fg_scale == 1.0
