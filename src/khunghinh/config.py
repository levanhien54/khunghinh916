"""Cấu hình ứng dụng — dataclass nạp/lưu JSON để dễ tùy chỉnh, không sửa code."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class AppConfig:
    # --- Khung hình đích (TikTok dọc) ---
    target_width: int = 1080
    target_height: int = 1920

    # --- Giới hạn slider zoom (1.0 = vừa khít base crop; >1 = cắt sát/phóng to) ---
    zoom_min: float = 1.0
    zoom_max: float = 3.0
    zoom_default: float = 1.0

    # --- Làm mượt camera ảo (One Euro filter) ---
    smoothing_min_cutoff: float = 1.0
    smoothing_beta: float = 0.05

    # --- Xuất video ---
    export_crf: int = 18          # H.264: thấp hơn = nét hơn (18 ~ "thị giác không mất mát")
    export_codec: str = "libx264"
    export_preset: str = "veryfast"   # x264/x265: nhanh hơn 'medium' 3-5x, size to hơn chút

    # --- Nền mờ (compose thủ công: video A trên nền mờ) ---
    blurred_background: bool = False
    bg_blur_downscale_divisor: int = 32
    bg_blur_dim: float = 0.55
    fg_scale_default: float = 1.0   # 1.0 = vừa khít (contain); >1 phóng to; <1 thu nhỏ
    fg_scale_min: float = 0.3
    fg_scale_max: float = 3.0

    # --- Model (để trống = tự dùng YuNet bundle ở models/, ngược lại fallback Haar) ---
    yunet_model_path: str = ""
    yunet_input_width: int = 320   # YuNet thu nhỏ về cỡ này trước suy luận (320 ~4x nhanh, vẫn nét)

    # --- Phát hiện khuôn mặt ---
    face_detect_width: int = 640        # downscale trước khi detect (tốc độ/độ chính xác)
    detect_stride: int = 1              # detect mỗi N frame (1=mỗi frame); >1 = nhanh hơn, coast track giữa các frame
    face_min_size_frac: float = 0.06    # mặt nhỏ nhất = 6% cạnh ngắn khung
    face_max_size_frac: float = 0.9
    face_score_threshold: float = 0.6   # dùng chung cho YuNet + ngưỡng "high" của tracker
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5

    # --- Phát hiện người nói (ASD heuristic) ---
    asd_vad_threshold: float = 0.5      # NGƯỠNG TRÊN ĐIỂM VAD MỀM [0,1] (không phải RMS thô)
    asd_min_dwell_frames: int = 8
    asd_switch_confirm_frames: int = 4
    asd_hysteresis_margin: float = 1.25

    # --- Quỹ đạo camera (Pass 2) ---
    path_min_cutoff: float = 0.6
    path_beta: float = 0.04
    path_deadzone_frac_x: float = 0.06
    path_deadzone_frac_y: float = 0.08
    path_settle_frames: int = 9
    # Recenter TẮT mặc định (0): đo cho thấy nó tạo răng cưa ~dz mỗi N frame khi pan
    # (giật), trong khi lỗi nó sửa (lệch tâm ~6%) là nhỏ. Bật (>0) nếu chấp nhận giật.
    path_recenter_frames: int = 0

    # --- Phát hiện cắt cảnh ---
    cut_threshold: float = 0.35
    cut_min_scene_len: int = 12

    # --- Phân tích ---
    analysis_downscale_width: int = 320
    analysis_detect_workers: int = 0   # số luồng detect song song (0 = auto = số core-1, tối đa 16)

    @property
    def target_aspect(self) -> float:
        return self.target_width / self.target_height

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            log.info("Đã nạp cấu hình từ %s", path)
            return cls(**known)
        except FileNotFoundError:
            log.info("Không có %s — dùng cấu hình mặc định.", path.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Lỗi đọc cấu hình %s (%s) — dùng mặc định.", path, exc)
        return cls()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )
