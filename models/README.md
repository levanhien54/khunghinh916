# Models

Thư mục chứa model ONNX (không commit vào git — xem `.gitignore`). **Không bắt buộc** —
mặc định app dùng Haar cascade có sẵn trong OpenCV (zero-download) cho phát hiện khuôn mặt
và VAD âm thanh + chuyển động môi (heuristic, không cần model) cho phát hiện người nói.

## YuNet — phát hiện khuôn mặt (tùy chọn, nâng cao độ chính xác so với Haar mặc định)

- **License:** MIT (tự do dùng cả thương mại).
- **Tải:** https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Đặt file `face_detection_yunet_*.onnx` vào thư mục này, rồi trỏ đường dẫn trong
  `config.json` → `yunet_model_path`. Không đặt (hoặc đường dẫn không tồn tại) → app tự
  fallback sang `HaarFaceDetector` (`src/khunghinh/detection/haar_face.py`), luôn hoạt động.

```json
{ "yunet_model_path": "models/face_detection_yunet_2023mar.onnx" }
```

## (Nâng cấp tùy chọn) Active Speaker Detection bằng model học sâu

Mặc định, app chọn người đang nói bằng heuristic CPU rẻ: VAD âm thanh (ffmpeg) × chuyển
động vùng miệng mỗi khuôn mặt (`src/khunghinh/detection/asd_fusion.py`, không cần model).
Nếu muốn độ chính xác cao hơn, có thể cắm model học sâu nhẹ, vẫn chạy tốt trên CPU (đều MIT):

- **LR-ASD** — https://github.com/Junhua-Liao/LR-ASD (mới nhất, 0.84M params, 94.45% mAP)
- **Light-ASD** — https://github.com/Junhua-Liao/Light-ASD
- **TalkNet-ASD** — https://github.com/TaoRuijie/TalkNet-ASD (hỗ trợ CPU chính thức)

Cắm vào qua interface `ActiveSpeakerDetector` trong `src/khunghinh/detection/base.py`:
truyền instance vào `ActiveSpeakerSelector(detector=..., audio_provider=...)` — không cần
sửa `camera_path.py` hay `exporter.py`.
