# KhungHinh916

Ứng dụng desktop **PyQt6** tự cắt khung video sang **tỉ lệ dọc 9:16 (TikTok/Reels/Shorts)**,
chạy hoàn toàn trên **CPU** (không cần GPU NVIDIA). Hỗ trợ cả **reframe thủ công**
(chọn tâm + zoom theo trục X/Y) và **reframe tự động bám người nói** (active speaker
detection trên CPU, camera ảo di chuyển mượt bằng One Euro filter), cùng tuỳ chọn
**nền mờ** bật kèm — lớp nền phủ kín 9:16 vẽ phía dưới khung hình đã cắt.

---

## Tính năng

- 📂 Nhập video, **tự nhận diện độ phân giải / tỉ lệ / fps / thời lượng**.
- 🖼️ Xem trước bằng **QGraphicsView** với khung cắt 9:16 (tối hoá vùng ngoài + đường 1/3)
  và thanh tua frame.
- 🎯 **Chế độ Thủ công** — kéo khung để chọn tâm; **slider Zoom X/Y độc lập** (khoá X=Y
  chống méo).
- 🤖 **Chế độ Tự động (bám người nói)** — nút "Phân tích video" chạy ở luồng nền:
  - Phát hiện khuôn mặt CPU (**YuNet** nếu có model `.onnx`, tự **fallback Haar cascade**
    có sẵn trong OpenCV — **zero-download**, luôn chạy được).
  - Bám đa khuôn mặt qua các frame (**IouTracker**, kiểu ByteTrack-lite, 2 tầng high/low score).
  - Phát hiện người đang nói = **VAD âm thanh** (qua ffmpeg, không cần model) **×
    chuyển động môi** mỗi khuôn mặt, có hysteresis/dwell chống nhấp nháy khi đổi người nói.
  - Quỹ đạo camera 2 lượt: phát hiện **cắt cảnh** để *snap* (không lia ngang qua cú cắt),
    **dead-zone + One Euro filter** cho chuyển động mượt như tripod.
  - Overlay khuôn mặt + người đang nói ngay trên preview, tua được để xem trước kết quả.
- 🌫️ **Nền mờ (compose kiểu CapCut)** — đặt video A (nguyên khung) lên nền mờ của
  chính nó: cỡ chỉnh tay (kéo góc trong ô xem trước / slider "Cỡ video A"), vị trí
  tự bám người nói (pan ngang giữ người trong khung, cắt 2 rìa, letterbox mờ trên/dưới).
- 💾 **Xuất MP4 1080×1920** (H.264) kèm **giữ nguyên audio gốc** — pipe frame thô **thẳng
  vào ffmpeg** (encode 1 lần, không file tạm), tốc độ chỉnh qua `export_preset` (mặc định `veryfast`).
- 🧱 Kiến trúc phân lớp, **115 unit test** cho phần logic, **logging** ra file để debug.

## Giới hạn giấy phép (đáng tin cậy cho thương mại hóa)

Toàn bộ stack mặc định (**YuNet/Haar + IouTracker + VAD heuristic + One Euro +
OpenCV/ffmpeg**) đều **MIT/Apache** — an toàn dùng closed-source. Không phụ thuộc
Ultralytics YOLO (AGPL-3.0). Chỗ cắm sẵn cho model ASD học sâu mạnh hơn (LR-ASD/TalkNet,
cũng MIT, chạy CPU) qua interface `ActiveSpeakerDetector`.

---

## Cài đặt

```bash
# (khuyến nghị) tạo virtualenv
python -m venv .venv && .venv\Scripts\activate     # Windows

pip install -r requirements.txt
# cần ffmpeg trên PATH để ghép audio + encode H.264:  https://ffmpeg.org/download.html
```

> ⚠️ **Chỉ giữ một bản OpenCV.** Nếu đã cài cả `opencv-python` lẫn `opencv-contrib-python`
> có thể xung đột. Gỡ bản thường, giữ bản contrib (đã gồm `FaceDetectorYN`/YuNet):
> `pip uninstall opencv-python` rồi `pip install --force-reinstall opencv-contrib-python`.

## Chạy

```bash
python run.py
```

## Kiểm thử

```bash
python -m pytest          # 115 unit test: geometry, smoothing, tracking, ASD, camera path,
                           # audio VAD, face detection, compositing, analysis result...
```

---

## Cấu trúc dự án

```
KhungHinh916/
├── run.py                      # điểm khởi chạy (python run.py)
├── pyproject.toml              # metadata + cấu hình pytest (pythonpath=src)
├── requirements.txt
├── config.example.json         # sao chép thành config.json để tùy chỉnh
├── models/                     # model ONNX (YuNet…) — tải riêng, xem models/README.md
├── src/khunghinh/
│   ├── app.py                  # bootstrap QApplication
│   ├── config.py               # AppConfig (nạp/lưu JSON, ~30 tham số tinh chỉnh)
│   ├── logging_setup.py        # logging console + file xoay vòng
│   ├── audio_vad.py            # trích audio (ffmpeg) + Voice Activity Detection mềm [0,1]
│   ├── core/                   # LÕI THUẦN (không phụ thuộc Qt) — dễ test
│   │   ├── geometry.py         #   toán cắt khung + zoom theo trục
│   │   ├── smoothing.py        #   One Euro filter / EMA
│   │   ├── reframe_engine.py   #   ghép geometry + smoothing
│   │   ├── camera_path.py      #   Pass2: cắt cảnh + dead-zone + One Euro -> quỹ đạo camera
│   │   ├── scene_features.py   #   đặc trưng cảnh (luma + HSV histogram) cho cắt cảnh
│   │   ├── analysis_result.py  #   kết quả phân tích auto (dataclass thuần, validate)
│   │   └── compositing.py      #   nền mờ (blur rẻ trên CPU) + ghép khung hình đã cắt lên trên
│   ├── mediaio/
│   │   ├── reader.py           # đọc video + metadata (OpenCV)
│   │   └── exporter.py         # cắt/ghép nền mờ + ghi video + ghép audio (ffmpeg)
│   ├── detection/
│   │   ├── base.py             #   Protocol: FaceDetector, ActiveSpeakerDetector
│   │   ├── yunet_face.py       #   YuNet (MIT, cần tải .onnx — tùy chọn)
│   │   ├── haar_face.py        #   Haar cascade có sẵn OpenCV — zero-download, + mouth_roi()
│   │   ├── factory.py          #   chọn YuNet nếu có model, ngược lại Haar
│   │   └── asd_fusion.py       #   VAD × chuyển động môi -> chọn người đang nói (hysteresis)
│   ├── tracking/
│   │   └── iou_tracker.py      # IouTracker (ByteTrack-lite, 2 tầng high/low score)
│   └── ui/
│       ├── main_window.py      # kết nối toàn bộ, vòng đời worker nền an toàn
│       ├── preview_view.py     # QGraphicsView + overlay khung cắt/khuôn mặt
│       ├── control_panel.py    # nút nhập/xuất, chế độ, zoom X/Y, nền mờ
│       ├── analysis_worker.py  # QThread: detect -> track -> ASD -> quỹ đạo camera
│       └── workers.py          # QThread xuất video (không treo GUI)
└── tests/                      # 115 test cho toàn bộ core/ + detection/ + tracking/
```

### Nguyên tắc thiết kế (để dễ debug / nâng cấp / tối ưu)

- **Tách lõi thuần khỏi GUI:** mọi toán học (crop, smoothing, camera path, compositing)
  nằm trong `core/`, không import Qt → kiểm thử nhanh, tái dùng được cho CLI/batch.
- **Plug point rõ ràng cho model sâu hơn:** `ActiveSpeakerSelector(detector=...,
  audio_provider=...)` nhận bất kỳ `ActiveSpeakerDetector` nào (vd. LR-ASD/TalkNet) thay
  cho heuristic mặc định, không cần sửa camera path hay exporter.
- **Xử lý nặng ở luồng nền** (`QThread`) với tiến độ + hủy, mỗi luồng có `VideoReader`
  riêng (cv2.VideoCapture không an toàn đa luồng) → GUI luôn mượt, không đụng widget
  từ luồng nền (chỉ giao tiếp qua `pyqtSignal`).
- **Vòng đời worker an toàn:** hủy + chờ có timeout khi đổi video/đóng app; nếu worker
  không dừng kịp thì ngắt kết nối signal thay vì để nó bắn vào widget đã đóng.
- **Cache có xác thực:** kết quả phân tích auto được gắn `params_fingerprint`, chỉ chấp
  nhận nếu khớp đúng video/tỉ lệ hiện tại — tránh kết quả "trễ" của video trước ghi đè.
- **Logging tập trung**: `logs/khunghinh.log` (DEBUG) để truy vết lỗi.
