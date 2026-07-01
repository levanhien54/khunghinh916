# Thiết kế: Hợp nhất "Nền mờ" với Reframe (Thủ công/Tự động)

Ngày: 2026-07-01

## Bối cảnh

Hiện tại "Nền mờ" và reframe (Thủ công/Tự động) là **hai chế độ loại trừ nhau**:
bật checkbox "Nền mờ" sẽ vô hiệu hóa hoàn toàn zoom/bám người nói, chỉ hiển thị
nguyên khung hình gốc (không cắt) đặt "vừa trọn" (contain) lên nền mờ phủ kín.

Yêu cầu mới: video A phải luôn được xử lý theo pipeline reframe (bám người nói
tự động hoặc tâm/zoom thủ công) để CẮT khung hình theo chuyển động, và tuỳ chọn
"Nền mờ" chỉ là một lớp nền an toàn được vẽ *trước*, phía dưới khung hình đã cắt.

## Quyết định thiết kế đã chốt (qua trao đổi với người dùng)

1. **Foreground fit**: foreground luôn là crop cắt theo `target_aspect` 9:16
   (giống hệt logic `crop_fill`/`compute_crop_rect` đang dùng cho Thủ công/Tự
   động), resize phủ kín toàn bộ canvas 1080×1920 — không dùng kiểu "vừa trọn,
   giữ nguyên tỉ lệ gốc, chừa viền" như trước.
2. **Vai trò nền mờ**: nền mờ luôn được vẽ trước (từ frame gốc, đầy đủ, qua
   `make_blurred_background`), sau đó khung hình đã crop phủ kín đè lên trên
   toàn bộ canvas. **Hệ quả đã được xác nhận rõ**: vì crop luôn resize đúng
   bằng kích thước canvas đích, nền mờ trong thực tế **hầu như không bao giờ
   hiển thị** được trong video xuất — nó tồn tại như một lớp nền dự phòng/an
   toàn kiến trúc, không phải để tạo viền mờ nhìn thấy được. Đây là đánh đổi
   được người dùng xác nhận rõ ràng (không phải lỗi thiết kế bị bỏ sót).
3. **Chế độ cũ (`blur_fit`: full-frame không crop) bị loại bỏ hoàn toàn**,
   thay thế bằng luồng mới. Không giữ lại như một lựa chọn riêng.

## Thay đổi theo file

### `src/khunghinh/core/compositing.py`

- Giữ nguyên `fit_dimensions` và `make_blurred_background` (không đổi, vẫn có
  test cũ dùng tới `fit_dimensions`; `make_blurred_background` vẫn là hàm lõi
  tạo nền mờ).
- Xoá `composite_fit_on_blurred_background` (foreground = full frame, fit
  contain) — không còn dùng.
- Thêm hàm dùng chung `crop_and_resize(frame, rect: CropRect, target_w,
  target_h) -> np.ndarray` (chuyển từ `_crop_and_resize` riêng trong
  `mediaio/exporter.py` lên đây làm hàm public dùng chung giữa export thường
  và export có nền mờ).
- Thêm `composite_crop_on_blurred_background(frame, rect: CropRect, target_w,
  target_h, downscale_divisor=32, dim=0.55) -> np.ndarray`:
  - `bg = make_blurred_background(frame, target_w, target_h, downscale_divisor, dim)`
  - `fg = crop_and_resize(frame, rect, target_w, target_h)`
  - Ghi `fg` đè lên `bg` toàn bộ canvas, trả về kết quả (về mặt pixel tương
    đương `fg`, nhưng vẫn "vẽ" `bg` trước theo đúng yêu cầu kiến trúc).
  - Thêm comment ngắn giải thích rõ vì sao nền mờ bị đè kín 100% (tránh gây
    khó hiểu cho người đọc code sau này).

### `src/khunghinh/config.py`

- Thay `background_mode: str = "crop_fill"` bằng `blurred_background: bool =
  False`. Giữ nguyên `bg_blur_downscale_divisor`, `bg_blur_dim`.

### `src/khunghinh/mediaio/exporter.py`

- `ExportSettings`: thay `background_mode: str` bằng `blurred_background: bool
  = False`.
- `VideoExporter.run()`: luôn tính `rect` qua `center_provider`/`engine` (như
  nhánh không-mờ hiện tại) bất kể `blurred_background`. Chỉ rẽ nhánh ở bước
  ghép frame cuối:
  - `blurred_background=True` → `composite_crop_on_blurred_background(frame, rect, tw, th, ...)`
  - `blurred_background=False` → `crop_and_resize(frame, rect, tw, th)` (import
    từ `core/compositing.py` thay vì hàm nội bộ `_crop_and_resize`).
- Xoá nhánh đặc biệt `blur_fit` bỏ qua `center_provider`.

### `src/khunghinh/ui/control_panel.py`

- `_on_blur_bg_toggle`: bỏ vòng lặp disable `rad_manual`, `rad_auto`,
  `sld_zoom_x`, `sld_zoom_y`, `chk_link`, `btn_reset`. Nền mờ giờ độc lập,
  không còn tắt reframe.
- `_refresh_analyze_enabled`: bỏ điều kiện `and not self.chk_blur_bg.isChecked()`.
- Cập nhật text hint của `bg_hint` (không còn "khi bật, khung cắt thủ công bị
  tắt").

### `src/khunghinh/ui/main_window.py`

- `on_export`: bỏ nhánh đặc biệt `if self._blur_bg: center_provider, smooth =
  None, True`. `center_provider`/`smooth` tính giống hệt nhau bất kể
  `_blur_bg`. Bỏ điều kiện `not self._blur_bg` trong cảnh báo "Cần phân tích"
  trước khi xuất (giờ áp dụng cho mọi trường hợp mode="auto" chưa phân tích,
  kể cả khi bật nền mờ). `ExportSettings(..., blurred_background=self._blur_bg, ...)`
  thay cho `background_mode=...`.
- `_redraw_for_frame`: bỏ nhánh đặc biệt render composite full-frame khi
  `self._blur_bg`. Preview luôn hiển thị overlay khung cắt (rect) giống Thủ
  công/Tự động, bất kể nền mờ bật/tắt — nền mờ chỉ ảnh hưởng tới **xuất
  video**, không ảnh hưởng preview tương tác (tránh tốn CPU dựng composite khi
  tua frame, và nhất quán với việc nền mờ không hiển thị trong kết quả cuối).
- `on_background_mode_changed`: chỉ còn lưu cờ `self._blur_bg` + cập nhật
  status bar text; bỏ `self.preview.set_overlay_enabled(not is_blur)` và
  `_redraw_for_frame` composite đặc biệt.

### `config.example.json`

- Nếu có khoá `background_mode`, đổi thành `blurred_background` (kiểu bool).

## Testing

- `tests/test_compositing.py`: xoá 3 test dựa trên
  `composite_fit_on_blurred_background`
  (`test_composite_matching_aspect_fills_without_visible_border`,
  `test_composite_mismatched_aspect_shows_blurred_border`,
  `test_composite_output_dtype_and_no_crash_on_odd_sizes`). Thêm test cho
  `composite_crop_on_blurred_background`:
  - shape/dtype đúng target.
  - Với `rect` phủ toàn bộ frame lẫn `rect` là vùng zoom/lệch tâm, output
    bằng **chính xác** `crop_and_resize(frame, rect, tw, th)` (khẳng định rõ
    ràng: nền mờ không còn ảnh hưởng pixel cuối — đây là hành vi được xác
    nhận, không phải bug).
  - Thêm test cho `crop_and_resize` (chuyển từ test nội bộ cũ của exporter
    nếu có, hoặc viết mới) để đảm bảo hàm dùng chung hoạt động đúng khi tách
    ra `compositing.py`.
- Rà lại các test hiện có của `exporter.py`/`control_panel`/`main_window`
  (nếu có) tham chiếu `background_mode`/`blur_fit` — cập nhật theo tên field
  mới `blurred_background`.
- Chạy lại toàn bộ `pytest` (86 test hiện có + test mới/sửa) để đảm bảo không
  hồi quy.

## Ngoài phạm vi

- Không thay đổi thuật toán `make_blurred_background` (vẫn thu nhỏ ~1/32 rồi
  phóng to, giữ nguyên tham số `downscale_divisor`, `dim`).
- Không thêm chế độ nào khác ngoài việc hợp nhất nền mờ + reframe như mô tả;
  không giữ lại chế độ "nền mờ giữ nguyên khung hình" cũ dưới bất kỳ hình
  thức nào (đã được xác nhận loại bỏ hoàn toàn).
