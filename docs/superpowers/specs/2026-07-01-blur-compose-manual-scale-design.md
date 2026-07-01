# Thiết kế: Chế độ Nền mờ = Compose thủ công (cỡ tay + bám người) — kiểu CapCut

Ngày: 2026-07-01

## Bối cảnh

Hiện tại (sau lần hợp nhất trước), khi bật **Nền mờ**, foreground là crop 9:16
**lấp kín** canvas → nền mờ không bao giờ hiện ra (`composite_crop_on_blurred_background`
short-circuit về `crop_and_resize`).

Yêu cầu mới (người dùng xác nhận qua ảnh + minh hoạ): chế độ Nền mờ phải giống
CapCut — đặt **video A (nguyên khung, giữ tỉ lệ)** lên nền mờ, **cỡ chỉnh tay**
(phóng to/thu nhỏ), và **vị trí tự bám người nói** để người luôn trong khung.

## Mô hình (đã duyệt)

Canvas 9:16 (1080×1920) cố định, gồm 2 lớp:

1. **Nền**: chính video A làm mờ (`make_blurred_background`), phủ kín canvas.
2. **Foreground**: video A nguyên khung, giữ tỉ lệ, đặt lên trên với:
   - **Cỡ = thủ công** (`fg_scale`). Baseline `fg_scale=1.0` = **contain-fit** (vừa
     khít khung, với nguồn landscape → khít chiều rộng, lộ viền mờ trên/dưới — đúng
     ảnh người dùng gửi). `>1` phóng to (tràn khung → cắt), `<1` thu nhỏ (lộ viền
     mờ nhiều hơn mọi phía).
   - **Vị trí = tự động bám người**: đặt sao cho tâm người (từ phân tích auto, hoặc
     tâm thủ công) rơi vào giữa canvas, rồi **kẹp theo từng trục**:
     - Trục mà foreground **≥ canvas** → kẹp để phủ kín trục đó (không lộ viền), tâm
       **bám người** (pan theo người).
     - Trục mà foreground **< canvas** → **căn giữa** trục đó (viền mờ đối xứng),
       không pan (toàn bộ trục đã hiển thị).
   - Phần foreground tràn ra ngoài canvas bị **cắt bỏ** (clip).

Hệ quả: `fg_scale=1.0` với nguồn landscape = ảnh tĩnh contain (không pan vì chiều
rộng vừa khít, chiều cao nhỏ hơn → căn giữa). Càng phóng to, pan-bám-người càng
kích hoạt (trục vượt canvas). Bám người dùng CHUNG nguồn tâm với chế độ cắt: auto
đã phân tích → tâm người nói đã làm mượt (camera path); manual → tâm khung/tâm kéo.

Blur mode **trực giao** với Thủ công/Tự động: bật Nền mờ chỉ đổi CÁCH ghép
(compose-on-blur thay vì crop-lấp-kín); nguồn tâm vẫn theo mode đang chọn.

**KHÔNG làm trong v1** (ghi rõ để tránh phình scope): xoay (icon tròn = RESET, không
phải xoay); kéo-tay-di-chuyển vị trí (vị trí do bám-người quyết định); hoạt ảnh
scale theo thời gian (fg_scale tĩnh cho cả clip).

## Thành phần & thay đổi theo file

### `core/geometry.py` (hoặc `core/compositing.py`) — hàm thuần đặt foreground

Thêm hàm thuần (dễ test, không cv2):

```
place_foreground(src_w, src_h, canvas_w, canvas_h, fg_scale,
                 person_cx_norm, person_cy_norm) -> ForegroundPlacement
```

- `ForegroundPlacement` (dataclass): `fg_w, fg_h` (int, cỡ foreground sau scale),
  `x, y` (int, toạ độ góc trên-trái của foreground trên canvas — CÓ THỂ âm).
- Tính: `contain = min(canvas_w/src_w, canvas_h/src_h)`; `fg_w = round(src_w*contain*fg_scale)`,
  `fg_h = round(src_h*contain*fg_scale)` (giữ tỉ lệ).
- Đặt tâm người vào giữa canvas: `x = round(canvas_w/2 - person_cx_norm*fg_w)`,
  `y = round(canvas_h/2 - person_cy_norm*fg_h)`.
- Kẹp theo trục:
  - Nếu `fg_w >= canvas_w`: `x = clamp(x, canvas_w - fg_w, 0)` (phủ kín, pan trong biên).
  - Ngược lại: `x = round((canvas_w - fg_w)/2)` (căn giữa).
  - Tương tự cho `y`/`fg_h`/`canvas_h`.
- `fg_scale`, kích thước phải > 0 (validate, raise ValueError khi ≤ 0).

### `core/compositing.py` — ghép compose lên nền mờ

Thêm:

```
composite_manual_on_blurred_background(frame, canvas_w, canvas_h, fg_scale,
    person_cx_norm, person_cy_norm, downscale_divisor=32, dim=0.55) -> np.ndarray
```

- `bg = make_blurred_background(frame, canvas_w, canvas_h, downscale_divisor, dim)`.
- `p = place_foreground(frame_w, frame_h, canvas_w, canvas_h, fg_scale, cx, cy)`.
- `resized = cv2.resize(frame, (p.fg_w, p.fg_h), INTER_CUBIC nếu phóng to else INTER_AREA)`
  (dùng lại quy tắc nội suy theo hướng đã có ở `crop_and_resize`).
- Tính vùng chồng lấn của foreground `[p.x, p.x+fg_w) × [p.y, p.y+fg_h)` với canvas
  `[0,canvas_w) × [0,canvas_h)`; dán sub-region tương ứng của `resized` vào `bg`.
  Nếu không chồng lấn (foreground ra ngoài hoàn toàn — không xảy ra với kẹp trên,
  nhưng phòng vệ) → trả `bg`.
- Trả canvas đã ghép.

`composite_crop_on_blurred_background` cũ: GIỮ (vẫn dùng nếu sau này cần crop-lấp-kín),
nhưng exporter/preview chế độ nền mờ chuyển sang dùng hàm compose mới.

### `mediaio/exporter.py`

- `ExportSettings`: thêm `fg_scale: float = 1.0`.
- `VideoExporter.run()`: nhánh `blurred_background`:
  - Lấy tâm người theo frame: nếu có `center_provider` → `(cx,cy)=center_provider(idx)`;
    ngược lại tâm khung `(src_w/2, src_h/2)`. (KHÔNG cần crop rect ở nhánh blur.)
  - `out = composite_manual_on_blurred_background(frame, tw, th, fg_scale,
    cx/src_w, cy/src_h, bg_blur_downscale_divisor, bg_blur_dim)`.
  - Nhánh không-mờ giữ nguyên (`crop_and_resize`).
- Lưu ý: ở nhánh blur, `center_provider` cấp tâm người ĐÃ làm mượt (auto: camera
  path Pass2; manual: tâm cố định). Không gọi `engine.crop_for_center` cho blur.

### `ui/main_window.py`

- State mới: `self._fg_scale: float = 1.0`.
- `on_export`: khi `_blur_bg`, truyền `fg_scale=self._fg_scale` vào `ExportSettings`,
  và center_provider: auto+đã phân tích → `analysis.make_center_provider()`; ngược
  lại (manual) → provider trả tâm thủ công `self._center_px` cố định (hoặc None →
  exporter tự dùng tâm khung). Bỏ nhánh đặc biệt cũ.
- `_redraw_for_frame`: khi `_blur_bg`, dựng canvas compose để xem trước (xem
  PreviewView) với tâm người của frame hiện tại (auto: `analysis.centers_for_frame`;
  manual: tâm khung/tâm kéo) + `_fg_scale`.
- Nối tín hiệu scale từ control panel/gizmo → cập nhật `_fg_scale` → vẽ lại.

### `ui/preview_view.py` — xem trước canvas compose + gizmo scale

Chế độ nền mờ hiển thị **canvas 9:16 đã ghép** (không phải frame gốc):

- Thêm đường vẽ `set_compose(canvas_bgr, fg_rect_on_canvas)`:
  - Đặt pixmap = canvas đã ghép (blur+fg) do `composite_manual_on_blurred_background`
    tạo (kích thước canvas_w×canvas_h).
  - Vẽ **gizmo scale**: hình chữ nhật quanh `fg_rect_on_canvas` (vùng foreground
    hiển thị) + 4 tay nắm góc + nút reset (icon tròn dưới).
- Tương tác: kéo tay nắm góc → đổi `fg_scale` (giữ tỉ lệ, scale quanh tâm người);
  phát tín hiệu `fgScaleChanged(scale)`. Nút reset → `fgScaleReset` (về 1.0).
- Overlay crop cũ (`_CropOverlay`) chỉ dùng cho chế độ không-mờ; gizmo compose là
  item riêng, bật khi nền mờ. Không trộn logic hai overlay.

### `ui/control_panel.py`

- Trong nhóm "Nền": khi bật Nền mờ, hiện **slider "Cỡ video A"** (`fg_scale`,
  vd. 0.3–3.0, mặc định 1.0) + nút "Đặt lại cỡ". Phát `fgScaleChanged` /
  `fgScaleReset`. Slider và gizmo cùng điều khiển một `_fg_scale` (đồng bộ 2 chiều).
- Ẩn/hiện slider theo trạng thái checkbox nền mờ.

### `config.py`

- `fg_scale_default: float = 1.0`, `fg_scale_min: float = 0.3`, `fg_scale_max: float = 3.0`.

## Kiểm thử

- `place_foreground` (thuần, nhiều case):
  - Nguồn landscape 1920×1080, canvas 1080×1920, scale=1 → fg 1080×607, x=0 (căn/phủ
    ngang), y=(1920-607)/2 (căn giữa dọc) — khớp ảnh.
  - scale=2 → fg lớn hơn canvas cả 2 trục → x,y kẹp trong biên, tâm người rơi giữa
    (pan). Kiểm tra person_cx_norm khác 0.5 → x lệch theo người nhưng vẫn kẹp.
  - scale < 1 → fg < canvas cả 2 trục → căn giữa cả 2 (không phụ thuộc tâm người).
  - person ở mép (cx_norm=0 hoặc 1) + fg>canvas → x kẹp biên (không lộ viền).
  - scale ≤ 0 → ValueError.
- `composite_manual_on_blurred_background`:
  - shape/dtype = (canvas_h, canvas_w, 3) uint8.
  - scale=1, nguồn landscape → hàng giữa (foreground) sáng hơn hàng biên trên (nền
    mờ tối) — foreground hiện đúng chỗ.
  - scale<1 → có viền mờ cả 4 phía (pixel góc canvas thuộc nền mờ, không phải fg).
  - Phòng vệ: frame kích thước lẻ, canvas nhỏ → không crash, đúng shape.
- Exporter/preview/control_panel/gizmo: verify bằng smoke-test headless + build lại
  exe (không unit-test lớp Qt/ffmpeg, theo tiền lệ dự án).

## Ngoài phạm vi

- Xoay foreground; kéo-tay-di-chuyển vị trí (vị trí do bám-người); scale động theo
  thời gian; nhiều lớp overlay. Có thể thêm sau nếu cần.
