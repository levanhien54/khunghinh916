"""Khung xem trước: QGraphicsView hiển thị frame video + overlay khung cắt 9:16.

- Frame hiển thị qua QGraphicsPixmapItem (chuyển BGR→RGB trước khi tạo QImage).
- Overlay làm tối vùng ngoài khung cắt + vẽ viền + đường 1/3, và cho phép kéo để
  di chuyển tâm camera. Khi kéo, view phát tín hiệu `cropCenterChanged(cx, cy)`
  (theo pixel ảnh gốc) để cửa sổ chính tính lại CropRect (đã kẹp + áp zoom).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from ..core.geometry import CropRect

log = logging.getLogger(__name__)


def ndarray_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Chuyển frame OpenCV (BGR) sang QPixmap (RGB)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class _CropOverlay(QGraphicsItem):
    """Item phủ toàn ảnh: tối hoá vùng ngoài khung cắt + vẽ khung + cho kéo di chuyển."""

    def __init__(self, view: "PreviewView"):
        super().__init__()
        self._view = view
        self._img_w = 1
        self._img_h = 1
        self._rect = QRectF(0, 0, 1, 1)
        self._dragging = False
        self._drag_offset = QPointF(0, 0)
        self._faces: list = []
        self._active_id = -1
        self._draw_faces = False
        self._drag_enabled = True
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def set_faces(self, faces: list, active_id: int) -> None:
        self._faces = faces or []
        self._active_id = active_id
        self.update()

    def set_draw_faces(self, on: bool) -> None:
        self._draw_faces = on
        self.update()

    def set_drag_enabled(self, on: bool) -> None:
        self._drag_enabled = on

    def set_image_size(self, w: int, h: int) -> None:
        self.prepareGeometryChange()
        self._img_w, self._img_h = max(1, w), max(1, h)

    def set_crop(self, rect: CropRect) -> None:
        self._rect = QRectF(rect.x, rect.y, rect.width, rect.height)
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._img_w, self._img_h)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ANN001
        # Tối hoá vùng ngoài khung cắt.
        outer = QPainterPath()
        outer.addRect(QRectF(0, 0, self._img_w, self._img_h))
        inner = QPainterPath()
        inner.addRect(self._rect)
        painter.fillPath(outer.subtracted(inner), QColor(0, 0, 0, 130))

        # Viền khung cắt.
        pen = QPen(QColor(0, 220, 255), max(2, int(self._img_w / 400)))
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(self._rect)

        # Đường 1/3 (rule of thirds).
        thin = QPen(QColor(255, 255, 255, 90), 1)
        painter.setPen(thin)
        for i in (1, 2):
            x = self._rect.left() + self._rect.width() * i / 3
            painter.drawLine(QPointF(x, self._rect.top()), QPointF(x, self._rect.bottom()))
            y = self._rect.top() + self._rect.height() * i / 3
            painter.drawLine(QPointF(self._rect.left(), y), QPointF(self._rect.right(), y))

        # Khuôn mặt + đánh dấu người nói (chế độ Auto).
        if self._draw_faces:
            lw = max(2, int(self._img_w / 500))
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            for f in self._faces:
                is_active = f.track_id is not None and f.track_id == self._active_id
                color = QColor(255, 210, 0) if is_active else QColor(120, 255, 120)
                painter.setPen(QPen(color, lw + (1 if is_active else 0)))
                painter.drawRect(QRectF(f.x, f.y, f.w, f.h))

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if not self._drag_enabled:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._rect.contains(event.pos()):
            self._dragging = True
            self._drag_offset = event.pos() - self._rect.center()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._dragging:
            center = event.pos() - self._drag_offset
            self._view.notify_center(center.x(), center.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self._dragging = False
        super().mouseReleaseEvent(event)


def _pt_dist(a: QPointF, b: QPointF) -> float:
    """Khoảng cách Euclid giữa 2 QPointF (không dùng manhattanLength — không có ở mọi bản Qt)."""
    return ((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5


class _ComposeGizmo(QGraphicsItem):
    """Gizmo compose: viền quanh foreground + 4 tay nắm góc (kéo = scale) + nút reset.

    Toạ độ theo pixel CANVAS. Kéo góc đổi cỡ theo tỉ lệ khoảng cách góc→tâm foreground,
    phát fgScaleDragged(scale_mới). Bấm nút tròn dưới -> fgResetClicked.
    """

    HANDLE = 9  # bán kính tay nắm (px canvas)

    def __init__(self, view: "PreviewView"):
        super().__init__()
        self._view = view
        self._cw, self._ch = 1, 1
        self._fg = QRectF(0, 0, 1, 1)   # rect foreground trên canvas (có thể tràn)
        self._cur_scale = 1.0
        self._drag = False
        self._half0 = 1.0               # nửa đường chéo fg lúc bắt đầu kéo
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def set_canvas(self, cw: int, ch: int) -> None:
        self.prepareGeometryChange()
        self._cw, self._ch = max(1, cw), max(1, ch)

    def set_fg(self, x: int, y: int, w: int, h: int, scale: float) -> None:
        self.prepareGeometryChange()
        self._fg = QRectF(x, y, w, h)
        self._cur_scale = scale
        self.update()

    def _reset_center(self) -> QPointF:
        return QPointF(self._cw / 2, self._ch + self.HANDLE * 3)

    def boundingRect(self) -> QRectF:
        r = self._fg.adjusted(-self.HANDLE, -self.HANDLE, self.HANDLE, self.HANDLE)
        rc = self._reset_center()
        return r.united(QRectF(0, 0, self._cw, self._ch)).united(
            QRectF(rc.x() - self.HANDLE, rc.y() - self.HANDLE, self.HANDLE * 2, self.HANDLE * 2))

    def paint(self, painter, option, widget=None) -> None:  # noqa: ANN001
        pen = QPen(QColor(255, 255, 255), max(2, int(self._cw / 400)))
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(self._fg)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        for c in (self._fg.topLeft(), self._fg.topRight(),
                  self._fg.bottomLeft(), self._fg.bottomRight()):
            painter.drawEllipse(c, self.HANDLE, self.HANDLE)
        rc = self._reset_center()
        painter.setBrush(QBrush(QColor(30, 32, 40)))
        painter.drawEllipse(rc, self.HANDLE + 2, self.HANDLE + 2)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawArc(QRectF(rc.x() - self.HANDLE, rc.y() - self.HANDLE,
                               self.HANDLE * 2, self.HANDLE * 2), 30 * 16, 300 * 16)

    def _near(self, a: QPointF, b: QPointF, r: float) -> bool:
        return _pt_dist(a, b) <= r * 1.6

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        pos = event.pos()
        if self._near(pos, self._reset_center(), self.HANDLE + 2):
            self._view.emit_fg_reset()
            event.accept()
            return
        corners = (self._fg.topLeft(), self._fg.topRight(),
                   self._fg.bottomLeft(), self._fg.bottomRight())
        if any(self._near(pos, c, self.HANDLE + 3) for c in corners):
            self._drag = True
            self._half0 = max(1.0, _pt_dist(self._fg.center(), pos))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._drag:
            half = max(1.0, _pt_dist(self._fg.center(), event.pos()))
            new_scale = self._cur_scale * (half / self._half0)
            self._view.emit_fg_scale(new_scale)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self._drag = False
        super().mouseReleaseEvent(event)


class PreviewView(QGraphicsView):
    cropCenterChanged = pyqtSignal(float, float)  # tâm mong muốn (pixel ảnh gốc)
    fgScaleDragged = pyqtSignal(float)            # cỡ mới khi kéo góc gizmo
    fgResetClicked = pyqtSignal()                 # bấm nút reset trên gizmo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor(24, 24, 28))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(360, 360)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._overlay = _CropOverlay(self)
        self._scene.addItem(self._overlay)
        self._overlay.setVisible(False)
        self._overlay_wanted = True  # ý muốn hiện overlay của caller, độc lập set_frame()
        self._gizmo = _ComposeGizmo(self)
        self._scene.addItem(self._gizmo)
        self._gizmo.setVisible(False)
        self._compose_mode = False

    def set_frame(self, frame: np.ndarray) -> None:
        self._pixmap_item.setPixmap(ndarray_to_qpixmap(frame))
        h, w = frame.shape[:2]
        self._overlay.set_image_size(w, h)
        self._overlay.setVisible(self._overlay_wanted and not self._compose_mode)
        self._scene.setSceneRect(0, 0, w, h)
        self.fit()

    def set_compose_mode(self, on: bool) -> None:
        self._compose_mode = on
        self._overlay.setVisible(self._overlay_wanted and not on)
        self._gizmo.setVisible(on)

    def set_compose(self, canvas_bgr, fg_x, fg_y, fg_w, fg_h, scale) -> None:  # noqa: ANN001
        """Chế độ nền mờ: hiện canvas đã ghép + gizmo scale quanh foreground."""
        h, w = canvas_bgr.shape[:2]
        self._pixmap_item.setPixmap(ndarray_to_qpixmap(canvas_bgr))
        self._gizmo.set_canvas(w, h)
        self._gizmo.set_fg(fg_x, fg_y, fg_w, fg_h, scale)
        rect = self._gizmo.boundingRect()
        self._scene.setSceneRect(rect)     # mở rộng để thấy tay nắm tràn ra ngoài canvas
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def emit_fg_scale(self, scale: float) -> None:
        self.fgScaleDragged.emit(float(scale))

    def emit_fg_reset(self) -> None:
        self.fgResetClicked.emit()

    def set_crop_rect(self, rect: CropRect) -> None:
        self._overlay.set_crop(rect)

    def set_auto_crop_rect(self, rect: CropRect) -> None:
        self._overlay.set_crop(rect)

    def set_faces(self, faces: list, active_track_id: int) -> None:
        self._overlay.set_faces(faces, active_track_id)

    def clear_faces(self) -> None:
        self._overlay.set_faces([], -1)

    def set_auto_mode(self, on: bool) -> None:
        self._overlay.set_drag_enabled(not on)
        self._overlay.set_draw_faces(on)

    def set_overlay_enabled(self, on: bool) -> None:
        """Ẩn/hiện lớp overlay (khung cắt + tối hoá + khuôn mặt). Tắt khi xem trước nền mờ."""
        self._overlay_wanted = on
        self._overlay.setVisible(on)

    def fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.fit()

    def notify_center(self, x: float, y: float) -> None:
        self.cropCenterChanged.emit(float(x), float(y))
