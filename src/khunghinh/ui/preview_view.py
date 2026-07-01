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


class PreviewView(QGraphicsView):
    cropCenterChanged = pyqtSignal(float, float)  # tâm mong muốn (pixel ảnh gốc)

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

    def set_frame(self, frame: np.ndarray) -> None:
        self._pixmap_item.setPixmap(ndarray_to_qpixmap(frame))
        h, w = frame.shape[:2]
        self._overlay.set_image_size(w, h)
        self._overlay.setVisible(self._overlay_wanted)
        self._scene.setSceneRect(0, 0, w, h)
        self.fit()

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
