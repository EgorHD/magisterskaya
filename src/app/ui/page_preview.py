from __future__ import annotations
import math
from typing import Iterable, Optional, List, Tuple

from PyQt6.QtCore import Qt, QRectF, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from PIL import Image
from core.models.ocr import OCRPageResult, OCRSpan


# Перевод изображения PIL в QPixmap для отображения в Qt
def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# Безопасное извлечение координат bounding box
def _safe_bbox(span: OCRSpan) -> Optional[tuple[int, int, int, int]]:
    try:
        x1, y1, x2, y2 = span.bbox()
    except Exception:
        return None

    try:
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
    except Exception:
        return None

    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return int(x1), int(y1), int(x2), int(y2)


class PagePreview(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._img_item = None
        self._img_w = 0
        self._img_h = 0

        # Храним рамки как простые координаты
        self._boxes: List[Tuple[int, int, int, int]] = []

        self.max_boxes = 2000
        self.box_width = 1

        self._zoom = 0
        self._zoom_step = 1.25
        self._zoom_min = -15
        self._zoom_max = 30
        self._auto_fit = True

        self._panning = False
        self._pan_start = QPoint()

    def clear(self) -> None:
        self._scene.clear()
        self._img_item = None
        self._img_w = 0
        self._img_h = 0
        self._boxes.clear()
        self.resetTransform()
        self._zoom = 0
        self._auto_fit = True
        self.viewport().update()

    def set_page(
        self,
        pil_image: Image.Image,
        ocr_page: Optional[OCRPageResult] = None,
    ) -> None:
        self.clear()

        base = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        pix = pil_to_qpixmap(base)

        self._img_w = pix.width()
        self._img_h = pix.height()

        self._img_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(0, 0, self._img_w, self._img_h))

        if ocr_page is not None and ocr_page.spans:
            self._store_boxes(ocr_page.spans)

        if self._auto_fit:
            self.fit_to_view()

        self.viewport().update()

    def fit_to_view(self) -> None:
        if not self._img_item:
            return

        self.resetTransform()
        self._zoom = 0
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_in(self) -> None:
        if self._zoom >= self._zoom_max:
            return
        self._zoom += 1
        self.scale(self._zoom_step, self._zoom_step)
        self._auto_fit = False
        self.viewport().update()

    def zoom_out(self) -> None:
        if self._zoom <= self._zoom_min:
            return
        self._zoom -= 1
        self.scale(1 / self._zoom_step, 1 / self._zoom_step)
        self._auto_fit = False
        self.viewport().update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_in()
        elif delta < 0:
            self.zoom_out()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )

            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self._auto_fit = True
        self.fit_to_view()
        self.viewport().update()
        super().mouseDoubleClickEvent(event)

    def _store_boxes(self, spans: Iterable[OCRSpan]) -> None:
        w, h = self._img_w, self._img_h
        count = 0

        for sp in spans:
            if count >= self.max_boxes:
                break

            bb = _safe_bbox(sp)
            if bb is None:
                continue

            x1, y1, x2, y2 = bb

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue

            self._boxes.append((x1, y1, x2, y2))
            count += 1

    # Рисуем рамки поверх сцены, без создания отдельных QGraphicsRectItem
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)

        if not self._boxes:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(0)          # 1 экранный пиксель
        pen.setCosmetic(True)    # не зависит от масштаба
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        visible = rect.adjusted(-2, -2, 2, 2)

        for x1, y1, x2, y2 in self._boxes:
            box = QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            if box.intersects(visible):
                painter.drawRect(box)

        painter.restore()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_fit and self._img_item:
            self.fit_to_view()
        self.viewport().update()