from PySide6.QtCore import Qt, Signal, QPointF, QLineF
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen
import math


class ScaledLineCanvas(QGraphicsView):
    """
    A QGraphicsView canvas for drawing a single measurement line on a zoomable image.
    Automatically handles scroll bars, coordinate transformations, and zooming.
    """
    line_completed = Signal(list)  # Emits: [length, x1, y1, x2, y2]

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Scene and View Setup ---
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)

        # --- State Variables ---
        self._image_item = None
        self._active_line_item = None
        self._completed_line_item = None

        # --- Drawing Styles ---
        self._active_pen = QPen(QColor("red"), 5)
        self._active_pen.setCosmetic(True)  # Keeps the line 2px thick regardless of zoom
        self._completed_pen = QPen(QColor("#00FF00"), 5)
        self._completed_pen.setCosmetic(True)

    def set_background(self, pixmap: QPixmap):
        """Clears the scene and sets a new background image."""
        self.clear()
        self._image_item = self._scene.addPixmap(pixmap)
        self.reset_view()

    def reset_view(self):
        """Resets the view to fit the entire image within the viewport."""
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def clear(self):
        """Clears all items from the canvas."""
        self._scene.clear()
        self._image_item = None
        self._active_line_item = None
        self._completed_line_item = None

    def undo_last_line(self):
        """Removes the completed line from the canvas and resets the pipeline scale."""
        if self._completed_line_item:
            self._scene.removeItem(self._completed_line_item)
            self._completed_line_item = None
            # Emit a 0-length line to safely clear the scale from the pipeline
            self.line_completed.emit([0.0, 0.0, 0.0, 0.0, 0.0])

    def _cancel_active_line(self):
        """Removes the currently active (uncompleted) red line."""
        if self._active_line_item:
            self._scene.removeItem(self._active_line_item)
            self._active_line_item = None

    def _zoom(self, factor):
        """Applies a zoom factor, centered on the mouse cursor."""
        if self._image_item is None:
            return
        if factor < 1.0:
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            if h_bar.maximum() <= 0 and v_bar.maximum() <= 0:
                return
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

    # ——————————————
    # Mouse & key events

    def wheelEvent(self, event):
        """Handles zooming and panning via the mouse wheel."""
        if self._image_item is None:
            return

        angle = event.angleDelta().y()

        if event.modifiers() == Qt.ControlModifier:
            if angle > 0:
                self._zoom(1.15)
            else:
                self._zoom(1 / 1.15)
        elif event.modifiers() == Qt.ShiftModifier:
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - angle)
        else:
            super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Resets the view on double-click and cancels any accidental line."""
        if event.button() == Qt.LeftButton:
            self._cancel_active_line()
            self.reset_view()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Handles the start and end points of the line."""
        if event.button() != Qt.LeftButton or self._image_item is None:
            super().mousePressEvent(event)
            return

        # Let the framework handle all coordinate mapping natively!
        scene_pos = self.mapToScene(event.pos())

        if not self._image_item.boundingRect().contains(scene_pos):
            return

        if self._active_line_item is None:
            # First click: Start a new active (red) line
            self._active_line_item = QGraphicsLineItem(QLineF(scene_pos, scene_pos))
            self._active_line_item.setPen(self._active_pen)
            self._scene.addItem(self._active_line_item)
        else:
            # Second click: Finish the line
            current_line = self._active_line_item.line()
            current_line.setP2(scene_pos)
            self._active_line_item.setLine(current_line)

            # 1. Clean up the old green line if it exists
            if self._completed_line_item:
                self._scene.removeItem(self._completed_line_item)

            # 2. Promote the red line to green
            self._completed_line_item = self._active_line_item
            self._completed_line_item.setPen(self._completed_pen)
            self._active_line_item = None

            # 3. Extract the clean scene data and emit
            finished_line = self._completed_line_item.line()
            length_px = finished_line.length()

            flat_data = [
                length_px,
                finished_line.x1(),
                finished_line.y1(),
                finished_line.x2(),
                finished_line.y2()
            ]
            self.line_completed.emit(flat_data)

    def mouseMoveEvent(self, event):
        """Updates the active line's endpoint as the mouse moves."""
        if self._active_line_item:
            scene_pos = self.mapToScene(event.pos())
            if self._image_item.boundingRect().contains(scene_pos):
                current_line = self._active_line_item.line()
                current_line.setP2(scene_pos)
                self._active_line_item.setLine(current_line)
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Handles keyboard shortcuts."""
        if event.key() == Qt.Key_Escape:
            self._cancel_active_line()
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self._active_line_item:
                self._cancel_active_line()
            else:
                self.undo_last_line()
        elif event.key() in (Qt.Key_Equal, Qt.Key_Plus):
            self._zoom(1.5)
            event.accept()
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom(1 / 1.5)
            event.accept()
        else:
            super().keyPressEvent(event)

    # ——————————————
    # Data Injection

    def inject_line(self, coords: list):
        """Draws a saved line directly onto the canvas."""
        self._cancel_active_line()

        # Remove any existing completed line
        if self._completed_line_item:
            self._scene.removeItem(self._completed_line_item)
            self._completed_line_item = None

        if len(coords) == 2:
            p1, p2 = coords[0], coords[1]
            line_f = QLineF(p1[0], p1[1], p2[0], p2[1])
            self._completed_line_item = QGraphicsLineItem(line_f)
            self._completed_line_item.setPen(self._completed_pen)
            self._scene.addItem(self._completed_line_item)