from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen, QImage
from PySide6.QtWidgets import QWidget
import numpy as np
from processing.data_transform import qimage_to_numpy


class PolygonCanvas(QWidget):
    color_changed = Signal(QColor)
    tool_changed = Signal(str)
    image_flattened = Signal(np.ndarray)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.completed_polygons = []
        self.active_polygon = None
        self.current_pos = None
        self.background_pixmap = None
        self.current_tool = 'polygon'
        self.is_drawing_lasso = False
        self.final_color = QColor(Qt.black)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.CLOSE_THRESHOLD = 10
        self.PEN_WIDTH = 2
        self.ACTIVE_COLOR = QColor("red")
        self.CLOSING_COLOR = QColor(0, 255, 0)

    def set_tool(self, tool_name: str):
        """Sets the active drawing tool ('polygon' or 'lasso')."""
        if tool_name in ['polygon', 'lasso'] and self.current_tool != tool_name:
            self.current_tool = tool_name

            # Cancel any active drawing when switching tools
            self.active_polygon = None
            self.is_drawing_lasso = False
            self.current_pos = None
            self.update()

    def set_background(self, pixmap: QPixmap):
        """Set a fixed-size background and reset polygons."""
        self.background_pixmap = pixmap
        self.setFixedSize(pixmap.size())  # enforce 1:1 mapping
        self.completed_polygons = []
        self.active_polygon = None
        self.update()
        self._flatten_and_emit()

    def set_final_color(self, color: QColor):
        """Update the default color for future polygons."""
        self.final_color = color
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = event.pos()

        # --- Polygon Tool Logic ---
        if self.current_tool == 'polygon':
            if self.active_polygon is None:
                self.active_polygon = {'points': [pos], 'color': None}
            else:
                pts = self.active_polygon['points']
                if len(pts) >= 3 and (pos - pts[0]).manhattanLength() < self.CLOSE_THRESHOLD:
                    self.active_polygon['color'] = QColor(self.final_color)
                    self.completed_polygons.append(self.active_polygon)
                    self.active_polygon = None
                    self.current_pos = None
                    self._flatten_and_emit()
                else:
                    pts.append(pos)
        # --- Lasso Tool Logic ---
        elif self.current_tool == 'lasso':
            self.active_polygon = {'points': [pos], 'color': None}
            self.is_drawing_lasso = True

        self.update()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        # --- Polygon Tool Logic ---
        if self.current_tool == 'polygon':
            if self.active_polygon is not None:
                self.current_pos = pos
                self.update()
        # --- Lasso Tool Logic ---
        elif self.current_tool == 'lasso':
            if self.is_drawing_lasso:
                self.active_polygon['points'].append(pos)
                self.update()

    def mouseReleaseEvent(self, event):
        """This event is now used to finalize the lasso drawing."""
        if event.button() != Qt.LeftButton:
            return

        if self.current_tool == 'lasso' and self.active_polygon:
            self.is_drawing_lasso = False
            if len(self.active_polygon['points']) > 2:
                self.active_polygon['color'] = QColor(self.final_color)
                self.completed_polygons.append(self.active_polygon)
                self._flatten_and_emit()  # Add this call

            self.active_polygon = None
            self.update()

    def keyPressEvent(self, event):
        # Ctrl + Z cancels the current polygon or undoes the previously drawn polygon
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self.active_polygon is not None:
                self.active_polygon = None
                self.current_pos = None  # Reset live preview
                self.is_drawing_lasso = False
                self.update()
            else:
                self.undo_last_polygon()
            return

        # Esc cancels the in-progress polygon only
        if event.key() == Qt.Key_Escape and self.active_polygon is not None:
            self.active_polygon = None
            self.current_pos = None  # Reset live preview
            self.is_drawing_lasso = False
            self.update()
            return

        # Toggle color with Spacebar
        if event.key() == Qt.Key_Space:
            if self.final_color == QColor(Qt.black):
                self.final_color = QColor(Qt.white)
            else:
                self.final_color = QColor(Qt.black)
            # Emit the signal to notify the main UI of the change
            self.color_changed.emit(self.final_color)
            self.update()  # Schedule a repaint to reflect potential live-edge color changes
            return

        # T changes the tool
        if event.key() == Qt.Key_T:
            if self.current_tool == 'polygon':
                new_tool = 'lasso'
            else:
                new_tool = 'polygon'
            self.tool_changed.emit(new_tool)
            return

        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # background
        if self.background_pixmap:
            painter.drawPixmap(0, 0, self.background_pixmap)

        # 2. Draw all the completed, filled polygons
        for poly in self.completed_polygons:
            color = poly['color']
            painter.setBrush(color)
            painter.setPen(QPen(color, self.PEN_WIDTH))
            painter.drawPolygon(poly['points'])

        # 3. Draw the active polygon (if it exists)
        if self.active_polygon:
            pts = self.active_polygon['points']
            if not pts:
                return

            painter.setBrush(Qt.NoBrush)
            pen_color = self.ACTIVE_COLOR

            # For polygon tool, check for closing proximity
            if self.current_tool == 'polygon' and self.current_pos and len(pts) >= 3:
                if (self.current_pos - pts[0]).manhattanLength() < self.CLOSE_THRESHOLD:
                    pen_color = self.CLOSING_COLOR

            painter.setPen(QPen(pen_color, self.PEN_WIDTH))

            # Draw the active shape's segments
            if len(pts) > 1:
                painter.drawPolyline(pts)

            # Draw the "live edge" for the polygon tool
            if self.current_tool == 'polygon' and self.current_pos:
                painter.drawLine(pts[-1], self.current_pos)

    def undo_last_polygon(self):
        """Remove the most recently completed polygon."""
        if self.completed_polygons:
            self.completed_polygons.pop()
            self.update()
            self._flatten_and_emit()

    def _flatten_and_emit(self):
        """
        Draws polygons onto a QImage, converts it to a NumPy array,
        and emits the array.
        """
        if not self.background_pixmap:
            return

        # 1. Create and draw on a temporary QImage
        w, h = self.background_pixmap.width(), self.background_pixmap.height()
        temp_image = QImage(w, h, QImage.Format_Grayscale8)

        painter = QPainter(temp_image)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.drawPixmap(0, 0, self.background_pixmap)
        for poly in self.completed_polygons:
            painter.setBrush(poly['color'])
            painter.setPen(QPen(poly['color'], 1))
            painter.drawPolygon(poly['points'])
        painter.end()

        # 2. Immediately convert the QImage to a NumPy array
        final_array = qimage_to_numpy(temp_image)
        self.image_flattened.emit(final_array)
