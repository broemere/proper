from PySide6.QtCore import Qt, Signal, QPointF, QLineF
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsLineItem
from PySide6.QtGui import QPixmap, QColor, QPainter, QPen
import math


class MeasurementLineItem(QGraphicsLineItem):
    """
    A custom line item that draws perpendicular caps at its endpoints.
    """

    def __init__(self, line, parent=None):
        super().__init__(line, parent)
        self.cap_length = 4.0  # The length of the end caps in pixels

    def boundingRect(self):
        """
        Returns the bounding rectangle, expanded to include the end caps.
        This is crucial for ensuring the item is redrawn correctly.
        """
        rect = super().boundingRect()
        buffer = self.pen().widthF() + self.cap_length
        return rect.adjusted(-buffer, -buffer, buffer, buffer)

    def paint(self, painter: QPainter, option, widget=None):
        """
        Handles all the drawing for this item.
        """
        painter.setPen(self.pen())
        main_line = self.line()

        # 1. Draw the main measurement line
        painter.drawLine(main_line)

        # 2. Calculate and draw the perpendicular end caps
        if main_line.length() > 0:
            # --- THIS IS THE CORRECTED LOGIC ---
            # Get the angle of the main line in degrees
            angle = main_line.angle()

            # Define a line for one half of the cap, rotated 90 degrees
            # from the main line's angle. QLineF.fromPolar is perfect for this.
            half_cap_vector = QLineF.fromPolar(self.cap_length / 2.0, angle + 90.0)

            # Create the full cap lines by translating this vector to each endpoint
            p1_cap = QLineF(main_line.p1() + half_cap_vector.p2(), main_line.p1() - half_cap_vector.p2())
            p2_cap = QLineF(main_line.p2() + half_cap_vector.p2(), main_line.p2() - half_cap_vector.p2())

            painter.drawLine(p1_cap)
            painter.drawLine(p2_cap)


class ThicknessCanvas(QGraphicsView):
    """
    A QGraphicsView-based canvas for drawing lines on a zoomable/pannable image.
    It automatically handles scroll bars and coordinate transformations.
    """
    lines_updated = Signal(list)

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
        self._completed_lines = []
        self._active_line_item = None

        # --- Drawing Style ---
        self._line_color = QColor("red")
        self._line_pen = QPen(self._line_color, 2)
        self._line_pen.setCosmetic(True)

    def set_background(self, pixmap: QPixmap):
        """Clears the scene and sets a new background image."""
        self.clear()
        self._image_item = self._scene.addPixmap(pixmap)
        self.reset_view()
        self._emit_lines_updated()

    def reset_view(self):
        """Resets the view to fit the entire image within the viewport."""
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def clear(self):
        """Clears all items from the canvas."""
        self._scene.clear()
        self._image_item = None
        self._active_line_item = None
        self._completed_lines.clear()

    def undo_last_line(self):
        """Removes the last drawn line from the canvas."""
        if self._completed_lines:
            last_line = self._completed_lines.pop()
            self._scene.removeItem(last_line)
            self._emit_lines_updated()

    def _emit_lines_updated(self):
        """Calculates line lengths and emits the signal."""
        lengths = [item.line().length() for item in self._completed_lines]
        self.lines_updated.emit(lengths)

    # --- NEW: Helper to cancel the active line ---
    def _cancel_active_line(self):
        """Removes the currently active (uncompleted) line."""
        if self._active_line_item:
            self._scene.removeItem(self._active_line_item)
            self._active_line_item = None

    # --- NEW: Centralized zoom logic ---
    def _zoom(self, factor):
        """
        Helper function to apply a zoom factor, centered on the mouse cursor.
        """
        if self._image_item is None:
            return

        # For zooming out, first check if we are already zoomed in.
        # This prevents getting stuck when fully zoomed out.
        if factor < 1.0:
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            if h_bar.maximum() <= 0 and v_bar.maximum() <= 0:
                return

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)

    # --- Reimplemented Events for Interaction ---

    def wheelEvent(self, event):
        """Handles zooming and panning via the mouse wheel."""
        if self._image_item is None:
            return

        angle = event.angleDelta().y()

        # MODIFIED: Call the new helper method for zooming
        if event.modifiers() == Qt.ControlModifier:
            if angle > 0:  # Zoom In
                self._zoom(1.15)
            else:  # Zoom Out
                self._zoom(1 / 1.15)

        elif event.modifiers() == Qt.ShiftModifier:
            h_bar = self.horizontalScrollBar()
            h_bar.setValue(h_bar.value() - angle)

        else:
            super().wheelEvent(event)

    # --- MODIFIED: Cancels the line created by the first click ---
    def mouseDoubleClickEvent(self, event):
        """Resets the view on double-click and cancels any accidental line."""
        if event.button() == Qt.LeftButton:
            # A double-click's first press creates an active line. Cancel it.
            self._cancel_active_line()
            self.reset_view()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Handles the start and end points of a line."""
        if event.button() != Qt.LeftButton or self._image_item is None:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if not self._image_item.boundingRect().contains(scene_pos):
            return

        if self._active_line_item is None:
            # Create an instance of our new custom item instead of a standard line
            self._active_line_item = MeasurementLineItem(QLineF(scene_pos, scene_pos))
            self._active_line_item.setPen(self._line_pen)
            self._scene.addItem(self._active_line_item)
        else:
            current_line = self._active_line_item.line()
            current_line.setP2(scene_pos)
            self._active_line_item.setLine(current_line)  # This triggers a repaint

            self._completed_lines.append(self._active_line_item)
            self._active_line_item = None
            self._emit_lines_updated()

    def mouseMoveEvent(self, event):
        """Updates the active line's endpoint as the mouse moves."""
        if self._active_line_item:
            scene_pos = self.mapToScene(event.pos())
            if self._image_item.boundingRect().contains(scene_pos):
                current_line = self._active_line_item.line()
                current_line.setP2(scene_pos)
                self._active_line_item.setLine(current_line)

        super().mouseMoveEvent(event)

    # --- NEW: Handle keyboard shortcuts ---
    def keyPressEvent(self, event):
        """Handles keyboard shortcuts for canceling and undoing lines."""
        # --- Press Escape to cancel an active line ---
        if event.key() == Qt.Key_Escape:
            self._cancel_active_line()

        # --- Press Ctrl+Z to undo ---
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            # If a line is being drawn, cancel it. Otherwise, undo the last completed line.
            if self._active_line_item:
                self._cancel_active_line()
            else:
                self.undo_last_line()

        # --- NEW: Handle zooming shortcuts ---
        # Keys for Zoom In: = and + (which is Shift+=)
        elif event.key() in (Qt.Key_Equal, Qt.Key_Plus):
            self._zoom(1.5)
            event.accept()

        # Keys for Zoom Out: - and _ (which is Shift+-)
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom(1 / 1.5)
            event.accept()

        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """The base implementation handles resizing the viewport."""
        super().resizeEvent(event)
