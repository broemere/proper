from PySide6.QtCore import Qt, Signal, QPointF, QRect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup
)
from PySide6.QtGui import QPalette, QPixmap, QColor, QPainter, QImage, QPen, QCursor, QIcon
from data_pipeline import DataPipeline
import numpy as np
from processing.resource_loader import load_cursor


class ScaledLineCanvas(QWidget):
    """
    A canvas that:
      • Always scales its background image to fit the widget (preserving aspect ratio).
      • Lets the user draw exactly one line (two clicks), but does NOT erase
        the old line until the new line is finished.
      • Lets the user draw a zoom‐box (two clicks) to crop/zoom the image,
        then automatically returns to line mode.
      • Emits mode_changed(str) whenever the mode flips.
    """

    mode_changed = Signal(str)
    line_completed = Signal(float)  # emit “pixel length” when a line is finished

    def __init__(self, parent=None):
        super().__init__(parent)

        # Stores the full-resolution QPixmap of the background:
        self._pix_full: QPixmap | None = None

        # “Official” line (image-space coords):
        #   'points': [QPointF, QPointF] or fewer
        #   'complete': bool
        #   'color': QColor
        self.line = {
            'points': [],
            'complete': False,
            'color': QColor(Qt.black)
        }

        # If the user begins a new line while one is already complete,
        # we stash the old one here until the new one finishes or is cancelled.
        self._old_line = None  # type: dict[str, object] | None

        # For zoom: store up to two image-space corners
        self.zoom_pts: list[QPointF] = []

        # Last mouse position (widget coords) for preview (line or zoom box)
        self._mouse_pos: QPointF | None = None

        # Current mode: either 'line' or 'zoom'
        self.mode = 'zoom'

        # Default final color for any newly completed line
        self.final_color = QColor(Qt.green)

        # Cached values computed each paintEvent:
        self._scaled_pix: QPixmap | None = None
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._scale_factor: float = 1.0

        self._zoom_cursor = load_cursor("zoom", hot_x=1, hot_y=1)
        self._scale_cursor = load_cursor("scale", hot_x=1, hot_y=1)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ——————————————
    # Public API

    def set_background(self, pixmap: QPixmap):
        """
        Replace the full-res background pixmap, clear any lines or zooms.
        """
        self._pix_full = pixmap

        # Reset the official line
        self.line = {
            'points': [],
            'complete': False,
            'color': QColor(self.final_color)
        }
        self._old_line = None
        # Reset any in-progress zoom
        self.zoom_pts = []
        self._mouse_pos = None
        self.set_mode("zoom")
        self.update()

    def set_final_color(self, color: QColor):
        """
        Change default color for new lines. If a line is already complete,
        recolor it immediately.
        """
        self.final_color = color
        if self.line['complete']:
            self.line['color'] = QColor(color)
        self.update()

    def set_mode(self, mode: str):
        """
        Switch between 'line' and 'zoom' modes. Cancels any half-done line or zoom.
        Emits mode_changed if the mode actually switches.
        """
        if mode not in ('line', 'zoom') or mode == self.mode:
            return

        old_mode = self.mode
        self.mode = mode

        # If we had stashed an old line (_old_line) but were mid-drawing a new one,
        # restore the old line now, because switching mode cancels the new.
        if self._old_line is not None and not self.line['complete']:
            self.line = self._old_line
            self._old_line = None

        # Cancel any incomplete line
        if not self.line['complete']:
            self.line = {
                'points': [],
                'complete': False,
                'color': QColor(self.final_color)
            }
            self._old_line = None

        # Cancel any incomplete zoom
        self.zoom_pts = []
        self._mouse_pos = None

        self.update()
        self.mode_changed.emit(self.mode)

        if mode != old_mode:
            self._update_cursor()

    def undo_last_line(self):
        """
        If there is a completed line, remove it immediately.
        """
        if self.line['complete']:
            self.line = {
                'points': [],
                'complete': False,
                'color': QColor(self.final_color)
            }
            self._old_line = None
            self._mouse_pos = None
            self.update()

    def get_line_length(self) -> float | None:
        """
        Return the length in pixels of the completed line (in full-image coords),
        or None if no line is complete.
        """
        pts = self.line['points']
        if self.line['complete'] and len(pts) == 2:
            dx = pts[1].x() - pts[0].x()
            dy = pts[1].y() - pts[0].y()
            return (dx * dx + dy * dy) ** 0.5
        return None

    # ——————————————
    # Mouse & key events

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._pix_full is None:
            return

        img_pt = self._widget_to_image(event.position())
        if img_pt is None:
            return

        # 1) Zoom mode
        if self.mode == 'zoom':
            if not self.zoom_pts:
                # First corner
                self.zoom_pts.append(img_pt)
            else:
                # Second corner → complete the zoom
                self.zoom_pts.append(img_pt)
                self._apply_zoom_box()
            return

        # 2) Line mode
        pts = self.line['points']
        if self.line['complete']:
            # There is already a completed line. The user clicked to start a new one.
            # Save the old line so we can restore it if the new one is canceled.
            self._old_line = {
                'points': [QPointF(pts[0]), QPointF(pts[1])],
                'complete': True,
                'color': QColor(self.line['color'])
            }
            # Start the new line with this first endpoint
            self.line = {
                'points': [img_pt],
                'complete': False,
                'color': QColor(self.final_color)
            }
            self._mouse_pos = None
            self.update()
            return

        # If no old line (or line not complete) …
        if not pts:
            # First endpoint of a brand-new line
            self.line['points'].append(img_pt)
        else:
            # One endpoint exists but not complete → second click completes the line
            if len(pts) == 1:
                self.line['points'].append(img_pt)
                self.line['complete'] = True
                self.line['color'] = QColor(self.final_color)

                # Discard _old_line because the new line is now official
                self._old_line = None

                # Instead of manually computing, just call get_line_length():
                length_px = self.get_line_length() or 0.0
                self.line_completed.emit(length_px)
        self.update()

    def mouseMoveEvent(self, event):
        if self._pix_full is None:
            return

        self._mouse_pos = event.position()

        # In zoom mode, preview only if first corner is set
        if self.mode == 'zoom' and len(self.zoom_pts) == 1:
            self.update()
            return

        # In line mode, preview only if exactly one endpoint and not complete
        if self.mode == 'line' and self.line['points'] and not self.line['complete']:
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # If user presses Esc while drawing a new line (and old_line is stashed),
            # revert to the old line:
            if self._old_line is not None and not self.line['complete']:
                self.line = self._old_line
                self._old_line = None
                self._mouse_pos = None
                self.update()
                return

            # If user pressed Esc while drawing a brand-new line (no old_line),
            # simply cancel it:
            if self.mode == 'line' and self.line['points'] and not self.line['complete']:
                self.line = {
                    'points': [],
                    'complete': False,
                    'color': QColor(self.final_color)
                }
                self._old_line = None
                self._mouse_pos = None
                self.update()
                return

            # If user pressed Esc while zooming (one corner set)
            if self.mode == 'zoom' and len(self.zoom_pts) == 1:
                self.zoom_pts = []
                self._mouse_pos = None
                self.update()
                return

        super().keyPressEvent(event)

    # ——————————————
    # Painting & scaling

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._pix_full is None:
            return

        widget_w = self.width()
        widget_h = self.height()
        pix_w = self._pix_full.width()
        pix_h = self._pix_full.height()

        # 1) Scale the full pixmap to fit, preserving aspect ratio.
        scaled_pix = self._pix_full.scaled(
            widget_w, widget_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self._scaled_pix = scaled_pix
        self._offset_x = (widget_w - scaled_pix.width()) / 2
        self._offset_y = (widget_h - scaled_pix.height()) / 2
        self._scale_factor = scaled_pix.width() / pix_w

        # Draw the scaled background
        painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled_pix)

        # 2) If in zoom mode, preview rectangle
        if self.mode == 'zoom' and len(self.zoom_pts) == 1 and self._mouse_pos:
            p0 = self._image_to_widget(self.zoom_pts[0])
            p1 = self._mouse_pos
            if p0 and p1:
                rect = QRect(p0.toPoint(), p1.toPoint()).normalized()
                pen = QPen(Qt.red, 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
            return

        # 3) In line mode, draw the old line if it exists
        if self.mode == 'line' and self._old_line is not None:
            pts_old = self._old_line['points']
            if len(pts_old) == 2:
                p0_old = self._image_to_widget(pts_old[0])
                p1_old = self._image_to_widget(pts_old[1])
                if p0_old and p1_old:
                    pen_old = QPen(self._old_line['color'], 2)
                    painter.setPen(pen_old)
                    painter.drawLine(p0_old, p1_old)

        # 4) Now draw either the completed new line, or a preview
        if self.mode == 'line':
            pts = self.line['points']
            if self.line['complete'] and len(pts) == 2:
                p0 = self._image_to_widget(pts[0])
                p1 = self._image_to_widget(pts[1])
                if p0 and p1:
                    pen = QPen(self.line['color'], 2)
                    painter.setPen(pen)
                    painter.drawLine(p0, p1)

            elif len(pts) == 1 and self._mouse_pos:
                p0 = self._image_to_widget(pts[0])
                p1 = self._mouse_pos
                if p0:
                    pen = QPen(Qt.red, 2)
                    painter.setPen(pen)
                    painter.drawLine(p0, p1)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._update_cursor()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        # back to whatever the OS default is
        self.unsetCursor()

    # ——————————————
    # Private helpers

    def _update_cursor(self):
        if self.mode == "zoom":
            # show your custom zoom-in cursor
            self.setCursor(self._zoom_cursor)
        else:
            # line mode or anything else: go back to default arrow
            self.setCursor(self._scale_cursor)

    def _widget_to_image(self, pt: QPointF) -> QPointF | None:
        """
        Convert a point in widget-space to image-space (float), or return None if
        outside the drawn image area.
        """
        if self._pix_full is None or self._scaled_pix is None:
            return None

        x_w, y_w = pt.x(), pt.y()
        x0, y0 = self._offset_x, self._offset_y
        x1 = x_w - x0
        y1 = y_w - y0

        if x1 < 0 or y1 < 0 or x1 > self._scaled_pix.width() or y1 > self._scaled_pix.height():
            return None

        inv_scale = 1.0 / self._scale_factor
        img_x = x1 * inv_scale
        img_y = y1 * inv_scale
        return QPointF(img_x, img_y)

    def _image_to_widget(self, pt: QPointF) -> QPointF | None:
        """
        Convert a point in image-space to widget-space (float).
        """
        if self._pix_full is None or self._scaled_pix is None:
            return None

        x_img, y_img = pt.x(), pt.y()
        x_w = x_img * self._scale_factor + self._offset_x
        y_w = y_img * self._scale_factor + self._offset_y
        return QPointF(x_w, y_w)

    def _apply_zoom_box(self):
        """
        Once two corners are clicked in zoom mode, crop the pixmap to that rectangle,
        reset state, and switch back to line mode.
        """
        if self._pix_full is None or len(self.zoom_pts) != 2:
            self.zoom_pts = []
            return

        p0, p1 = self.zoom_pts
        x0 = max(0, min(p0.x(), p1.x()))
        y0 = max(0, min(p0.y(), p1.y()))
        x1 = max(0, max(p0.x(), p1.x()))
        y1 = max(0, max(p0.y(), p1.y()))

        i_x0 = int(round(x0))
        i_y0 = int(round(y0))
        i_x1 = int(round(x1))
        i_y1 = int(round(y1))

        i_x0 = max(0, min(i_x0, self._pix_full.width() - 1))
        i_y0 = max(0, min(i_y0, self._pix_full.height() - 1))
        i_x1 = max(0, min(i_x1, self._pix_full.width() - 1))
        i_y1 = max(0, min(i_y1, self._pix_full.height() - 1))

        w = max(1, i_x1 - i_x0 + 1)
        h = max(1, i_y1 - i_y0 + 1)
        crop_rect = QRect(i_x0, i_y0, w, h)

        cropped = self._pix_full.copy(crop_rect)
        self.set_background(cropped)

        self.zoom_pts = []
        self._mouse_pos = None

        # Switch back to line via set_mode so the signal is emitted
        self.set_mode('line')
        self.update()