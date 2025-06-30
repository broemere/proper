from PySide6.QtCore import Qt, Signal, QPointF, QRect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout
)
from PySide6.QtGui import QPalette, QPixmap, QColor, QPainter, QImage, QPen
from data_pipeline import DataPipeline
import numpy as np


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

    # ——————————————
    # Private helpers

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


class ScaleTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self._has_been_shown = False  # A flag to prevent unnecessary reloads

        self.pipeline.register_observer("scale_image", self._show_scale_image)

    def showEvent(self, event):
        """
        Overridden method that is called every time the widget is shown.
        """
        # Call the parent class's implementation first
        super().showEvent(event)

        # We only need to do this the first time it's shown,
        # or you could remove this check to force a refresh every time.
        if not self._has_been_shown:
            self._has_been_shown = True

            # Proactively ask the pipeline for the current image.
            current_image = self.pipeline.scale_image
            if current_image is not None:
                # If data already exists in the pipeline, display it now.
                self._show_scale_image(current_image)

    def init_ui(self):
        tab_layout = QVBoxLayout(self)

        # --- (1) ScaledLineCanvas setup ---
        self.line_canvas = ScaledLineCanvas(parent=self)
        self.line_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tab_layout.addWidget(self.line_canvas, stretch=1)

        # Connect the new signal
        self.line_canvas.line_completed.connect(self._on_line_completed)

        # --- (2) Control buttons row ---
        ctrl_row = QHBoxLayout()

        # Refresh
        self.refresh_btn_line = QPushButton()
        self.refresh_btn_line.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn_line.setToolTip("Refresh")
        ctrl_row.addWidget(self.refresh_btn_line)
        self.refresh_btn_line.clicked.connect(self._refresh_scale)

        # Zoom / Line toggle
        self.zoom_btn = QPushButton("Zoom Box")
        self.line_btn = QPushButton("Line Mode")
        self.zoom_btn.setCheckable(True)
        self.line_btn.setCheckable(True)
        self.zoom_btn.setChecked(True)  # default to zoom mode

        self.zoom_btn.clicked.connect(lambda checked: self._set_canvas_mode('zoom', checked))
        self.line_btn.clicked.connect(lambda checked: self._set_canvas_mode('line', checked))

        # Keep buttons in sync if canvas itself changes mode
        self.line_canvas.mode_changed.connect(self._on_canvas_mode_changed)

        ctrl_row.addWidget(self.zoom_btn)
        ctrl_row.addWidget(self.line_btn)

        # Undo
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.line_canvas.undo_last_line)
        ctrl_row.addWidget(undo_btn)

        tab_layout.addLayout(ctrl_row)

        # --- (3) Scale details row ---
        scale_row = QHBoxLayout()

        # Known length [mm]
        scale_row.addWidget(QLabel("Enter Known Length [mm]:"))
        self.known_length_spin = QDoubleSpinBox()
        self.known_length_spin.setDecimals(3)
        self.known_length_spin.setRange(0.0, 1e6)
        self.known_length_spin.setValue(getattr(self.pipeline, "known_length", 0.0))
        scale_row.addWidget(self.known_length_spin)
        self.known_length_spin.valueChanged.connect(self._on_known_length_changed)

        # Pixel length [px]
        scale_row.addWidget(QLabel("Line Length [px]:"))
        self.pixel_label = QLabel("0.00")
        scale_row.addWidget(self.pixel_label)

        # Conversion [px/mm]
        scale_row.addWidget(QLabel("Conversion [px/mm]:"))
        self.conversion_label = QLabel("0.00")
        scale_row.addWidget(self.conversion_label)

        # 3d) Manual override checkbox
        self.manual_check = QCheckBox("Set manually")
        self.manual_check.stateChanged.connect(self._on_manual_override_toggled)
        scale_row.addWidget(self.manual_check)

        # 3e) Manual conversion spin (hidden by default)
        self.manual_conversion_spin = QDoubleSpinBox()
        self.manual_conversion_spin.setDecimals(4)
        self.manual_conversion_spin.setRange(0.0, 1e6)
        # Pre-set to 1.0 or to whatever conversion_label is initially
        self.manual_conversion_spin.setValue(1.0)
        self.manual_conversion_spin.setVisible(False)
        self.manual_conversion_spin.valueChanged.connect(self._on_manual_conversion_changed)
        scale_row.addWidget(self.manual_conversion_spin)
        tab_layout.addLayout(scale_row)

    def _show_scale_image(self, img_array: np.ndarray):
        # convert numpy to QPixmap and set as background
        h, w = img_array.shape[:2]
        bytes_per_line = w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.line_canvas.set_background(pix)

    def _refresh_scale(self):
        self._show_scale_image(self.pipeline.scale_image)

    def _on_canvas_mode_changed(self, new_mode: str):
        """
        Keep the Zoom/Line buttons in sync if the canvas changes mode on its own.
        """
        if new_mode == 'zoom':
            self.zoom_btn.setChecked(True)
            self.line_btn.setChecked(False)
        else:  # 'line'
            self.line_btn.setChecked(True)
            self.zoom_btn.setChecked(False)

    def _set_canvas_mode(self, mode: str, checked: bool):
        """
        Called when the user clicks either Zoom Box or Line Mode button.
        If checked=True, switch the canvas mode; if checked=False, revert to line mode.
        """
        if checked:
            self.line_canvas.set_mode(mode)
            if mode == 'zoom':
                self.line_btn.setChecked(False)
            else:
                self.zoom_btn.setChecked(False)
        else:
            # If the user unchecks either button, force line mode
            self.line_canvas.set_mode('line')
            self.line_btn.setChecked(True)
            self.zoom_btn.setChecked(False)

    def _on_line_completed(self, length_px: float):
        """
        Called when the canvas finishes a new line.
        1) Update the pixel_label to show length_px.
        2) If manual override is OFF, recalc conversion automatically.
        3) If manual override is ON, preload the manual spin with the old auto‐value.
        """
        # 1) Update pixel length label
        self.pixel_label.setText(f"{length_px:.2f}")

        # 2) Compute new auto conversion = length_px / known_length_mm
        known_mm = self.known_length_spin.value()
        if known_mm > 0:
            auto_conv = length_px / known_mm
        else:
            auto_conv = 0.0

        # 3) If manual override is OFF → show auto label
        if not self.manual_check.isChecked():
            self.conversion_label.setText(f"{auto_conv:.4f}")
        else:
            # If manual override is ON, copy the auto_conv into the spin so user can tweak
            self.manual_conversion_spin.setValue(auto_conv)

    def _on_known_length_changed(self, new_val: float):
        """
        Called whenever known_length_spin changes. If manual override is OFF and
        we already have a pixel length, recalc conversion. If manual override is ON,
        do nothing here (user controls the spin).
        """
        if self.manual_check.isChecked():
            return

        # If pixel_label holds a valid float, recalc
        try:
            px = float(self.pixel_label.text())
        except ValueError:
            return

        if new_val > 0 and px > 0:
            conv = px / new_val
            self.conversion_label.setText(f"{conv:.4f}")
        else:
            self.conversion_label.setText("0.00")

    def _on_manual_override_toggled(self, state: int):
        """
        When “Set manually” is checked, hide conversion_label and show manual spin, preloading it
        with whatever conversion_label currently says. When unchecked, hide manual spin and show
        conversion_label, then recalc auto‐conversion.
        """
        if state == Qt.Checked:
            # 1) Hide the label, show the spin
            self.conversion_label.setVisible(False)
            self.manual_conversion_spin.setVisible(True)

            # 2) Preload the spin with the current automatic conversion
            try:
                current_auto = float(self.conversion_label.text())
            except ValueError:
                current_auto = 0.0
            self.manual_conversion_spin.setValue(current_auto)

        else:
            # 1) Hide the spin, show the label
            self.manual_conversion_spin.setVisible(False)
            self.conversion_label.setVisible(True)

            # 2) Recalculate auto conversion immediately
            self._recalculate_conversion()

    def _on_manual_conversion_changed(self, new_val: float):
        """
        If manual override is ON and user changes the spin, write new_val into conversion_label.
        """
        if self.manual_check.isChecked():
            self.conversion_label.setText(f"{new_val:.4f}")

    def _recalculate_conversion(self):
        """
        Helper: read pixel_label and known_length_spin, compute conv = px / mm, update conversion_label.
        """
        try:
            px = float(self.pixel_label.text())
        except ValueError:
            px = 0.0

        mm = self.known_length_spin.value()
        if mm > 0 and px > 0:
            conv = px / mm
            self.conversion_label.setText(f"{conv:.4f}")
        else:
            self.conversion_label.setText("0.00")