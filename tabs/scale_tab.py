from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QSizePolicy,
    QStyle,  QButtonGroup, QStackedWidget, QCheckBox
)
from PySide6.QtGui import  QPixmap, QImage, QIcon
from data_pipeline import DataPipeline
import numpy as np
from widgets.scale_widget import ScaledLineCanvas


class ScaleTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.pixel_length = 0.0  # Store pixel length internally
        self.init_ui()
        self._has_been_shown = False  # A flag to prevent unnecessary reloads
        self.pipeline.register_observer("scale_image", self._show_scale_image)

    def showEvent(self, event):
        super().showEvent(event)
        # Refresh from pipeline if an image is already available
        current_image = self.pipeline.scale_image
        if current_image is not None:
            self._show_scale_image(current_image)
            self._sync_ui_from_pipeline()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- Canvas ---
        self.line_canvas = ScaledLineCanvas()
        self.line_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.line_canvas.line_completed.connect(self._on_line_completed)
        self.line_canvas.mode_changed.connect(self._on_canvas_mode_changed)
        layout.addWidget(self.line_canvas, stretch=1)

        # --- Controls ---
        ctrl_row = QHBoxLayout()
        ctrl_row.addStretch()
        self.refresh_btn = QPushButton(self.style().standardIcon(QStyle.SP_BrowserReload), "")
        self.refresh_btn.setToolTip("Reload original image and clear zoom/line.")
        self.refresh_btn.clicked.connect(self._reload_base_image)
        ctrl_row.addWidget(self.refresh_btn)

        self.zoom_btn = QPushButton(QIcon("resources/zoom.png"), "")
        self.zoom_btn.setToolTip("Zoom Mode")
        self.zoom_btn.setCheckable(True)

        self.line_btn = QPushButton(QIcon("resources/scale.png"), "")
        self.line_btn.setToolTip("Line Drawing Mode")
        self.line_btn.setCheckable(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.zoom_btn)
        self.mode_group.addButton(self.line_btn)
        self.zoom_btn.toggled.connect(lambda c: c and self.line_canvas.set_mode('zoom'))
        self.line_btn.toggled.connect(lambda c: c and self.line_canvas.set_mode('line'))

        ctrl_row.addWidget(self.zoom_btn)
        ctrl_row.addWidget(self.line_btn)

        self.undo_btn = QPushButton("Undo Line")
        self.undo_btn.clicked.connect(self.line_canvas.undo_last_line)
        ctrl_row.addWidget(self.undo_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # --- Scale Details ---
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Known Length [mm]:"))
        self.known_length_spin = QDoubleSpinBox()
        self.known_length_spin.setDecimals(3)
        self.known_length_spin.setRange(0.0, 1e6)
        self.known_length_spin.setSingleStep(0.1)
        self.known_length_spin.valueChanged.connect(self._on_known_length_changed)
        scale_row.addWidget(self.known_length_spin)

        scale_row.addWidget(QLabel("Line Length [px]:"))
        self.pixel_label = QLabel("0.00")
        scale_row.addWidget(self.pixel_label)
        scale_row.addStretch()

        # --- Manual Mode and Conversion Factor Widgets ---
        scale_row.addWidget(QLabel("<b>Scale [px/mm]:</b>"))

        # Use a QStackedWidget to easily swap between the label and spinbox
        self.conversion_stack = QStackedWidget()
        self.conversion_stack.setFixedWidth(100) # Give it a fixed width for stability

        # Widget 0: The Label for auto mode
        self.conversion_label = QLabel("0.0000")
        self.conversion_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.conversion_stack.addWidget(self.conversion_label)

        # Widget 1: The SpinBox for manual mode
        self.manual_conversion_spin = QDoubleSpinBox()
        self.manual_conversion_spin.setDecimals(4)
        self.manual_conversion_spin.setRange(0.0, 1e6)
        self.manual_conversion_spin.setSingleStep(0.1)
        self.conversion_stack.addWidget(self.manual_conversion_spin)

        scale_row.addWidget(self.conversion_stack)

        self.manual_mode_check = QCheckBox("Set manually")
        scale_row.addWidget(self.manual_mode_check)
        layout.addLayout(scale_row)

        # --- Connect Signals ---
        self.line_canvas.line_completed.connect(self._on_line_completed)
        self.line_canvas.mode_changed.connect(self._on_canvas_mode_changed)
        self.refresh_btn.clicked.connect(self._reload_base_image)
        self.zoom_btn.toggled.connect(lambda c: c and self.line_canvas.set_mode('zoom'))
        self.line_btn.toggled.connect(lambda c: c and self.line_canvas.set_mode('line'))
        self.undo_btn.clicked.connect(self.line_canvas.undo_last_line)
        self.known_length_spin.valueChanged.connect(self._on_known_length_changed)

        # NEW: Connect signals for manual mode
        self.manual_mode_check.toggled.connect(self._on_manual_mode_toggled)
        self.manual_conversion_spin.valueChanged.connect(self._on_manual_conversion_changed)

        # --- Finalize ---
        self._sync_ui_from_pipeline()
        self._on_manual_mode_toggled(False) # Ensure initial state is "auto"

    # --- UI Synchronization and State Management ---

    def _sync_ui_from_pipeline(self):
        """Update UI controls with values from the data pipeline."""
        self.known_length_spin.setValue(getattr(self.pipeline, "known_length", 0.0))
        self._on_canvas_mode_changed(self.line_canvas.mode)
        self._update_conversion_factor()

    def _show_scale_image(self, img_array: np.ndarray):
        """Convert numpy array to QPixmap and set it as the canvas background."""
        h, w = img_array.shape[:2]
        qimg = QImage(img_array.data, w, h, w, QImage.Format_Grayscale8)
        self.line_canvas.set_background(QPixmap.fromImage(qimg))

    def _reload_base_image(self):
        """Fetches the original image from the pipeline and resets the canvas."""
        if self.pipeline.scale_image is not None:
            self._show_scale_image(self.pipeline.scale_image)

    # --- Central Calculation Logic ---

    def _update_conversion_factor(self):
        """
        Central method to calculate, display, and store the conversion factor.
        This is the single source of truth for the scale calculation.
        """
        if self.manual_mode_check.isChecked():
            return
        known_length = self.known_length_spin.value()
        new_factor = 0.0

        if known_length > 0 and self.pixel_length > 0:
            new_factor = self.pixel_length / known_length

        self.conversion_label.setText(f"{new_factor:.4f}")
        self.pipeline.set_conversion_factor(new_factor)

    # --- SLOTS ---

    def _on_manual_mode_toggled(self, is_checked: bool):
        """Handles switching between automatic and manual modes."""
        # 1. Enable/disable automatic controls
        self.known_length_spin.setEnabled(not is_checked)
        self.line_canvas.setEnabled(not is_checked)
        self.undo_btn.setEnabled(not is_checked)

        # 2. Swap between the label and the spin box
        self.conversion_stack.setCurrentIndex(1 if is_checked else 0)

        # 3. Update state based on the new mode
        if is_checked:
            # ENTERING manual mode
            try:
                # Get last auto-calculated value from the label
                current_factor = float(self.conversion_label.text())
            except ValueError:
                current_factor = 0.0
            # Sync the spin box and update the pipeline
            self.manual_conversion_spin.setValue(current_factor)
            self.pipeline.set_conversion_factor(current_factor)
        else:
            # LEAVING manual mode
            self._update_conversion_factor()

    def _on_manual_conversion_changed(self, new_value: float):
        """Handles a change in the manual conversion spin box."""
        self.pipeline.set_conversion_factor(new_value)

    def _on_canvas_mode_changed(self, new_mode: str):
        """Keeps the mode buttons in sync with the canvas's internal mode."""
        is_zoom = new_mode == 'zoom'
        self.zoom_btn.setChecked(is_zoom)
        self.line_btn.setChecked(not is_zoom)

    def _on_line_completed(self, length_px: float):
        """Handles new line length from the canvas."""
        if self.manual_mode_check.isChecked(): return
        self.pixel_length = length_px
        self.pixel_label.setText(f"{length_px:.2f}")
        self._update_conversion_factor()

    def _on_known_length_changed(self, new_length: float):
        """Handles user input for the known physical length."""
        if self.manual_mode_check.isChecked(): return
        self.pipeline.set_known_length(new_length)
        self._update_conversion_factor()