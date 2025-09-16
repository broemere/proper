from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDoubleSpinBox, QSizePolicy,
    QStyle,  QButtonGroup, QStackedWidget, QCheckBox
)
from PySide6.QtGui import QIcon
from data_pipeline import DataPipeline
import numpy as np
from widgets.scale_widget import ScaledLineCanvas
from processing.data_transform import numpy_to_qpixmap
import logging
log = logging.getLogger(__name__)


class ScaleTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self.connect_signals()

    # def showEvent(self, event):
    #     super().showEvent(event)
    #     # Refresh from pipeline if an image is already available
    #     current_image = self.pipeline.left_image
    #     if current_image is not None:
    #         self._show_scale_image(current_image)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # --- Canvas ---
        self.line_canvas = ScaledLineCanvas()
        self.line_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.line_canvas, stretch=1)

        # --- Controls ---
        ctrl_row = QHBoxLayout()
        ctrl_row.addStretch()
        self.refresh_btn = QPushButton(self.style().standardIcon(QStyle.SP_BrowserReload), "")
        self.refresh_btn.setToolTip("Reload original image and clear zoom/line.")
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
        ctrl_row.addWidget(self.zoom_btn)
        ctrl_row.addWidget(self.line_btn)
        self.undo_btn = QPushButton("Undo Line")
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

    def connect_signals(self):
        """Connect UI events to the pipeline (Model) and pipeline events to UI updates."""
        # --- Connections FROM UI TO PIPELINE (User Actions) ---
        self.known_length_spin.editingFinished.connect(self._on_known_length_finished)
        self.line_canvas.line_completed.connect(self.pipeline.set_pixel_length)
        self.manual_mode_check.toggled.connect(self.pipeline.set_scale_is_manual)
        self.manual_conversion_spin.editingFinished.connect(self._on_manual_factor_finished)

        # --- Connections FROM PIPELINE TO UI (Data Updates) ---
        self.pipeline.known_length_changed.connect(self.known_length_spin.setValue)
        self.pipeline.pixel_length_changed.connect(lambda l: self.pixel_label.setText(f"{l:.2f}"))
        self.pipeline.conversion_factor_changed.connect(lambda f: self.conversion_label.setText(f"{f:.4f}"))
        self.pipeline.scale_is_manual_changed.connect(self.manual_mode_check.setChecked)  # Updates checkbox
        self.pipeline.scale_is_manual_changed.connect(self._on_manual_mode_toggled)  # Updates other UIs
        self.pipeline.left_image_changed.connect(self._show_scale_image)
        self.pipeline.manual_conversion_factor_changed.connect(self.manual_conversion_spin.setValue)

        # --- Local UI Connections (No Data Logic) ---
        self.refresh_btn.clicked.connect(self._reload_base_image)
        self.undo_btn.clicked.connect(self.line_canvas.undo_last_line)
        self.line_canvas.mode_changed.connect(self._on_canvas_mode_changed)

    # --- UI Synchronization and State Management ---

    def _show_scale_image(self, img_array: np.ndarray):
        """Convert numpy array to QPixmap and set it as the canvas background."""
        pixmap = numpy_to_qpixmap(img_array)
        self.line_canvas.set_background(pixmap)

    def _reload_base_image(self):
        """Fetches the original image from the pipeline and resets the canvas."""
        if self.pipeline.left_image is not None:
            self._show_scale_image(self.pipeline.left_image)

    def _on_canvas_mode_changed(self, new_mode: str):
        """Keeps the mode buttons in sync with the canvas's internal mode."""
        is_zoom = new_mode == 'zoom'
        self.zoom_btn.setChecked(is_zoom)
        self.line_btn.setChecked(not is_zoom)

    def _on_manual_mode_toggled(self, is_checked: bool):
        """Handles switching between automatic and manual modes."""
        # 1. Enable/disable automatic controls
        self.known_length_spin.setEnabled(not is_checked)
        self.line_canvas.setEnabled(not is_checked)
        self.undo_btn.setEnabled(not is_checked)
        # 2. Swap between the label and the spin box
        self.conversion_stack.setCurrentIndex(1 if is_checked else 0)

    def _on_known_length_finished(self):
        # Get the final value from the spin box and update the pipeline
        self.pipeline.set_known_length(self.known_length_spin.value())

    def _on_manual_factor_finished(self):
        # Get the final value from the spin box and update the pipeline
        self.pipeline.set_manual_conversion_factor(self.manual_conversion_spin.value())
