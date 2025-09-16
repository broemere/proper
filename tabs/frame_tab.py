from data_pipeline import DataPipeline
from PySide6.QtCore import Qt, Signal, QPointF, QRect, QSize, QTimer, Slot, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup, QSlider, QFrame
)
from PySide6.QtGui import QPalette, QPixmap
import numpy as np
from processing.data_transform import numpy_to_qpixmap
from widgets.adaptive_image import AutoResizeImage
import logging
log = logging.getLogger(__name__)

class FrameTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        style = self.style()
        self._yes_pix = style.standardIcon(QStyle.SP_DialogYesButton).pixmap(16, 16)
        self._no_pix  = style.standardIcon(QStyle.SP_DialogNoButton).pixmap(16, 16)
        self._is_state_synced = False
        self.init_ui()
        self.connect_signals()

        self._currently_displayed_left_frame_idx = -1
        self._currently_displayed_right_frame_idx = -1

        self._left_debounce = QTimer(self, singleShot=True)
        self._left_debounce.setInterval(2000)
        self._left_debounce.timeout.connect(self._fire_left_index)
        self._right_debounce = QTimer(self, singleShot=True)
        self._right_debounce.setInterval(2000)
        self._right_debounce.timeout.connect(self._fire_right_index)

    def showEvent(self, event: QEvent):
        """This Qt event fires every time the widget is shown."""
        super().showEvent(event)
        if self.isVisible() and not self._is_state_synced:
            self._sync_ui_to_pipeline()

    def init_ui(self):
        tab_layout = QVBoxLayout(self)
        images_row = QHBoxLayout()

        # 1) Image placeholder
        self.frame_label1 = AutoResizeImage("No data")
        self.frame_label1.setFrameShape(QFrame.Box)
        self.frame_label1.setSizePolicy(
            QSizePolicy.Ignored,  # allow width to shrink/grow
            QSizePolicy.Ignored  # allow height to shrink/grow
        )
        images_row.addWidget(self.frame_label1, alignment=Qt.AlignHCenter | Qt.AlignVCenter)

        self.frame_label2 = AutoResizeImage("No data")
        self.frame_label2.setFrameShape(QFrame.Box)
        self.frame_label2.setSizePolicy(
            QSizePolicy.Ignored,  # allow width to shrink/grow
            QSizePolicy.Ignored  # allow height to shrink/grow
        )
        images_row.addWidget(self.frame_label2, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        tab_layout.addLayout(images_row, stretch=1)

        ctrl_row = QHBoxLayout()
        # ─── LEFT PANEL ───────────────────────────────────
        left_vbox = QVBoxLayout()
        # 2) Slider + spin-box, two-way linked
        self.left_slider = QSlider(Qt.Horizontal)
        self.left_slider.setRange(0, 1)   # start with [0..1], will update later
        self.left_spin   = QSpinBox()
        self.left_spin.setRange(0, 1)
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.left_slider, 1)
        slider_row.addWidget(self.left_spin, 0)
        left_vbox.addLayout(slider_row)
        # 3) Title + value label
        title_lbl = QLabel("Pressure [mmHg]:")
        self.left_value_pre = QLabel("")
        self.left_value_post = QLabel("")
        self.left_value_lbl = QLabel("0.00")
        self.left_value_lbl.setStyleSheet("font-weight: bold;")
        self.left_status_icon = QLabel()
        self.left_status_icon.setPixmap(self._yes_pix)   # start in “ready” state
        self.left_status_icon.setAlignment(Qt.AlignCenter)
        self.left_status_icon.setToolTip("Buffer before sending frame data request. Red=Paused, Green=Request Sent")
        #self.left_value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_row = QHBoxLayout()
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)  # pushes labels to the right
        title_row.addWidget(self.left_value_pre)
        title_row.addWidget(self.left_value_lbl)
        title_row.addWidget(self.left_value_post)
        title_row.addStretch(1)  # pushes labels to the right
        title_row.addWidget(self.left_status_icon)  # built-in icon on the left
        left_vbox.addLayout(title_row)
        goto_row = QHBoxLayout()
        self.left_goto_button = QPushButton("Go to")
        self.left_goto = QDoubleSpinBox()
        self.left_goto.setDecimals(2)
        self.left_goto.setRange(0.0, 1e6)
        self.left_goto.setValue(getattr(self.pipeline, "initial_pressure", 0.0))
        mmhg_label = QLabel("mmHg")
        goto_row.addStretch(1)  # pushes labels to the right
        goto_row.addWidget(self.left_goto_button)
        goto_row.addWidget(self.left_goto)
        goto_row.addWidget(mmhg_label)
        goto_row.addStretch(1)  # pushes labels to the right
        left_vbox.addLayout(goto_row)
        ctrl_row.addLayout(left_vbox)
        # ─── RIGHT PANEL ───────────────────────────────────
        right_vbox = QVBoxLayout()
        # 2) Slider + spin-box, two-way linked
        self.right_slider = QSlider(Qt.Horizontal)
        self.right_slider.setRange(0, 1)   # start with [0..1], will update later
        self.right_spin   = QSpinBox()
        self.right_spin.setRange(0, 1)
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.right_slider, 1)
        slider_row.addWidget(self.right_spin, 0)
        right_vbox.addLayout(slider_row)
        # 3) Title + value label
        title_lbl = QLabel("Pressure [mmHg]:")
        self.right_value_pre = QLabel("")
        self.right_value_post = QLabel("")
        self.right_value_lbl = QLabel("0.00")
        self.right_value_lbl.setStyleSheet("font-weight: bold;")
        self.right_status_icon = QLabel()
        self.right_status_icon.setPixmap(self._yes_pix)   # start in “ready” state
        self.right_status_icon.setAlignment(Qt.AlignCenter)
        self.right_status_icon.setToolTip("Buffer before sending frame data request. Red=Paused, Green=Request Sent")
        #self.left_value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_row = QHBoxLayout()
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)  # pushes labels to the right
        title_row.addWidget(self.right_value_pre)
        title_row.addWidget(self.right_value_lbl)
        title_row.addWidget(self.right_value_post)
        title_row.addStretch(1)  # pushes labels to the right
        title_row.addWidget(self.right_status_icon)  # built-in icon on the left
        right_vbox.addLayout(title_row)
        goto_row = QHBoxLayout()
        self.right_goto_button = QPushButton("Go to")
        self.right_goto = QDoubleSpinBox()
        self.right_goto.setDecimals(2)
        self.right_goto.setRange(0.0, 1e6)
        #self.right_goto.setValue(getattr(self.pipeline, "final_pressure", 25.0))
        mmhg_label = QLabel("mmHg")
        goto_row.addStretch(1)  # pushes labels to the right
        goto_row.addWidget(self.right_goto_button)
        goto_row.addWidget(self.right_goto)
        goto_row.addWidget(mmhg_label)
        goto_row.addStretch(1)  # pushes labels to the right
        right_vbox.addLayout(goto_row)
        ctrl_row.addLayout(right_vbox)
        tab_layout.addLayout(ctrl_row)

    def connect_signals(self):
        """Connects all pipeline signals to the appropriate slots in this tab."""
        # --- Connections FROM UI TO PIPELINE (User Actions) ---
        self.left_spin.valueChanged.connect(self._on_left_index_changed)
        self.right_spin.valueChanged.connect(self._on_right_index_changed)
        self.right_goto.editingFinished.connect(self._on_final_pressure_changed)

        # --- Connections FROM PIPELINE TO UI (Data Updates) ---
        self.pipeline.state_loaded.connect(self._on_state_loaded)
        self.pipeline.left_image_changed.connect(self._update_left_frame)
        self.pipeline.right_image_changed.connect(self._update_right_frame)
        self.pipeline.left_keypoint_changed.connect(self.on_left_keypoint_updated)
        self.pipeline.right_keypoint_changed.connect(self.on_right_keypoint_updated)
        self.pipeline.final_pressure_changed.connect(self.right_goto.setValue)

        # --- Local UI Connections (No Data Logic) ---
        self.left_goto_button.clicked.connect(self._goto_left)
        self.right_goto_button.clicked.connect(self._goto_right)
        self.left_slider.valueChanged.connect(self.left_spin.setValue)
        self.left_spin.valueChanged.connect(self.left_slider.setValue)
        self.right_slider.valueChanged.connect(self.right_spin.setValue)
        self.right_spin.valueChanged.connect(self.right_slider.setValue)


    @Slot()
    def _on_state_loaded(self, _):
        """Slot for the 'state_loaded' signal. Marks the UI as dirty."""
        log.info("Received 'state_loaded' notification.")
        self._is_state_synced = False
        # Sync immediately if visible, otherwise, showEvent will handle it when the user clicks the tab.
        if self.isVisible():
            self._sync_ui_to_pipeline()

        # perform the "lazy load" check.
        # If the frame we are showing is out of sync with the pipeline's state,
        # command the pipeline to load the correct one.
        if self._currently_displayed_left_frame_idx != self.pipeline.left_index:
            log.info("Left frame is stale. Requesting updated frame from pipeline.")
            self.pipeline.set_left_keypoint(self.pipeline.left_index, load_frame=True)

        if self._currently_displayed_right_frame_idx != self.pipeline.right_index:
            log.info("Right frame is stale. Requesting updated frame from pipeline.")
            self.pipeline.set_right_keypoint(self.pipeline.right_index, load_frame=True)

    def _sync_ui_to_pipeline(self):
        """
        Pulls the complete state from the pipeline and updates all UI controls at once.
        This is the single source of truth for synchronizing the FrameTab UI.
        """
        log.info("Synchronizing entire UI to pipeline state.")

        widgets_to_block = [
            self.left_slider, self.left_spin,
            self.right_slider, self.right_spin,
            self.left_goto, self.right_goto
        ]
        for widget in widgets_to_block:
            widget.blockSignals(True)

        try:
            # 1. Get the latest trim and keypoint values from the pipeline
            start, stop = self.pipeline.trim_start, self.pipeline.trim_stop
            left_idx, right_idx = self.pipeline.left_index, self.pipeline.right_index

            # 2. Update slider/spinbox RANGES
            for slider, spin in ((self.left_slider, self.left_spin), (self.right_slider, self.right_spin)):
                slider.setRange(start, stop)
                spin.setRange(start, stop)

            # 3. Update slider/spinbox VALUES
            self.left_spin.setValue(left_idx)
            self.right_spin.setValue(right_idx)
            self.left_slider.setValue(left_idx)
            self.right_slider.setValue(right_idx)

            # 4. Update "Go To" pressure boxes
            self.left_goto.setValue(self.pipeline.initial_pressure)
            self.right_goto.setValue(self.pipeline.final_pressure)

            # 5. Refresh pressure labels with the correct indices
            self._refresh_left_pressures(left_idx)
            self._refresh_right_pressures(right_idx)

            self._is_state_synced = True

        finally:
            # ALWAYS unblock signals
            for widget in widgets_to_block:
                widget.blockSignals(False)

    def _goto_left(self):
        """Handles the 'Go to' button click for the left panel."""
        target_pressure = self.left_goto.value()
        # Tell the pipeline to find the frame for this pressure. The pipeline does all the work.
        self.pipeline.find_and_set_keypoint_by_pressure('left', target_pressure)
        self._refresh_left_pressures(self.pipeline.left_index)

    def _goto_right(self):
        """Handles the 'Go to' button click for the right panel."""
        target_pressure = self.right_goto.value()
        self.pipeline.find_and_set_keypoint_by_pressure('right', target_pressure)
        self._refresh_right_pressures(self.pipeline.right_index)

    def _refresh_left_pressures(self, index: int):
        # The View asks the Model for the data it needs, already processed.
        pressure_data = self.pipeline.get_pressure_display_data(index)
        self.left_value_pre.setText(pressure_data["pre"])
        self.left_value_lbl.setText(pressure_data["current"])
        self.left_value_post.setText(pressure_data["post"])

    def _refresh_right_pressures(self, index: int):
        pressure_data = self.pipeline.get_pressure_display_data(index)
        self.right_value_pre.setText(pressure_data["pre"])
        self.right_value_lbl.setText(pressure_data["current"])
        self.right_value_post.setText(pressure_data["post"])

    def _on_left_index_changed(self, new_val: int):
        log.info(f"left_index changed {new_val}")
        self.left_status_icon.setPixmap(self._no_pix)
        self._refresh_left_pressures(new_val)
        self._left_pending = new_val
        self._left_debounce.start()

    def _on_right_index_changed(self, new_val: int):
        log.info(f"right_index changed {new_val}")
        self.right_status_icon.setPixmap(self._no_pix)
        self._refresh_right_pressures(new_val)
        self._right_pending = new_val
        self._right_debounce.start()

    def _fire_left_index(self):
        # now that 2 s passed with no new moves:
        self.left_status_icon.setPixmap(self._yes_pix)
        self.pipeline.set_left_keypoint(self._left_pending, load_frame=True)

    def _fire_right_index(self):
        # now that 2 s passed with no new moves:
        self.right_status_icon.setPixmap(self._yes_pix)
        self.pipeline.set_right_keypoint(self._right_pending, load_frame=True)

    def _update_left_frame(self, frame: np.ndarray):
        """
        Slot that receives the NumPy array from the pipeline and updates the UI.
        """
        log.info("Updating left frame")
        pixmap = numpy_to_qpixmap(frame)
        self.frame_label1.setPixmap(pixmap)

    def _update_right_frame(self, frame: np.ndarray):
        """
        Slot that receives the NumPy array from the pipeline and updates the UI.
        """
        log.info("Updating right frame")
        pixmap = numpy_to_qpixmap(frame)
        self.frame_label2.setPixmap(pixmap)

    @Slot(int)
    def on_left_keypoint_updated(self, new_index):
        # This slot updates the UI (e.g., a spinbox) to reflect the new index
        # but does NOT trigger a frame load.
        self.left_spin.blockSignals(True)
        self.left_slider.blockSignals(True)
        self.left_spin.setValue(new_index)
        self.left_slider.setValue(new_index)
        self.left_spin.blockSignals(False)
        self.left_slider.blockSignals(False)

    @Slot(int)
    def on_right_keypoint_updated(self, new_index):
        # This slot updates the UI (e.g., a spinbox) to reflect the new index
        # but does NOT trigger a frame load.
        self.right_spin.blockSignals(True)
        self.right_slider.blockSignals(True)
        self.right_spin.setValue(new_index)
        self.right_slider.setValue(new_index)
        self.right_spin.blockSignals(False)
        self.right_slider.blockSignals(False)

    def _on_final_pressure_changed(self):
        """Sends the user-edited final pressure to the pipeline."""
        self.pipeline.set_final_pressure(self.right_goto.value())
