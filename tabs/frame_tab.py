from data_pipeline import DataPipeline
from PySide6.QtCore import Qt, Signal, QPointF, QRect, QSize, QTimer, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup, QSlider, QFrame
)
from PySide6.QtGui import QPalette, QPixmap
import numpy as np
from processing.data_transform import numpy_to_qpixmap
from widgets.adaptive_image import AutoResizeImage



class FrameTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        style = self.style()
        self._yes_pix = style.standardIcon(QStyle.SP_DialogYesButton).pixmap(16, 16)
        self._no_pix  = style.standardIcon(QStyle.SP_DialogNoButton).pixmap(16, 16)
        self.init_ui()
        self._has_been_shown = False  # A flag to prevent unnecessary reloads

        self.pipeline.register_observer("frame_count", self._update_frame_count)
        self.pipeline.register_observer("left_image", self._update_left_frame)
        self.pipeline.register_observer("right_image", self._update_right_frame)
        self.pipeline.register_observer("trimming", self._update_trim_range)

        self._left_debounce = QTimer(self, singleShot=True)
        self._left_debounce.setInterval(2000)
        self._left_debounce.timeout.connect(self._fire_left_index)

        self._right_debounce = QTimer(self, singleShot=True)
        self._right_debounce.setInterval(2000)
        self._right_debounce.timeout.connect(self._fire_right_index)


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
        # two-way link
        self.left_slider.valueChanged.connect(self.left_spin.setValue)
        self.left_spin.valueChanged.connect(self.left_slider.setValue)
        self.left_spin.valueChanged.connect(self._on_left_index_changed)
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
        self.left_goto_button.clicked.connect(self._goto_left)
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
        # two-way link
        self.right_slider.valueChanged.connect(self.right_spin.setValue)
        self.right_spin.valueChanged.connect(self.right_slider.setValue)
        self.right_spin.valueChanged.connect(self._on_right_index_changed)
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
        self.right_goto_button.clicked.connect(self._goto_right)
        self.right_goto = QDoubleSpinBox()
        self.right_goto.setDecimals(2)
        self.right_goto.setRange(0.0, 1e6)
        self.right_goto.setValue(getattr(self.pipeline, "final_pressure", 25.0))
        mmhg_label = QLabel("mmHg")
        goto_row.addStretch(1)  # pushes labels to the right
        goto_row.addWidget(self.right_goto_button)
        goto_row.addWidget(self.right_goto)
        goto_row.addWidget(mmhg_label)
        goto_row.addStretch(1)  # pushes labels to the right
        right_vbox.addLayout(goto_row)
        ctrl_row.addLayout(right_vbox)

        tab_layout.addLayout(ctrl_row)

    def _goto_left(self):
        p = self.left_goto.value()
        if self.pipeline.csv_path:
            idx = np.argmin(np.abs(np.subtract(self.pipeline.smoothed_data["p"], p)))
            self.left_spin.setValue(idx+self.pipeline.trim_start)

    def _goto_right(self):
        p = self.right_goto.value()
        if self.pipeline.csv_path:
            idx = np.argmin(np.abs(np.subtract(self.pipeline.smoothed_data["p"], p)))
            self.right_spin.setValue(idx+self.pipeline.trim_start)

    def _refresh_left_pressures(self, index: int):
        if self.pipeline.csv_path:
            new_index = index - self.pipeline.trim_start
            pre = self.pipeline.smoothed_data["p"][max(0, new_index - 3):new_index]
            p = self.pipeline.smoothed_data["p"][new_index]
            post = self.pipeline.smoothed_data["p"][new_index+1:new_index+4]
            self.left_value_pre.setText(", ".join(str(x) for x in pre))
            self.left_value_lbl.setText(str(p))
            self.left_value_post.setText(", ".join(str(x) for x in post))

    def _refresh_right_pressures(self, index: int):
        if self.pipeline.csv_path:
            new_index = index - self.pipeline.trim_start
            pre = self.pipeline.smoothed_data["p"][max(0, new_index - 3):new_index]
            p = self.pipeline.smoothed_data["p"][new_index]
            post = self.pipeline.smoothed_data["p"][new_index+1:new_index+4]
            self.right_value_pre.setText(", ".join(str(x) for x in pre))
            self.right_value_lbl.setText(str(p))
            self.right_value_post.setText(", ".join(str(x) for x in post))

    def _on_left_index_changed(self, new_val: int):
        self.left_status_icon.setPixmap(self._no_pix)
        #self.pipeline.left_index_changed(new_val)
        self._refresh_left_pressures(new_val)
        self._left_pending = new_val
        self._left_debounce.start()
        #print(new_val)

    def _on_right_index_changed(self, new_val: int):
        self.right_status_icon.setPixmap(self._no_pix)
        #self.pipeline.left_index_changed(new_val)
        self._refresh_right_pressures(new_val)
        self._right_pending = new_val
        self._right_debounce.start()
        #print(new_val)

    def _fire_left_index(self):
        # now that 2 s passed with no new moves:
        self.left_status_icon.setPixmap(self._yes_pix)
        self.pipeline.load_left_frame(self._left_pending)

    def _fire_right_index(self):
        # now that 2 s passed with no new moves:
        self.right_status_icon.setPixmap(self._yes_pix)
        self.pipeline.load_right_frame(self._right_pending)

    def _update_frame_count(self, frame_count):
        frame_index = frame_count - 1
        if self.pipeline.trim_stop != 42:
            frame_index = self.pipeline.trim_stop
        for slider, spin in (
            (self.left_slider,  self.left_spin),
            (self.right_slider, self.right_spin)
        ):
            slider.setMaximum(frame_index)
            spin.setMaximum(frame_index)
            # clamp the current value if needed:
            if spin.value() > frame_index:
                spin.setValue(frame_index)
        self._refresh_left_pressures(self.left_slider.value())
        self._refresh_right_pressures(self.right_slider.value())

    def _update_trim_range(self, vals):
        start, stop = vals
        if stop > self.pipeline.frame_count - 1:
            stop = max(1, self.pipeline.frame_count - 1)
        for slider, spin in (
            (self.left_slider,  self.left_spin),
            (self.right_slider, self.right_spin)
        ):
            slider.setMinimum(start)
            spin.setMinimum(start)
            slider.setMaximum(stop)
            spin.setMaximum(stop)
            # clamp the current value if needed:
            if spin.value() < start:
                spin.setValue(start)
            if spin.value() > stop:
                spin.setValue(stop)

        self._refresh_left_pressures(self.left_slider.value())
        self._refresh_right_pressures(self.right_slider.value())

    def _update_left_frame(self, frame: np.ndarray):
        """
        Slot that receives the NumPy array from the pipeline and updates the UI.
        """
        pixmap = numpy_to_qpixmap(frame)
        self.frame_label1.setPixmap(pixmap)

    def _update_right_frame(self, frame: np.ndarray):
        """
        Slot that receives the NumPy array from the pipeline and updates the UI.
        """
        pixmap = numpy_to_qpixmap(frame)
        self.frame_label2.setPixmap(pixmap)
