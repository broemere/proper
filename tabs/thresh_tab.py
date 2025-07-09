from data_pipeline import DataPipeline
from PySide6.QtCore import Qt, Signal, QPointF, QRect, QSize, QTimer, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup, QSlider, QFrame
)
from PySide6.QtGui import QPalette, QPixmap, QColor, QPainter, QImage, QPen, QCursor, QIcon
import numpy as np
from processing.data_transform import numpy_to_qpixmap
from widgets.adaptive_image import AutoResizeImage


class ThreshTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self.pipeline.register_observer("threshed", self._update_frames)


    def init_ui(self):
        tab_layout = QVBoxLayout(self)
        images_row = QHBoxLayout()

        # 1) Image placeholder
        self.frame_label1 = AutoResizeImage("No data")
        self.frame_label1.setFrameShape(QFrame.Box)
        self.frame_label1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # allow width/height to shrink/grow
        images_row.addWidget(self.frame_label1, alignment=Qt.AlignHCenter | Qt.AlignVCenter)

        self.frame_label2 = AutoResizeImage("No data")
        self.frame_label2.setFrameShape(QFrame.Box)
        self.frame_label2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        images_row.addWidget(self.frame_label2, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        tab_layout.addLayout(images_row, stretch=1)

        ctrl_row = QVBoxLayout()
        # Thresh slider
        row_t = QHBoxLayout()
        row_t.addWidget(QLabel("Threshold"))
        self.t_slider = QSlider(Qt.Horizontal, minimum=0, maximum=255, value=127)
        row_t.addWidget(self.t_slider)
        self.t_value = QLabel("127")
        row_t.addWidget(self.t_value)
        ctrl_row.addLayout(row_t)

        # row_d = QHBoxLayout()
        # reset_button = QPushButton("Reset")
        # reset_button.clicked.connect(self._reset_thresh)
        # row_d.addWidget(reset_button)
        # ctrl_row.addLayout(row_d)

        auto_methods = ["Linear Clip", "Gamma Correction", "CLAHE", "Disk Opening", "Retinex"]
        auto_layout = QHBoxLayout()
        auto_layout.addWidget(QLabel('Auto Thresh Method:'))
        self.cb_auto = QComboBox()
        self.cb_auto.addItems(auto_methods)
        self.cb_auto.setDisabled(True)
        #self.cb_auto.setCurrentText(self.pipeline.zeroing_method)
        #self.cb_auto.currentTextChanged.connect(self._apply_zeroing)
        auto_layout.addWidget(self.cb_auto)
        #self.spin_zero_window = QSpinBox(minimum=0, maximum=999, value=7)
        #self.spin_zero_window.installEventFilter(self)
        #self.spin_zero_window.valueChanged.connect(self._apply_zeroing)
        #zero_layout.addWidget(QLabel('Window'))
        #zero_layout.addWidget(self.spin_zero_window)
        ctrl_row.addLayout(auto_layout)

        tab_layout.addLayout(ctrl_row)

        self.t_slider.valueChanged.connect(lambda v: self._on_thresh_change(self.t_value, "threshold", v))


    def _update_frames(self, frames):
        """
        Slot that receives the NumPy array from the pipeline and updates the UI.
        """
        frame, frame_right = frames
        pixmap = numpy_to_qpixmap(frame)
        self.frame_label1.setPixmap(pixmap)
        if frame_right is not None:
            pixmap = numpy_to_qpixmap(frame_right)
            self.frame_label2.setPixmap(pixmap)

    def _on_thresh_change(self, value_label: QLabel, attr: str, value: int):
        # show new slider value
        value_label.setText(str(value))
        # persist to pipeline
        setattr(self.pipeline, attr, value)
        self.pipeline.apply_thresh()

    def _reset_thresh(self):
        self.t_slider.setValue(127)