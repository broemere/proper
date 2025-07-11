from data_pipeline import DataPipeline
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSizePolicy, QSlider, QFrame, \
    QPushButton
from processing.data_transform import numpy_to_qpixmap
from widgets.adaptive_image import AutoResizeImage
from widgets.canvas_window import CanvasWindow
import numpy as np


class ThreshTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.canvas_window1 = None
        self.canvas_window2 = None
        self.init_ui()
        self.pipeline.register_observer("threshed", self._update_frames)

    def init_ui(self):
        tab_layout = QVBoxLayout(self)
        images_row = QHBoxLayout()

        self.frame_label1 = AutoResizeImage("No data", show_button=True)
        self.frame_label1.setFrameShape(QFrame.Box)
        self.frame_label1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # allow width/height to shrink/grow
        images_row.addWidget(self.frame_label1, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        self.frame_label1.button_clicked.connect(self._on_frame1_action)

        self.frame_label2 = AutoResizeImage("No data", show_button=True)
        self.frame_label2.setFrameShape(QFrame.Box)
        self.frame_label2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        images_row.addWidget(self.frame_label2, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        self.frame_label2.button_clicked.connect(self._on_frame2_action)
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

        row_d = QHBoxLayout()
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_thresh)
        row_d.addWidget(reset_button)
        ctrl_row.addLayout(row_d)

        auto_methods = ["Linear Clip", "Gamma Correction", "CLAHE", "Disk Opening", "Retinex"]
        auto_layout = QHBoxLayout()
        auto_layout.addWidget(QLabel('Auto Thresh Method:'))
        self.cb_auto = QComboBox()
        self.cb_auto.addItems(auto_methods)
        self.cb_auto.setDisabled(True)
        auto_layout.addWidget(self.cb_auto)
        ctrl_row.addLayout(auto_layout)

        tab_layout.addLayout(ctrl_row)

        self.t_slider.valueChanged.connect(lambda v: self._on_thresh_change(self.t_value, "threshold", v))

    def _on_frame1_action(self):
        """Launches the drawing window for the first frame."""
        print("Draw button on Frame 1 clicked.")
        background_pixmap = self.frame_label1.pixmap()
        if background_pixmap:
            # Create and show the new window, passing the current image
            self.canvas_window1 = CanvasWindow(background_pixmap)
            self.canvas_window1.drawing_completed.connect(self._on_drawing1_completed)
            self.canvas_window1.show()
        else:
            print("Frame 1 has no image data to draw on.")

    def _on_frame2_action(self):
        """Launches the drawing window for the second frame."""
        print("Draw button on Frame 2 clicked.")
        background_pixmap = self.frame_label2.pixmap()
        if background_pixmap:
            self.canvas_window2 = CanvasWindow(background_pixmap)
            self.canvas_window2.drawing_completed.connect(self._on_drawing2_completed)
            self.canvas_window2.show()
        else:
            print("Frame 2 has no image data to draw on.")

    def _on_drawing1_completed(self, image_array: np.ndarray):
        """Receives the final drawing from the canvas window for frame 1."""
        print("Drawing for frame 1 completed. Updating pipeline.")
        if self.pipeline.left_thresh_blobs is not None:
            mask = (image_array != 127)
            self.pipeline.left_thresh_blobs[mask] = image_array[mask]
        else:
            self.pipeline.left_thresh_blobs = image_array
        self.pipeline.apply_thresh()
        self.pipeline.left_threshed_old = self.pipeline.left_threshed.copy()
        self.pipeline.segment_image(self.pipeline.left_threshed, "left")

    def _on_drawing2_completed(self, image_array: np.ndarray):
        """Receives the final drawing from the canvas window for frame 2."""
        print("Drawing for frame 2 completed. Updating pipeline.")
        if self.pipeline.right_thresh_blobs is not None:
            mask = (image_array != 127)
            self.pipeline.right_thresh_blobs[mask] = image_array[mask]
        else:
            self.pipeline.right_thresh_blobs = image_array
        self.pipeline.apply_thresh()
        self.pipeline.right_threshed_old = self.pipeline.right_threshed.copy()
        self.pipeline.segment_image(self.pipeline.right_threshed, "right")

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
        else:
            self.frame_label2.image_label.clear()
            self.frame_label2._pixmap = None

    def _on_thresh_change(self, value_label: QLabel, attr: str, value: int):
        # show new slider value
        value_label.setText(str(value))
        # persist to pipeline
        setattr(self.pipeline, attr, value)
        self.pipeline.apply_thresh()

    def _reset_thresh(self):
        self.t_slider.setValue(127)
        self.pipeline.left_thresh_blobs = None
        self.pipeline.right_thresh_blobs = None