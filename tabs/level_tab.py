import numpy as np
from data_pipeline import DataPipeline
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy, QSlider, QFrame
from processing.data_transform import numpy_to_qpixmap
from widgets.adaptive_image import AutoResizeImage
from widgets.canvas_window import CanvasWindow
import logging
log = logging.getLogger(__name__)


class LevelTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.canvas_window1 = None  # Store references to the canvas windows to prevent garbage collecting
        self.canvas_window2 = None
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        tab_layout = QVBoxLayout(self)
        images_row = QHBoxLayout()

        # Image placeholder with the button enabled
        self.frame_label1 = AutoResizeImage("No data", show_button=True)
        self.frame_label1.setFrameShape(QFrame.Box)
        self.frame_label1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        images_row.addWidget(self.frame_label1, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        self.frame_label2 = AutoResizeImage("No data", show_button=True)
        self.frame_label2.setFrameShape(QFrame.Box)
        self.frame_label2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        images_row.addWidget(self.frame_label2, alignment=Qt.AlignHCenter | Qt.AlignVCenter)
        tab_layout.addLayout(images_row, stretch=1)

        # --- Control rows remain the same ---
        ctrl_row = QVBoxLayout()
        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Brightness"))
        self.b_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        row_b.addWidget(self.b_slider)
        self.b_value = QLabel("50")
        row_b.addWidget(self.b_value)
        ctrl_row.addLayout(row_b)
        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("Contrast"))
        self.c_slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        row_c.addWidget(self.c_slider)
        self.c_value = QLabel("50")
        row_c.addWidget(self.c_value)
        ctrl_row.addLayout(row_c)
        row_d = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        row_d.addWidget(self.reset_button)
        ctrl_row.addLayout(row_d)
        # auto_methods = ["Linear Clip", "Gamma Correction", "CLAHE", "Disk Opening", "Retinex"]
        # auto_layout = QHBoxLayout()
        # auto_layout.addWidget(QLabel('Auto Level Method:'))
        # self.cb_auto = QComboBox()
        # self.cb_auto.addItems(auto_methods)
        # self.cb_auto.setDisabled(True)
        # auto_layout.addWidget(self.cb_auto)
        # ctrl_row.addLayout(auto_layout)
        tab_layout.addLayout(ctrl_row)

    def connect_signals(self):
        """Connects all pipeline signals to the appropriate slots in this tab."""
        # --- Connections FROM UI TO PIPELINE (User Actions) ---
        self.b_slider.valueChanged.connect(self.pipeline.set_brightness)
        self.c_slider.valueChanged.connect(self.pipeline.set_contrast)
        self.reset_button.clicked.connect(self.pipeline.reset_levels)

        # --- Connections FROM PIPELINE TO UI (Data Updates) ---
        self.pipeline.leveled_images.connect(self._update_frames)
        self.pipeline.brightness_changed.connect(self.b_slider.setValue)
        self.pipeline.contrast_changed.connect(self.c_slider.setValue)
        self.pipeline.brightness_changed.connect(lambda v: self.b_value.setText(str(v)))
        self.pipeline.contrast_changed.connect(lambda v: self.c_value.setText(str(v)))

        # --- Local UI Connections (No Data Logic) ---
        self.frame_label1.button_clicked.connect(self._on_frame1_action)
        self.frame_label2.button_clicked.connect(self._on_frame2_action)

    def _on_frame1_action(self):
        """Launches the drawing window for the first frame."""
        log.info("Draw button on Frame 1 clicked.")
        background_pixmap = self.frame_label1.pixmap()
        if background_pixmap:
            # Create and show the new window, passing the current image
            self.canvas_window1 = CanvasWindow(background_pixmap, current_tool=self.pipeline.drawing_tool)
            self.canvas_window1.drawing_completed.connect(self._on_drawing1_completed)
            self.canvas_window1.tool_toggled.connect(self.pipeline.set_drawing_tool)
            self.canvas_window1.show()
        else:
            log.warning("Frame 1 has no image data to draw on.")

    def _on_frame2_action(self):
        """Launches the drawing window for the second frame."""
        log.info("Draw button on Frame 2 clicked.")
        background_pixmap = self.frame_label2.pixmap()
        if background_pixmap:
            self.canvas_window2 = CanvasWindow(background_pixmap, current_tool=self.pipeline.drawing_tool)
            self.canvas_window2.drawing_completed.connect(self._on_drawing2_completed)
            self.canvas_window2.tool_toggled.connect(self.pipeline.set_drawing_tool)
            self.canvas_window2.show()
        else:
            log.warning("Frame 2 has no image data to draw on.")

    def _on_drawing1_completed(self, image_array: np.ndarray):
        """Receives the drawing and tells the pipeline to update its data."""
        self.pipeline.update_level_blobs("left", image_array)

    def _on_drawing2_completed(self, image_array: np.ndarray):
        """Receives the drawing and tells the pipeline to update its data."""
        self.pipeline.update_level_blobs("right", image_array)

    def _update_frames(self, frames):
        """Receives NumPy arrays from the pipeline and updates the UI."""
        frame, frame_right = frames
        pixmap = numpy_to_qpixmap(frame)
        self.frame_label1.setPixmap(pixmap)
        if frame_right is not None:
            pixmap = numpy_to_qpixmap(frame_right)
            self.frame_label2.setPixmap(pixmap)
        else:
            self.frame_label2.image_label.clear()
            self.frame_label2._pixmap = None
