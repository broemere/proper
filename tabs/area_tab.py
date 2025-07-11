from PySide6.QtCore import Qt, Signal, QPointF, QRect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup
)
from PySide6.QtGui import QPalette, QPixmap, QColor, QPainter, QImage, QPen, QCursor, QIcon
from data_pipeline import DataPipeline
import numpy as np
from processing.resource_loader import load_cursor
from widgets.area_widget import AreaAnalysisWidget

from processing.data_transform import numpy_to_qpixmap


class AreaTab(QWidget):

    def __init__(self, pipeline: DataPipeline, left_right=None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.segmented_result = None
        self._has_been_shown = False
        self.pipeline.left_cleaned_old = None
        self.init_ui()
        #self.pipeline.register_observer("segmented", self.show_result)
        self.pipeline.register_observer("visualization_ready", self.on_visualization_ready)

    def init_ui(self):
        self.analysis_widget = AreaAnalysisWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.analysis_widget)

    def showEvent(self, event):
        super().showEvent(event)
        # Assuming self.pipeline.left_cleaned is the numpy array from the polygon tool

        current_image = self.pipeline.left_cleaned
        if current_image is None:
            # Nothing to process if there's no image
            return

        # Determine if we need to run the segmentation task.
        # This is True if the backup doesn't exist OR if the current image is different from the backup.
        should_run_task = (self.pipeline.left_cleaned_old is None or
                           not np.array_equal(current_image, self.pipeline.left_cleaned_old))

        if should_run_task:
            # If a run is needed, update the backup copy and queue the task.
            self.pipeline.left_cleaned_old = current_image.copy()
            self.pipeline.segment_image(current_image)

    def on_visualization_ready(self, data: dict):
        """Receives the final data dictionary and updates the analysis widget."""

        labels_array = data['labels']
        visual_array = data['visual']

        # Convert the color numpy array to a QPixmap for display
        visual_pixmap = numpy_to_qpixmap(visual_array)

        # Pass the colorized pixmap (for display) and original labels (for analysis)
        self.analysis_widget.set_data(visual_pixmap, labels_array)