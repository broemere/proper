from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from data_pipeline import DataPipeline
import json
import logging
log = logging.getLogger(__name__)

class ExportTab(QWidget):
    """
    A widget tab displaying a canvas for drawing lines and a table showing their lengths.
    """

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        #self.pipeline.register_observer("state_dict", self.save_state)

    def init_ui(self):
        """
        Initializes the user interface, splitting it into a canvas and a table area.
        """
        main_layout = QHBoxLayout(self)

        # --- Left Side: Canvas and Controls ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Control buttons
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addStretch()

        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.pipeline.get_state)
        ctrl_layout.addWidget(self.export_btn)
        ctrl_layout.addStretch()
        canvas_layout.addLayout(ctrl_layout)

        # The canvas container gets a stretch factor of 1 to take up available space.
        main_layout.addWidget(canvas_container, stretch=1)
