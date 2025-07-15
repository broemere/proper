from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QSizePolicy, QStyle, QTableWidget, QTableWidgetItem, QScrollArea)
from data_pipeline import DataPipeline
import numpy as np
from widgets.thickness_widget import ThicknessCanvas
from processing.data_transform import numpy_to_qpixmap
import logging
log = logging.getLogger(__name__)

class ThicknessTab(QWidget):
    """
    A widget tab displaying a canvas for drawing lines and a table showing their lengths.
    """

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._has_been_shown = False  # A flag to prevent unnecessary reloads
        self.stored_lines = []
        self.init_ui()

        self.setMouseTracking(True)

        self.pipeline.register_observer("right_image", self._show_thickness_image)
        self.pipeline.register_observer("conversion_factor", self.refresh_table)


    def showEvent(self, event):
        """
        Overrides the show event to load the initial image if it's available.
        """
        super().showEvent(event)
        current_image = self.pipeline.right_image
        if current_image is not None and not self._has_been_shown:
            self._has_been_shown = True
            self._show_thickness_image(current_image)

    def init_ui(self):
        """
        Initializes the user interface, splitting it into a canvas and a table area.
        """
        main_layout = QHBoxLayout(self)

        # --- Left Side: Canvas and Controls ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Canvas for drawing
        self.line_canvas = ThicknessCanvas()
        self.line_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.line_canvas.lines_updated.connect(self._on_lines_updated)
        canvas_layout.addWidget(self.line_canvas)

        # Control buttons
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addStretch()
        self.refresh_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "")
        self.refresh_btn.setToolTip("Reload original image and clear lines.")
        self.refresh_btn.clicked.connect(self._reload_base_image)
        ctrl_layout.addWidget(self.refresh_btn)

        self.undo_btn = QPushButton("Undo Line")
        self.undo_btn.clicked.connect(self.line_canvas.undo_last_line)
        ctrl_layout.addWidget(self.undo_btn)
        ctrl_layout.addStretch()
        canvas_layout.addLayout(ctrl_layout)

        # The canvas container gets a stretch factor of 1 to take up available space.
        main_layout.addWidget(canvas_container, stretch=1)

        # --- Right Side: Scrollable Thickness Table ---
        self.table_container = QWidget()
        table_layout = QVBoxLayout(self.table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(1, 1)
        self.table.setHorizontalHeaderLabels(["Thickness [mm]"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        # Disable the table's own scrollbar; the QScrollArea will provide it.
        #self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table_layout.addWidget(self.table)

        # Create the scroll area and put the table's container inside.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.table_container)
        self.scroll_area.setWidgetResizable(True)

        # Add the scroll area (containing the table) to the main layout.
        main_layout.addWidget(self.scroll_area, stretch=0)

        self._update_table_size()  # Set initial size

    def _update_table_size(self):
        """
        Calculates and sets the fixed width of the scroll area.
        Height is now handled automatically by the scroll area.
        """
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

        # Calculate required width for the scroll area
        v_header_width = self.table.verticalHeader().width()
        col0_width = self.table.columnWidth(0)
        frame_width = self.table.frameWidth() * 2
        buffer = 20  # A little extra buffer for the scrollbar width
        total_width = v_header_width + col0_width + frame_width + buffer

        # Set the fixed width on the scroll area itself
        self.scroll_area.setFixedWidth(total_width)

    @Slot()
    def refresh_table(self, *args):
        """
        Refreshes the thickness table using stored lines and the current conversion factor.
        This is the single source of truth for table UI updates.
        """
        factor = self.pipeline.conversion_factor
        if factor is None or factor == 0:
            log.warning("Conversion factor is not set or is zero. Using raw pixel values.")
            factor = 1.0  # Fallback to a 1:1 ratio

        if not self.stored_lines:
            self.table.setRowCount(1)
            self.table.clearContents()
        else:
            self.table.setRowCount(len(self.stored_lines))
            for i, length_px in enumerate(self.stored_lines):
                scaled_length = length_px / factor
                item = QTableWidgetItem(f"{scaled_length:.2f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, 0, item)

        self._update_table_size()

    @Slot(list)
    def _on_lines_updated(self, lines: list[float]):
        """
        Updates the stored lines from the canvas and triggers a table refresh.
        """
        self.stored_lines = lines
        self.refresh_table()

    @Slot(np.ndarray)
    def _show_thickness_image(self, frame: np.ndarray):
        """Converts a numpy array to QPixmap and sets it as the canvas background."""
        pixmap = numpy_to_qpixmap(frame)
        self.line_canvas.set_background(pixmap)

    @Slot()
    def _reload_base_image(self):
        """Fetches the original image from the pipeline and resets the canvas."""
        if self.pipeline.right_image is not None:
            self._show_thickness_image(self.pipeline.right_image)