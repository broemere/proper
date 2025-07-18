from PySide6.QtCore import Qt, Signal, QPointF, QRect
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy, QScrollArea,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup, QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem
)
from PySide6.QtGui import QPalette, QPixmap, QColor, QPainter, QImage, QPen, QCursor, QIcon
from data_pipeline import DataPipeline
import numpy as np
from processing.resource_loader import load_cursor
from widgets.area_widget import AreaAnalysisWidget
from processing.data_transform import numpy_to_qpixmap
from collections import deque
import logging
log = logging.getLogger(__name__)


class AreaTab(QWidget):

    def __init__(self, pipeline: DataPipeline, left_right=None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.segmented_result = None
        self._has_been_shown = False
        self.pipeline.left_threshed_old = None
        self.pipeline.right_threshed_old = None
        self.pipeline.MAX_MARKERS = 5
        self.init_ui()
        self.pipeline.register_observer("visualization_ready", self.on_visualization_ready)
        self.pipeline.register_observer("conversion_factor", self.refresh_table)

        self.data_store1 = deque(maxlen=5)
        self.data_store2 = deque(maxlen=5)

    def init_ui(self):
        self.area_widget1 = AreaAnalysisWidget(self.pipeline.MAX_MARKERS)
        self.area_widget2 = AreaAnalysisWidget(self.pipeline.MAX_MARKERS)
        layout = QHBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._center_active_tab)

        # --- Tab 1 with ScrollArea ---
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.scroll_area1 = QScrollArea()  # Create a scroll area
        self.scroll_area1.setWidget(self.area_widget1)  # Place the widget inside
        self.scroll_area1.setWidgetResizable(False)  # Ensure the widget keeps its fixed size
        tab1_layout.addWidget(self.scroll_area1)  # Add the scroll area to the layout
        self.tabs.addTab(tab1, "First")

        # --- Tab 2 with ScrollArea ---
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.scroll_area2 = QScrollArea()  # Create another scroll area
        self.scroll_area2.setWidget(self.area_widget2)  # Place the second widget inside
        self.scroll_area2.setWidgetResizable(False)
        tab2_layout.addWidget(self.scroll_area2)  # Add it to the second tab
        self.tabs.addTab(tab2, "Last")

        layout.addWidget(self.tabs, 1)

        # --- Right-side Table UI (unchanged) ---
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.table1 = QTableWidget(5, 2)
        self.table1.setHorizontalHeaderLabels(["Position", "Area [mm²]"])
        self.table2 = QTableWidget(5, 2)
        self.table2.setHorizontalHeaderLabels(["Position", "Area [mm²]"])
        table_layout.addWidget(self.table1)
        table_layout.addWidget(self.table2)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(table_container, 0)

        self.table1.resizeColumnsToContents()
        self.table2.resizeColumnsToContents()
        # --- FIXED: Re-added the fixed width calculation ---
        v_header_width = self.table1.verticalHeader().width()
        col0_width = self.table1.columnWidth(0)
        col1_width = self.table1.columnWidth(1)
        frame_width = self.table1.frameWidth() * 2
        buffer = 5 # Small buffer for borders/padding
        total_width = v_header_width + col0_width + col1_width + frame_width + buffer
        table_container.setFixedWidth(total_width)

        header_height = self.table1.horizontalHeader().height()
        rows_height = sum(self.table1.rowHeight(i) for i in range(self.table1.rowCount()))
        total_height = (header_height + rows_height + self.table1.frameWidth() * 2) * 2 + table_layout.spacing()
        table_container.setFixedHeight(total_height)

        self.area_widget1.area_clicked.connect(
            lambda data: self.handle_area_update(self.table1, self.data_store1, data)
        )
        self.area_widget2.area_clicked.connect(
            lambda data: self.handle_area_update(self.table2, self.data_store2, data)
        )

    def showEvent(self, event):
        super().showEvent(event)
        if self.pipeline.left_threshed is None:
            return

        if not self._has_been_shown:
            self._has_been_shown = True
            self._center_active_tab(0)

        # Determine if we need to run the segmentation task.
        # This is True if the backup doesn't exist OR if the current image is different from the backup.
        should_run_task = (self.pipeline.left_threshed_old is None or
                           not np.array_equal(self.pipeline.left_threshed, self.pipeline.left_threshed_old))

        if should_run_task:
            # If a run is needed, update the backup copy and queue the task.
            self.pipeline.left_threshed_old = self.pipeline.left_threshed.copy()
            self.pipeline.segment_image(self.pipeline.left_threshed, "left")

        if self.pipeline.right_threshed is None:
            return
        # Determine if we need to run the segmentation task.
        # This is True if the backup doesn't exist OR if the current image is different from the backup.
        should_run_task = (self.pipeline.right_threshed_old is None or
                           not np.array_equal(self.pipeline.right_threshed, self.pipeline.right_threshed_old))

        if should_run_task:
            # If a run is needed, update the backup copy and queue the task.
            self.pipeline.right_threshed_old = self.pipeline.right_threshed.copy()
            self.pipeline.segment_image(self.pipeline.right_threshed, "right")

    def on_visualization_ready(self, data: dict):
        """Receives the final data dictionary and updates the analysis widget."""
        labels_array = data['labels']
        visual_array = data['visual']
        left_right = data['left_right']

        visual_pixmap = numpy_to_qpixmap(visual_array)
        if left_right == "left":
            self.area_widget1.set_data(visual_pixmap, labels_array)
            self._center_active_tab(0)
        else:
            self.area_widget2.set_data(visual_pixmap, labels_array)
            self._center_active_tab(1)

    def refresh_table(self, new_factor):
        """
        Observer slot that triggers a UI refresh for both tables
        when the conversion factor changes.
        """
        log.info(f"Refresh triggered by new factor: {new_factor}")
        self._update_table_ui(self.table1, self.data_store1)
        self._update_table_ui(self.table2, self.data_store2)

    def _calculate_positions(self, data_store):
        if len(data_store) != 5:
            return {}
        blobs = list(data_store)
        left_blob = min(blobs, key=lambda b: b[2])
        right_blob = max(blobs, key=lambda b: b[2])
        top_blob = min(blobs, key=lambda b: b[3])
        bottom_blob = max(blobs, key=lambda b: b[3])
        all_blobs_set = {tuple(b) for b in blobs}
        extreme_blobs_set = {tuple(left_blob), tuple(right_blob), tuple(top_blob), tuple(bottom_blob)}
        middle_blobs_set = all_blobs_set - extreme_blobs_set
        middle_blob_tuple = list(middle_blobs_set)[0] if middle_blobs_set else None
        position_map = {}
        if middle_blob_tuple:
            position_map[middle_blob_tuple] = "Middle"
        position_map[tuple(left_blob)] = "Left"
        position_map[tuple(right_blob)] = "Right"
        position_map[tuple(top_blob)] = "Top"
        position_map[tuple(bottom_blob)] = "Bottom"
        return position_map

    def _update_table_ui(self, target_table, data_store):
        """
        Refactored UI update logic. Clears and repopulates a table
        based on its data store and the current conversion factor.
        """
        position_map = self._calculate_positions(data_store)
        target_table.clearContents()

        for row_index, data in enumerate(data_store):
            if position_map:
                data_tuple = tuple(data)
                position_label = position_map.get(data_tuple, "")
                position_item = QTableWidgetItem(position_label)
                position_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                target_table.setItem(row_index, 0, position_item)

            # Recalculate area using the current conversion factor
            area_value = data[1] # This is the raw area
            try:
                # Use the pipeline's conversion factor for calculation
                scaled_area = area_value / (self.pipeline.conversion_factor ** 2)
            except (ZeroDivisionError, TypeError):
                log.warning("Scale not set or invalid!!! Using raw area.")
                scaled_area = area_value

            area_item = QTableWidgetItem(f"{scaled_area:.3f}")
            area_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            target_table.setItem(row_index, 1, area_item)

    def handle_area_update(self, target_table, data_store, blob_props):
        """
        This slot now only updates the data store and then calls the
        reusable UI update function.
        """
        log.info(f"Updating table {target_table.objectName()} with data: {blob_props}")
        data_store.append(blob_props)
        self._update_table_ui(target_table, data_store)

    def _center_active_tab(self, index):
        """
        Slot for the QTabWidget's currentChanged signal.
        Centers the scroll area of the newly visible tab.
        """
        if index == 0: # "First" tab is now visible
            h_bar = self.scroll_area1.horizontalScrollBar()
            v_bar = self.scroll_area1.verticalScrollBar()
            h_bar.setValue(h_bar.maximum() // 2)
            v_bar.setValue(v_bar.maximum() // 2)
        elif index == 1: # "Last" tab is now visible
            h_bar = self.scroll_area2.horizontalScrollBar()
            v_bar = self.scroll_area2.verticalScrollBar()
            h_bar.setValue(h_bar.maximum() // 2)
            v_bar.setValue(v_bar.maximum() // 2)