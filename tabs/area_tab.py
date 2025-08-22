from PySide6.QtCore import Qt, Signal, QEvent, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy, QScrollArea,
    QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem
)
from data_pipeline import DataPipeline
import numpy as np
from widgets.area_widget import AreaAnalysisWidget
from processing.data_transform import numpy_to_qpixmap
import logging
log = logging.getLogger(__name__)


class AreaTab(QWidget):

    def __init__(self, pipeline: DataPipeline, left_right=None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.segmented_result = None
        self._has_been_shown = False
        self._is_state_synced = False
        self.pipeline.left_threshed_old = None
        self.pipeline.right_threshed_old = None
        self.init_ui()
        self.pipeline.register_observer("state_loaded", self._on_state_loaded)
        self.pipeline.register_observer("visualization_ready", self.on_visualization_ready)
        self.pipeline.register_observer("conversion_factor", self.refresh_table)
        self.pipeline.register_observer("area_data_updated", self.on_area_data_updated)


    def init_ui(self):
        self.area_widget1 = AreaAnalysisWidget()
        self.area_widget2 = AreaAnalysisWidget()
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

        # Connect the click signal directly to a handler that calls the pipeline
        self.area_widget1.area_clicked.connect(
            lambda props: self.pipeline.add_area_data('left', props)
        )
        self.area_widget2.area_clicked.connect(
            lambda props: self.pipeline.add_area_data('right', props)
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

        if self.isVisible() and not self._is_state_synced:
            self._sync_ui_to_pipeline()

    @Slot()
    def _on_state_loaded(self):
        """Marks the UI as dirty when a new state is loaded."""
        print("AREA TAB: Received 'state_loaded' notification.")
        self._is_state_synced = False
        if self.isVisible():
            self._sync_ui_to_pipeline()

    def _sync_ui_to_pipeline(self):
        """Pulls all necessary state from the pipeline to fully refresh the tab."""
        print("AREA TAB: Synchronizing UI to pipeline state.")

        # 1. Update the left image and table
        if self.pipeline.left_image is not None:
            # Note: 'segmented_result' might not be available on load,
            # we'll need to regenerate it if we want the outlines.
            # For now, let's just show the thresholded image.
            pixmap = numpy_to_qpixmap(self.pipeline.left_threshed)
            #self.area_widget1.set_data(pixmap, self.pipeline.left_threshed)
            self._update_table_ui(self.table1, self.pipeline.area_data_left)

        # 2. Update the right image and table
        if self.pipeline.right_image is not None:
            pixmap = numpy_to_qpixmap(self.pipeline.right_threshed)
            #self.area_widget2.set_data(pixmap, self.pipeline.right_threshed)
            self._update_table_ui(self.table2, self.pipeline.area_data_right)

        self._is_state_synced = True

    @Slot(str)
    def on_area_data_updated(self, left_right: str):
        """
        This single slot handles updates from the pipeline for both sides.
        It refreshes the UI with the new, authoritative data.
        """
        if left_right == 'left':
            data = self.pipeline.area_data_left
            pixmap = self.area_widget1.display_pixmap # Reuse existing pixmap
            labels = self.area_widget1.labeled_array # Reuse existing array
            if pixmap and labels is not None:
                self.area_widget1.set_data(pixmap, labels, data)
                self._update_table_ui(self.table1, data)
        else:
            data = self.pipeline.area_data_right
            pixmap = self.area_widget2.display_pixmap
            labels = self.area_widget2.labeled_array
            if pixmap and labels is not None:
                self.area_widget2.set_data(pixmap, labels, data)
                self._update_table_ui(self.table2, data)

    def on_visualization_ready(self, data: dict):
        """Receives the final data dictionary and updates the analysis widget."""
        labels_array = data['labels']
        visual_array = data['visual']
        left_right = data['left_right']

        visual_pixmap = numpy_to_qpixmap(visual_array)
        if left_right == "left":
            self.area_widget1.set_data(
                numpy_to_qpixmap(data['visual']),
                data['labels'],
                self.pipeline.area_data_left
            )
            self._center_active_tab(0)
        else:
            self.area_widget2.set_data(
                numpy_to_qpixmap(data['visual']),
                data['labels'],
                self.pipeline.area_data_right
            )
            self._center_active_tab(1)

    def refresh_table(self, new_factor):
        """
        Observer slot that triggers a UI refresh for both tables
        when the conversion factor changes.
        """
        log.info(f"Refresh triggered by new factor: {new_factor}")
        self._update_table_ui(self.table1, self.pipeline.area_data_left)
        self._update_table_ui(self.table2, self.pipeline.area_data_right)


    def _update_table_ui(self, target_table, data_store):
        """
        Refactored UI update logic. Clears and repopulates a table
        based on its data store and the current conversion factor.
        """
        target_table.clearContents()

        blobs = data_store.values()

        for row_index, props in enumerate(blobs):
            area, cx, cy, label = props
            position_item = QTableWidgetItem(label)
            position_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            target_table.setItem(row_index, 0, position_item)

            # ... (your area calculation logic is fine)
            try:
                scaled_area = area / (self.pipeline.conversion_factor ** 2)
            except (ZeroDivisionError, TypeError):
                log.warning("Scale not set or invalid!!! Using raw area.")
                scaled_area = area

            area_item = QTableWidgetItem(f"{scaled_area:.3f}")
            area_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            target_table.setItem(row_index, 1, area_item)

        # for row_index, data in enumerate(data_store):
        #     if position_map:
        #         data_tuple = tuple(data)
        #         position_label = position_map.get(data_tuple, "")
        #         position_item = QTableWidgetItem(position_label)
        #         position_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        #         target_table.setItem(row_index, 0, position_item)
        #
        #     # Recalculate area using the current conversion factor
        #     area_value = data[1] # This is the raw area
        #     try:
        #         # Use the pipeline's conversion factor for calculation
        #         scaled_area = area_value / (self.pipeline.conversion_factor ** 2)
        #     except (ZeroDivisionError, TypeError):
        #         log.warning("Scale not set or invalid!!! Using raw area.")
        #         scaled_area = area_value
        #
        #     area_item = QTableWidgetItem(f"{scaled_area:.3f}")
        #     area_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        #     target_table.setItem(row_index, 1, area_item)

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