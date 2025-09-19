from PySide6.QtCore import Qt, Signal, QEvent, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy, QScrollArea,
    QTabWidget, QTableWidget, QHeaderView, QTableWidgetItem
)
from data_pipeline import DataPipeline
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
        self.init_ui()
        self.connect_signals()

    def showEvent(self, event):
        super().showEvent(event)
        self.pipeline.ensure_segmentation_is_up_to_date()

        if not self._has_been_shown:
            self._has_been_shown = True
            self._center_active_tab(0)

    def init_ui(self):
        self.area_widget1 = AreaAnalysisWidget()
        self.area_widget2 = AreaAnalysisWidget()
        layout = QHBoxLayout(self)

        self.tabs = QTabWidget()
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

    def connect_signals(self):
        # --- Connections FROM UI TO PIPELINE (User Actions) ---
        self.area_widget1.area_clicked.connect(
            lambda props: self.pipeline.add_area_data('left', props)
        )
        self.area_widget2.area_clicked.connect(
            lambda props: self.pipeline.add_area_data('right', props)
        )

        # --- Connections FROM PIPELINE TO UI (Data Updates) ---
        self.pipeline.visualization_changed.connect(self.on_visualization_ready)
        self.pipeline.area_data_changed.connect(self.on_area_data_updated)
        self.pipeline.conversion_factor_changed.connect(self.refresh_table)

        # --- Local UI Connections (No Data Logic) ---
        self.tabs.currentChanged.connect(self._center_active_tab)

    @Slot(str, dict)
    def on_area_data_updated(self, left_right: str, data: dict):
        """
        This single slot handles updates from the pipeline for both sides.
        It refreshes the UI with the new, authoritative data.
        """
        if left_right == 'left':
            pixmap = self.area_widget1.display_pixmap
            labels = self.area_widget1.labeled_array
            if pixmap and labels is not None:
                self.area_widget1.update_markers(data)
                self._update_table_ui(self.table1, 'left')  # Just pass the side
        else:
            pixmap = self.area_widget2.display_pixmap
            labels = self.area_widget2.labeled_array
            if pixmap and labels is not None:
                self.area_widget2.update_markers(data)
                self._update_table_ui(self.table2, 'right')  # Just pass the side

    def on_visualization_ready(self, data: dict):
        """Receives the final data dictionary and updates the analysis widget."""
        labels_array = data['labels']
        visual_array = data['visual']
        left_right = data['left_right']
        visual_pixmap = numpy_to_qpixmap(visual_array)
        if left_right == "left":
            self.area_widget1.set_image_data(
                visual_pixmap,
                labels_array
            )
            # We can also call update_markers here for the initial state
            self.area_widget1.update_markers(self.pipeline.area_data_left)
            self._center_active_tab(0)
        else:
            self.area_widget2.set_image_data(
                visual_pixmap,
                labels_array
            )
            self.area_widget2.update_markers(self.pipeline.area_data_right)
            self._center_active_tab(1)

    def refresh_table(self, new_factor):
        """
        Observer slot that triggers a UI refresh for both tables
        when the conversion factor changes.
        """
        log.info(f"Refresh triggered by new factor: {new_factor}")
        self._update_table_ui(self.table1, 'left')
        self._update_table_ui(self.table2, 'right')

    def _update_table_ui(self, target_table, side: str):
        """
        Refactored UI update logic. Clears and repopulates a table
        based on its data store and the current conversion factor.
        """
        target_table.clearContents()
        display_data = self.pipeline.get_display_area_data(side)
        for row_index, (position_label, area_string) in enumerate(display_data):
            position_item = QTableWidgetItem(position_label)
            position_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            target_table.setItem(row_index, 0, position_item)
            area_item = QTableWidgetItem(area_string)
            area_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            target_table.setItem(row_index, 1, area_item)

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
