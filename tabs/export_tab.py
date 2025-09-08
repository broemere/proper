from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QComboBox, QFormLayout
from data_pipeline import DataPipeline
import json
import logging
log = logging.getLogger(__name__)

class ExportTab(QWidget):
    """
    A widget tab displaying a canvas for drawing lines and a table showing their lengths.
    """

    FIELD_DEFINITIONS = {
        "Pressure": "mmHg",
        "Frame/Row": "",
        "Radius a": "mm",
        "Radius b": "mm",
        "Wall Thickness": "mm",
        "Volume": "mm³",
        "V_wall": "mm³",
        "V_lumen": "mm³",
    }

    def __init__(self, pipeline: DataPipeline, settings: QSettings, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.settings = settings  # Store the settings object
        self.first_labels = {}
        self.last_labels = {}
        self.first_selector_combo = None
        self.last_selector_combo = None
        self.init_ui()
        #self.pipeline.register_observer("state_dict", self.save_state)
        #self.export_btn.clicked.connect(self.pipeline.get_state)
        #self.selector_combo.currentIndexChanged.connect(lambda index: self._selection_changed(index))
        self.pipeline.register_observer("results_updated", self._update_all_fields)

    def init_ui(self):
        """
        Initializes the user interface, splitting it into a canvas and a table area.
        """
        main_layout = QVBoxLayout(self)

        ellipses_layout = QHBoxLayout()

        data_sections_layout = QHBoxLayout()

        ellipse_label = QLabel("Ellipses to use:")
        self.selector_combo = QComboBox()
        self.selector_combo.addItems([str(i) for i in range(1, 5)])
        self.selector_combo.currentIndexChanged.connect(self._selection_changed)
        #self.selector_combo.setCurrentIndex(3)
        self.pipeline.n_ellipses = self.selector_combo.currentIndex()
        ellipses_layout.addWidget(ellipse_label)
        ellipses_layout.addWidget(self.selector_combo)
        ellipses_layout.addStretch()
        main_layout.addLayout(ellipses_layout)

        # --- Create "First" and "Last" sections ---
        first_section_widget, self.first_labels = self._create_data_section("First")
        last_section_widget, self.last_labels = self._create_data_section("Last")

        data_sections_layout.addWidget(first_section_widget)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        data_sections_layout.addWidget(separator)

        data_sections_layout.addWidget(last_section_widget)

        main_layout.addLayout(data_sections_layout)
        main_layout.addStretch(1)

        # --- Bottom Section: Export Control ---
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_btn = QPushButton("Export Results")
        export_layout.addWidget(self.export_btn)
        self.export_btn.clicked.connect(self.pipeline.generate_report)
        export_layout.addStretch()

        main_layout.addLayout(export_layout)

        saved_index = self.settings.value("export/n_ellipses", defaultValue=3, type=int)
        self.selector_combo.setCurrentIndex(saved_index)

        # Manually call this once to ensure the pipeline has the correct initial value
        #self._selection_changed(self.selector_combo.currentIndex())

    def _create_data_section(self, title: str) -> tuple[QWidget, dict[str, QLabel], QComboBox]:
        """
        Helper function to create a UI section for displaying data.

        Args:
            title (str): The title of the section (e.g., "First" or "Last").

        Returns:
            A tuple containing the container widget, a dictionary of value labels,
            and the combo box selector.
        """
        container = QWidget()
        layout = QVBoxLayout(container)

        title_label = QLabel(f"<b>{title}</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        value_labels = {}
        for label_text, unit in self.FIELD_DEFINITIONS.items():
            value_label = QLabel(f"N/A {unit}")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_labels[label_text] = value_label
            form_layout.addRow(f"{label_text}:", value_label)

        layout.addLayout(form_layout)
        layout.addStretch()

        return container, value_labels

    def _format_value(self, value: float) -> str:
        """
        Formats a number with a variable number of decimal places based on its magnitude.
        - 3 decimal places for numbers < 1
        - 2 decimal places for numbers < 10
        - 1 decimal place for numbers >= 10
        """
        # Gracefully handle cases where the value might not be a number
        if not isinstance(value, (int, float)):
            return str(value)

        # NEW: Check if the number is effectively an integer (e.g., 12 or 12.0)
        if value == int(value):
            return f"{int(value)}" # Format as a simple integer

        num_abs = abs(value)

        if num_abs < 1:
            precision = 3
        elif num_abs < 10:
            precision = 2
        else:  # For numbers 10 and greater
            precision = 1

        # Use an f-string with a dynamic precision
        return f"{value:.{precision}f}"

    @Slot(dict)
    def _update_all_fields(self, data: dict):
        """
        Updates all data fields in the UI from a single data dictionary,
        using custom formatting for the values.
        """
        log.info("Updating all fields with new data.")

        # Update "First" section labels
        first_data = data.get("first", {})
        for key, value in first_data.items():
            if key in self.first_labels:
                unit = self.FIELD_DEFINITIONS.get(key, "")
                formatted_value = self._format_value(value)  # Use the helper
                self.first_labels[key].setText(f"{formatted_value} {unit}")

        # Update "Last" section labels
        last_data = data.get("last", {})
        for key, value in last_data.items():
            if key in self.last_labels:
                unit = self.FIELD_DEFINITIONS.get(key, "")
                formatted_value = self._format_value(value)  # Use the helper
                self.last_labels[key].setText(f"{formatted_value} {unit}")

    @Slot(int)
    def _selection_changed(self, index: int):
        """
        Handles changes in the number of ellipses selector, updates the pipeline,
        and saves the selection for the next session.
        """
        n_ellipses = index + 1
        log.info(f"Number of ellipses changed to: {n_ellipses}")

        # 1. Update the data pipeline with the new value
        self.pipeline.n_ellipses = n_ellipses

        # 2. Save the selected index to QSettings for persistence
        self.settings.setValue("export/n_ellipses", index)
