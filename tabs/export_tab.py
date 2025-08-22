from PySide6.QtCore import Qt, Slot
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
        "Radius a": "mm",
        "Radius b": "mm",
        "Wall Thickness": "mm",
        "Volume": "mm³",
        "V_wall": "mm³",
        "V_lumen": "mm³",
    }

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.first_labels = {}
        self.last_labels = {}
        self.first_selector_combo = None
        self.last_selector_combo = None
        self.init_ui()
        #self.pipeline.register_observer("state_dict", self.save_state)
        #self.export_btn.clicked.connect(self.pipeline.get_state)
        self.selector_combo.currentIndexChanged.connect(lambda index: self._selection_changed(index))
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
        self.selector_combo.setCurrentIndex(4)
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
        export_layout.addStretch()

        main_layout.addLayout(export_layout)

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

    @Slot(dict)
    def _update_all_fields(self, data: dict):
        """
        Updates all data fields in the UI from a single data dictionary.

        Args:
            data (dict): A dictionary containing 'first' and 'last' keys,
                         with sub-dictionaries for the data fields.
        """
        log.info("Updating all fields with new data.")

        # Update "First" section labels
        first_data = data.get("first", {})
        for key, value in first_data.items():
            if key in self.first_labels:
                unit = self.FIELD_DEFINITIONS.get(key, "")
                self.first_labels[key].setText(f"{value:.2f} {unit}")

        # Update "Last" section labels
        last_data = data.get("last", {})
        for key, value in last_data.items():
            if key in self.last_labels:
                unit = self.FIELD_DEFINITIONS.get(key, "")
                self.last_labels[key].setText(f"{value:.2f} {unit}")

    @Slot(str, int)
    def _selection_changed(self, index: int):
        """
        Handles changes in the dropdown selectors.

        Args:
            section (str): Identifier for the section ('first' or 'last').
            index (int): The new zero-based index of the combo box.
        """
        selected_option = index + 1
        log.info(f"Dropdown changed to: {selected_option}")
        # Example of how you might call your pipeline
        # if section == 'first':
        #     self.pipeline.set_first_ellipse(selected_option)
        # elif section == 'last':
        #     self.pipeline.set_last_ellipse(selected_option)