from PySide6.QtCore import Qt, Slot, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QComboBox, QFormLayout
from PySide6.QtGui import QDesktopServices
from data_pipeline import DataPipeline
from processing.data_transform import format_value
import platform
import subprocess
import logging
import os
log = logging.getLogger(__name__)

class ExportTab(QWidget):

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

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.first_labels = {}
        self.last_labels = {}
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initializes the user interface."""
        main_layout = QVBoxLayout(self)
        ellipses_layout = QHBoxLayout()
        data_sections_layout = QHBoxLayout()
        ellipse_label = QLabel("Ellipses to use:")
        self.selector_combo = QComboBox()
        self.selector_combo.addItems([str(i) for i in range(1, 5)])
        ellipses_layout.addWidget(ellipse_label)
        ellipses_layout.addWidget(self.selector_combo)
        ellipses_layout.addStretch()
        main_layout.addLayout(ellipses_layout)

        # --- Create "First" and "Last" sections ---
        first_section_widget, self.first_labels = self._create_data_section("First")
        last_section_widget, self.last_labels = self._create_data_section("Last")
        data_sections_layout.addWidget(first_section_widget)
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        data_sections_layout.addWidget(separator)
        data_sections_layout.addWidget(last_section_widget)
        main_layout.addLayout(data_sections_layout)
        main_layout.addStretch(1)

        # --- Bottom Section: Export Control ---
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        # 1. Style the Export Button
        self.export_btn = QPushButton("📥 Export Results")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.setMinimumWidth(150)
        self.export_btn.setStyleSheet("""
                    QPushButton {
                        border-radius: 5px;
                        border: 1px solid #223620;
                        font-weight: bold;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #88C484;
                    }
                    QPushButton:pressed {
                        background-color: #40633D;  /* Darker when clicked */
                        padding-left: 17px;        /* Subtle "push" effect */
                        padding-top: 7px;
                    }
                    QPushButton:disabled {
                        background-color: #a0a0a0;
                        border: 1px solid #808080;
                    }
                """)

        # 2. Style the Open Folder Button (Secondary Action)
        self.open_btn = QPushButton("📂 Open Folder")
        self.open_btn.setMinimumHeight(35)
        self.open_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #333333;
                        border: 1px solid #cccccc;
                        border-radius: 5px;
                        padding: 5px 15px;
                    }
                    QPushButton:hover {
                        background-color: #f0f0f0;
                        border-color: #999999;
                    }
                """)

        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.open_btn)
        export_layout.addStretch()
        main_layout.addLayout(export_layout)

    def connect_signals(self):
        """Connects UI widget signals to pipeline and pipeline signals to UI slots."""
        # When the user changes the combo box, tell the pipeline
        self.selector_combo.currentIndexChanged.connect(self._selection_changed)
        # When the pipeline's n_ellipses changes, update the combo box
        self.pipeline.n_ellipses_changed.connect(self._update_selector_display)
        # When pipeline results are updated, update the labels
        self.pipeline.results_updated.connect(self._update_all_fields)
        # Connect buttons
        self.export_btn.clicked.connect(self.pipeline.generate_report)
        self.open_btn.clicked.connect(self.open_current_directory)


    def _create_data_section(self, title: str) -> tuple[QWidget, dict[str, QLabel]]:
        """
        Helper function to create a UI section for displaying data.
        Args:
            title (str): The title of the section (e.g., "First" or "Last").
        Returns:
            A tuple containing the container widget, a dictionary of value labels
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
        """Updates all data fields in the UI from the pipeline's data dictionary."""
        log.info("Updating all fields with new data.")
        sections = {
            "first": self.first_labels,
            "last": self.last_labels,
        }
        for section_key, labels_dict in sections.items():
            section_data = data.get(section_key, {})
            for key, value in section_data.items():
                if key in labels_dict:
                    unit = self.FIELD_DEFINITIONS.get(key, "")
                    formatted_value = format_value(value)
                    labels_dict[key].setText(f"{formatted_value} {unit}")
                    if key == "V_lumen" and float(formatted_value) < 0:
                        labels_dict[key].setText(f"❌❌❌ {formatted_value} ❌❌❌ {unit}")

    @Slot(int)
    def _selection_changed(self, index: int):
        """Informs the pipeline that the user has changed the number of ellipses."""
        # Tell the pipeline about the change. The pipeline will then emit a
        # signal, which will trigger saving and any other UI updates.
        self.pipeline.set_n_ellipses(index + 1)

    @Slot(int)
    def _update_selector_display(self, n_ellipses: int):
        """Updates the QComboBox to reflect the current state of the pipeline."""
        self.selector_combo.blockSignals(True)
        self.selector_combo.setCurrentIndex(n_ellipses - 1)
        self.selector_combo.blockSignals(False)

    @Slot()
    def open_current_directory(self):
        """Opens the OS file explorer at the directory of the current CSV."""
        # Assuming self.pipeline.csv_path exists based on your description
        csv_path = getattr(self.pipeline, 'csv_path', None)

        if not csv_path or not os.path.exists(csv_path):
            log.warning("Cannot open folder: CSV path is invalid or does not exist.")
            return

        folder_path = os.path.dirname(os.path.abspath(csv_path))

        # The Cross-Platform Way
        if platform.system() == "Windows":
            # On Windows, QDesktopServices works, but using 'explorer' directly
            # allows you to pre-select the file if you wanted.
            # For just the folder, this is the most reliable:
            if self.pipeline.exported_file is not None:
                subprocess.run(['explorer', '/select,', os.path.normpath(self.pipeline.exported_file)])
            else:
                os.startfile(folder_path)
        else:
            # On macOS and Linux, QDesktopServices is excellent
            folder_url = QUrl.fromLocalFile(folder_path)
            QDesktopServices.openUrl(folder_url)
