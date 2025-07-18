import logging
from PySide6.QtCore import Slot, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QTabWidget, QMessageBox, QApplication, QLineEdit, \
    QHBoxLayout, QLabel, QSizePolicy
from data_pipeline import DataPipeline
from processing.task_manager import TaskManager
from processing.data_loader import get_system_username
from widgets.file_picker import FilePickerWidget
from widgets.file_autoloader import find_and_prompt_for_video, find_and_prompt_for_csv
from tabs.plot_tab import PlotTab
from tabs.scale_tab import ScaleTab
from tabs.frame_tab import FrameTab
from tabs.level_tab import LevelTab
from tabs.thresh_tab import ThreshTab
from tabs.area_tab import AreaTab
from tabs.thickness_tab import ThicknessTab
from tabs.export_tab import ExportTab
import os

log = logging.getLogger(__name__)

class AnalysisSessionWidget(QWidget):
    """
    A widget that encapsulates a single, self-contained analysis session.
    It manages its own DataPipeline and the UI components related to it.
    """
    tab_name_requested = Signal(str)

    def __init__(self, task_manager: TaskManager, parent=None):
        """
        Initializes the session widget.

        Args:
            task_manager (TaskManager): The shared TaskManager instance from MainWindow.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.task_manager = task_manager
        self.pipeline = DataPipeline()
        self.pipeline.task_manager = self.task_manager

        self.init_ui()
        self.connect_signals()
        QApplication.instance().installEventFilter(self)
        self.pipeline.register_observer("author", self.tab_author.setText)
        log.info("AnalysisSessionWidget created.")

    def init_ui(self):
        """Initializes the user interface for this session."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # Use the full space of the tab

        # File selection widgets for this session
        self.file_pickers = FilePickerWidget()
        self.file_pickers.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.file_pickers, 1)

        # Text field for author name
        author_widget = QWidget()
        author_widget.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Preferred
        )
        author_layout = QVBoxLayout(author_widget)
        author_label = QLabel("Author:")
        author_layout.addWidget(author_label)
        self.tab_author = QLineEdit()
        self.tab_author.setPlaceholderText("Enter name...")
        username = get_system_username()
        self.pipeline.on_author_changed(username)
        self.tab_author.setText(username)
        self.tab_author.setClearButtonEnabled(True)
        self.tab_author.textEdited.connect(self.pipeline.on_author_changed)
        author_layout.addWidget(self.tab_author)

        top_layout.addWidget(author_widget, 0)

        main_layout.addLayout(top_layout)

        #main_layout.addWidget(self.file_pickers)

        # A visual separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep)

        # The tab widget for all the analysis steps (Plot, Scale, etc.)
        self.analysis_tabs = QTabWidget()
        main_layout.addWidget(self.analysis_tabs)

        self.init_analysis_tabs()

    def eventFilter(self, watched, event):
        """
        Filters events for the entire application. Used here to clear focus
        from the new_tab_name_input when the user clicks elsewhere.
        """
        if event.type() == QEvent.MouseButtonPress:
            # Check if the input field currently has focus
            if self.tab_author.hasFocus():
                # Find out which widget was actually clicked on
                clicked_widget = QApplication.widgetAt(event.globalPos())

                # If the click was outside the line edit and its children (like the clear button), then unfocus it.
                if clicked_widget and clicked_widget != self.tab_author and not self.tab_author.isAncestorOf(
                        clicked_widget):
                    self.tab_author.clearFocus()

        # Pass the event on to the parent class for default processing
        return super().eventFilter(watched, event)

    def init_analysis_tabs(self):
        """Creates and adds all the analysis-specific tabs."""
        self.plot_tab = PlotTab(self.pipeline)
        self.analysis_tabs.addTab(self.plot_tab, "📈 Plot")
        self.scale_tab = ScaleTab(self.pipeline)
        self.analysis_tabs.addTab(self.scale_tab, "📏 Scale")
        self.frame_tab = FrameTab(self.pipeline)
        self.analysis_tabs.addTab(self.frame_tab, "🎞️ Frame Picker")
        self.level_tab = LevelTab(self.pipeline)
        self.analysis_tabs.addTab(self.level_tab, "🌗 Level")
        self.thresh_tab = ThreshTab(self.pipeline)
        self.analysis_tabs.addTab(self.thresh_tab, "🏁 Threshold")
        self.area_tab_left = AreaTab(self.pipeline, "left")
        self.analysis_tabs.addTab(self.area_tab_left, "⭕ Area")
        self.thickness_tab = ThicknessTab(self.pipeline)
        self.analysis_tabs.addTab(self.thickness_tab, "✒️ Thickness")
        self.export_tab = ExportTab(self.pipeline) # Assuming this is correct from original
        self.analysis_tabs.addTab(self.export_tab, "📦 Export")

        self.analysis_tabs.setCurrentIndex(0)

    def connect_signals(self):
        """Connects signals from widgets to the appropriate slots in this session."""
        self.file_pickers.csv_selected.connect(self.on_csv_file_selected)
        self.file_pickers.video_selected.connect(self.on_video_file_selected)

    @Slot(str)
    def on_csv_file_selected(self, path: str):
        """Handles the selection of a CSV file for this session."""
        log.info(f"Session received CSV file path: {path}")
        self._handle_csv_load(path)
        # Attempt to find and load the corresponding video file
        video_path_to_load = find_and_prompt_for_video(self, path)
        if video_path_to_load:
            self._handle_video_load(video_path_to_load)

    def _handle_csv_load(self, path: str):
        """Single source of truth for loading a CSV file within this session."""
        log.info(f"HANDLER: Loading CSV {path} for session.")
        self.pipeline.load_csv_file(path)
        self.file_pickers.set_csv_label(path)

        # Extract the base filename without the extension
        file_name = os.path.basename(path)
        base_name, _ = os.path.splitext(file_name)

        # Emit the signal to request the parent change the tab name
        self.tab_name_requested.emit(base_name)

    @Slot(str)
    def on_video_file_selected(self, path: str):
        """Handles the selection of a video file for this session."""
        log.info(f"Session received video file path: {path}")
        self._handle_video_load(path)
        # Attempt to find and load the corresponding CSV file
        csv_path_to_load = find_and_prompt_for_csv(self, path)
        if csv_path_to_load:
            self._handle_csv_load(csv_path_to_load)

    def _handle_video_load(self, path: str):
        """Single source of truth for loading a video file within this session."""
        log.info(f"HANDLER: Loading Video {path} for session.")
        self.pipeline.load_video_file(path)
        self.file_pickers.set_video_label(path)