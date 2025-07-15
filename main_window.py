from PySide6.QtCore import QSettings, Slot
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QFrame, QTabWidget, QMessageBox
from config import APP_NAME
from data_pipeline import DataPipeline
from processing.task_manager import TaskManager
from widgets.status_bar import StatusBarWidget
from widgets.file_picker import FilePickerWidget
from tabs.plot_tab import PlotTab
from tabs.scale_tab import ScaleTab
from tabs.frame_tab import FrameTab
from widgets.file_autoloader import find_and_prompt_for_video, find_and_prompt_for_csv
from tabs.level_tab import LevelTab
from tabs.thresh_tab import ThreshTab
from tabs.area_tab import AreaTab
from tabs.thickness_tab import ThicknessTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = QSettings()
        self._restore_window()

        self.pipeline = DataPipeline()
        self.task_manager = TaskManager()
        self.pipeline.task_manager = self.task_manager

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.file_pickers = FilePickerWidget()
        main_layout.addWidget(self.file_pickers)
        self._add_separator(main_layout)
        self.init_tabs(main_layout)
        self._add_separator(main_layout)
        self.status_bar = StatusBarWidget()
        main_layout.addWidget(self.status_bar)

        self.file_pickers.csv_selected.connect(self.on_csv_file_selected)
        self.file_pickers.video_selected.connect(self.on_video_file_selected)
        self.task_manager.status_updated.connect(self.status_bar.update_status)
        self.task_manager.progress_updated.connect(self.status_bar.update_progress)
        self.task_manager.batch_finished.connect(self.status_bar.batch_finished)
        self.status_bar.cancel_clicked.connect(self.task_manager.cancel_batch)
        self.task_manager.error_occurred.connect(self.show_error_dialog)

    def _add_separator(self, parent_layout) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        parent_layout.addWidget(sep)

    def init_tabs(self, parent_layout: QVBoxLayout):
        self.tabs = QTabWidget()
        # placeholder tabs:
        for i in (0,):
            placeholder = QWidget()
            self.tabs.addTab(placeholder, f"Tab {i+1}")

        self.plot_tab = PlotTab(self.pipeline)
        self.tabs.addTab(self.plot_tab, f"Plot")
        self.scale_tab = ScaleTab(self.pipeline)
        self.tabs.addTab(self.scale_tab, f"Scale")
        self.frame_tab = FrameTab(self.pipeline)
        self.tabs.addTab(self.frame_tab, f"Frame Picker")
        self.level_tab = LevelTab(self.pipeline)
        self.tabs.addTab(self.level_tab, f"Level")
        self.thresh_tab = ThreshTab(self.pipeline)
        self.tabs.addTab(self.thresh_tab, f"Threshold")
        self.area_tab_left = AreaTab(self.pipeline, "left")
        self.tabs.addTab(self.area_tab_left, f"Area")
        self.thickness_tab = ThicknessTab(self.pipeline)
        self.tabs.addTab(self.thickness_tab, f"Thickness")

        self.tabs.setCurrentIndex(1)
        parent_layout.addWidget(self.tabs)

    def closeEvent(self, event):
        # Before the window actually closes, save the geometry
        self.settings.setValue("windowGeometry", self.saveGeometry())
        super().closeEvent(event)

    def _restore_window(self):
        # Try to pull a saved QByteArray geometry; if not present, fall back to defaults:
        geometry = self.settings.value("windowGeometry")
        if geometry is not None:
            # restoreGeometry takes a QByteArray and restores size+pos+state
            self.restoreGeometry(geometry)
        else:
            self.resize(1300, 750)  # default size

    @Slot(str)
    def on_csv_file_selected(self, path: str):
        """This is the controller logic. It connects the View to the Model."""
        print(f"MainWindow received file path: {path}")
        self._handle_csv_load(path)
        video_path_to_load = find_and_prompt_for_video(self, path)
        if video_path_to_load:
            self._handle_video_load(video_path_to_load)

    def _handle_csv_load(self, path: str):
        """Single source of truth for loading a CSV file."""
        print(f"HANDLER: Loading CSV {path}")
        self.pipeline.load_csv_file(path)
        self.file_pickers.set_csv_label(path) # Optional: Update UI

    @Slot(str)
    def on_video_file_selected(self, path: str):
        """This is the controller logic. It connects the View to the Model."""
        print(f"MainWindow received file path: {path}")
        self._handle_video_load(path)
        csv_path_to_load = find_and_prompt_for_csv(self, path)
        if csv_path_to_load:
            self._handle_csv_load(csv_path_to_load)

    def _handle_video_load(self, path: str):
        """Single source of truth for loading a video file."""
        print(f"HANDLER: Loading Video {path}")
        self.pipeline.load_video_file(path)
        self.pipeline.load_left_frame()  # This can now be part of the definitive sequence
        self.file_pickers.set_video_label(path) # Optional: Update UI


    @Slot(tuple)
    def show_error_dialog(self, err_tb):
        """Displays a user-friendly error message."""
        exc, tb_str = err_tb
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("An Error Occurred")
        msg_box.setText(f"A background task has failed.\n\nError: {exc}")
        msg_box.setInformativeText(
            "Please check the 'proper.log' file for detailed technical information."
        )
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()
