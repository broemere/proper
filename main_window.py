import logging
from PySide6.QtCore import QSettings, Slot, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QMessageBox, QPushButton, QMenu, QFileDialog
from config import APP_NAME, APP_VERSION, SAVE_FILETYPE, REPO_URL
from processing.task_manager import TaskManager
from widgets.status_bar import StatusBarWidget
from widgets.analysis_session_widget import AnalysisSessionWidget
from widgets.error_bus import bus
from widgets.update_checker import UpdateChecker
from pathlib import Path
import json
import gzip
import os

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    The main application window. It manages the overall UI shell, including
    the 'supertabs' for different analysis sessions, the status bar, and the
    shared TaskManager.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = QSettings()
        self._restore_window()

        # These are the globally shared components
        self.task_manager = TaskManager()
        self.status_bar = StatusBarWidget()

        self.init_ui()
        self.connect_global_signals()

        # Start with a single analysis session by default
        self.add_new_super_tab()

        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self.on_update_available)
        self.update_checker.start()  # Runs in background, won't freeze app

        log.info("MainWindow initialized with one session.")

    def init_ui(self):
        """Initializes the main window's UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # Remove extra spacing between widgets
        main_layout.setSpacing(0)

        # --- Main QTabWidget for sessions ---
        self.super_tabs = QTabWidget()
        self.super_tabs.setTabsClosable(True)
        self.super_tabs.setMovable(True)
        self.super_tabs.setTabBarAutoHide(False)
        main_layout.addWidget(self.super_tabs)

        # --- Add buttons to the corners of the tab bar ---

        # Menu button (left corner)
        self.menu_button = QPushButton("☰ File")
        self.menu_button.setToolTip("File Menu")
        self.file_menu = QMenu(self)
        self.menu_button.setMenu(self.file_menu)
        self.super_tabs.setCornerWidget(self.menu_button, Qt.TopLeftCorner)

        # Add tab button (right corner)
        self.add_tab_button = QPushButton("+")
        self.add_tab_button.setToolTip("Open a new analysis session")
        self.super_tabs.setCornerWidget(self.add_tab_button, Qt.TopRightCorner)

        # --- Global status bar ---
        main_layout.addWidget(self.status_bar)

    def connect_global_signals(self):
        """Connects signals for globally shared components."""
        # Create menu actions
        new_action = QAction("New Session", self)
        save_action = QAction("Save Session", self)
        load_action = QAction("Load Session", self)
        self.file_menu.addAction(new_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(save_action)
        self.file_menu.addAction(load_action)

        # Connect actions to slots
        new_action.triggered.connect(self.add_new_super_tab)
        save_action.triggered.connect(self.on_save_session)
        load_action.triggered.connect(self.on_load_session)

        # Connect the "add tab" button to its slot
        self.add_tab_button.clicked.connect(self.add_new_super_tab)

        # Connect the close request for a supertab
        self.super_tabs.tabCloseRequested.connect(self.on_super_tab_close_requested)

        # Connect the shared TaskManager to the shared StatusBar
        self.task_manager.status_updated.connect(self.status_bar.update_status)
        self.task_manager.progress_updated.connect(self.status_bar.update_progress)
        self.task_manager.batch_finished.connect(self.status_bar.batch_finished)
        self.task_manager.error_occurred.connect(self.show_error_dialog)

        # Connect the cancel button from the status bar to the task manager
        self.status_bar.cancel_clicked.connect(self.task_manager.cancel_batch)

        # Connect error bus to dialog
        bus.user_error_details.connect(lambda exc, tb: self.show_error_dialog((exc, tb)))


    @Slot()
    def add_new_super_tab(self, unfocus=False):
        """Creates a new AnalysisSessionWidget and adds it as a new 'supertab'."""
        session_widget = AnalysisSessionWidget(self.task_manager, self.settings)
        session_widget.tab_name_requested.connect(self.on_tab_name_change_requested)
        session_widget.help_requested.connect(self.show_info_dialog)

        tab_name = f"Analysis {self.super_tabs.count() + 1}"
        index = self.super_tabs.addTab(session_widget, tab_name)
        if not unfocus:
            self.super_tabs.setCurrentIndex(index)
        log.info(f"Added new super tab: '{tab_name}'")
        return index

    @Slot(str)
    def on_tab_name_change_requested(self, new_name: str):
        """
        Sets the tab text for the widget that emitted the signal.
        This slot is connected when a new AnalysisSessionWidget is created.
        """
        sender_widget = self.sender()
        if sender_widget:
            index = self.super_tabs.indexOf(sender_widget)
            if index != -1:
                self.super_tabs.setTabText(index, new_name)
                log.info(f"Renamed tab at index {index} to '{new_name}'.")

    @Slot(int)
    def on_super_tab_close_requested(self, index: int):
        """Handles the request to close a 'supertab'."""
        widget_to_close = self.super_tabs.widget(index)
        if widget_to_close:
            # could add a confirmation dialog here if needed
            # e.g., if self.confirm_close():
            log.info(f"Closing super tab at index {index}.")
            self.super_tabs.removeTab(index)
            widget_to_close.deleteLater()  # Ensure proper memory cleanup
        if self.super_tabs.count() == 0:
            self.add_new_super_tab()

    @Slot()
    def on_save_session(self):
        """
        Initiates saving the state of the currently active analysis session.
        Gets the state and file path from the user, then queues the file
        writing and compression operation in the background TaskManager.
        """
        log.info("'Save Session' clicked")
        current_index = self.super_tabs.currentIndex()

        if current_index == -1:
            QMessageBox.information(self, "No Session to Save", "There is no open analysis session to save.")
            log.warning("Save action triggered, but no active tab was found.")
            return

        open_tab = self.super_tabs.widget(current_index)
        tab_name = self.super_tabs.tabText(current_index)

        try:
            state = open_tab.pipeline.get_state()
        except AttributeError:
            log.error(f"The widget in the current tab '{tab_name}' does not have a 'pipeline.get_state()' method.")
            QMessageBox.critical(self, "Error", f"Could not retrieve session state from '{tab_name}'.")
            return

        # Determine the initial directory for the file dialog
        initial_dir = ""
        if hasattr(open_tab, 'pipeline') and hasattr(open_tab.pipeline, 'csv_path') and open_tab.pipeline.csv_path:
            csv_dir = os.path.dirname(open_tab.pipeline.csv_path)
            if os.path.isdir(csv_dir):
                initial_dir = csv_dir
                log.info(f"Setting initial save directory based on CSV path: {initial_dir}")
            else:
                log.warning(f"Directory from CSV path does not exist: {csv_dir}")


        options = QFileDialog.Options()
        initial_filename = f"{tab_name}{SAVE_FILETYPE}"
        full_initial_path = os.path.join(initial_dir, initial_filename)


        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compressed Session As...",
            full_initial_path,
            f"Compressed JSON (*{SAVE_FILETYPE});;All Files (*)",
            options=options
        )

        if file_path:
            # Ensure the file has the correct extension
            if not file_path.endswith(SAVE_FILETYPE):
                file_path += SAVE_FILETYPE

            log.info(f"Queueing compressed save operation for: {file_path}")
            self.task_manager.queue_task(
                self._save_session_to_file_compressed,
                state,
                file_path,
                on_result=self._on_save_completion
            )
        else:
            log.info("Save operation was cancelled by the user.")

    def _save_session_to_file_compressed(self, signals, state: dict, file_path: str):
        """
        Worker function that writes the session state to a gzipped JSON file.
        Uses a compact JSON format for maximum compression.

        Args:
            signals: The signals object provided by the TaskManager.
            state: The JSON-serializable dictionary representing the session state.
            file_path: The full path to the file to be saved.

        Returns:
            The file_path if successful, otherwise None.
        """
        try:
            log.debug(f"Background task started: writing compressed data to {file_path}")
            signals.message.emit(f"Compressing and saving to {file_path}...")
            signals.progress.emit(25)

            # Use gzip.open with 'wt' for writing text compressed
            # encoding is important for json.dump.
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                # Dump the json in compact format (no indent) for better compression
                json.dump(state, f)

            signals.progress.emit(100)
            signals.message.emit("Save complete.")
            return file_path
        except (IOError, TypeError, gzip.BadGzipFile) as e:
            log.exception(f"Error in _save_session_to_file_compressed worker!")
            signals.message.emit(f"Error saving compressed file: {e}")
            return None

    @Slot(object)
    def _on_save_completion(self, result):
        """
        Callback function executed after the save task is completed.
        Updates the status bar with a success message if the task was successful.

        Args:
            result: The return value from the worker function (file_path or None).
        """
        if result:
            file_path = result
            log.info(f"Session saved successfully to {file_path}")
            #self.status_bar.update_status(f"Session saved to {file_path}", 5000)
        else:
            log.warning("Save task completed, but failed. See logs for details.")
            #self.status_bar.update_status("Save failed.", 5000)

    @Slot()
    def on_load_session(self):
        """
        Initiates loading a compressed session state from a file.
        Opens a file dialog, then queues a background task to read and
        decompress the file.
        """
        log.info("'Load Session' clicked")
        options = QFileDialog.Options()
        last_dir = self.settings.value("last_dir", "") or str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Compressed Session",
            last_dir,
            f"Compressed JSON (*{SAVE_FILETYPE});;All Files (*)",
            options=options
        )

        if file_path:
            log.info(f"Queueing load operation for: {file_path}")
            self.task_manager.queue_task(
                self._load_session_from_file_compressed,
                file_path,
                on_result=self._on_load_completion
            )
        else:
            log.info("Load operation was cancelled by the user.")

    def _load_session_from_file_compressed(self, signals, file_path: str):
        """
        Worker function that reads and decompresses a gzipped JSON file.
        """
        try:
            log.debug(f"Background task started: reading compressed data from {file_path}")
            signals.message.emit(f"Loading from {file_path}...")
            signals.progress.emit(25)
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                state = json.load(f)
            signals.progress.emit(100)
            signals.message.emit("Load complete.")
            return {'state': state, 'file_path': file_path}
        except (IOError, TypeError, gzip.BadGzipFile, json.JSONDecodeError) as e:
            log.exception(f"Error in _load_session_from_file_compressed worker!")
            signals.message.emit(f"Error loading file: {e}")
            return None

    @Slot(object)
    def _on_load_completion(self, result):
        """
        Callback function executed after the load task is completed.
        Creates a new tab and loads the session state into its pipeline.
        """
        if not (result and 'state' in result and 'file_path' in result):
            log.warning("Load task completed, but failed. See logs for details.")
            # self.status_bar.update_status("Load failed.", 5000)
            return
        state = result['state']
        file_path = result['file_path']
        log.info(f"Successfully loaded session from {file_path}. Creating new tab.")

        # Create a new tab but don't give it focus immediately
        new_tab_index = self.add_new_super_tab(unfocus=True)
        new_tab_widget = self.super_tabs.widget(new_tab_index)

        # Set the tab name
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.super_tabs.setTabText(new_tab_index, file_name)

        # Instead of loading the state immediately, schedule it to run
        # after the event loop has had a chance to fully create the new tab.
        QTimer.singleShot(0, lambda: self._finish_loading_state(new_tab_widget, state, file_name))

    def _finish_loading_state(self, new_tab_widget, state, file_name):
        """
        This method runs on the next event loop cycle, ensuring all widgets
        in the new_tab_widget are fully initialized before being accessed.
        """
        try:
            new_tab_widget.pipeline.load_session(state)
            new_tab_index = self.super_tabs.indexOf(new_tab_widget)
            self.super_tabs.setCurrentIndex(new_tab_index)
            # self.status_bar.update_status(f"Loaded session {file_name}", 5000)
            log.info(f"State loaded into new tab: '{file_name}'")
        except Exception as e:
            log.exception("Failed to load state into the new session's pipeline.")
            QMessageBox.critical(self, "Load Error", f"Could not apply the loaded session state:\n{e}")
            # Clean up the failed tab
            new_tab_index = self.super_tabs.indexOf(new_tab_widget)
            if new_tab_index != -1:
                self.super_tabs.removeTab(new_tab_index)

    def closeEvent(self, event):
        """Saves window geometry upon closing the application."""
        self.settings.setValue("windowGeometry", self.saveGeometry())
        log.info("Saving window geometry and closing application.")
        super().closeEvent(event)

    def _restore_window(self):
        """Restores the window's size and position from the last session."""
        geometry = self.settings.value("windowGeometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1300, 750)  # default size fallback

    @Slot(str)
    def on_update_available(self, new_version):
        """
        Slot called only if the UpdateChecker finds a newer version.
        """
        log.info(f"Update available: {new_version}")

        # Construct a friendly message with a link
        repo_url = "https://github.com/broemere/proper/releases/latest"
        msg = (
            f"A new version of {APP_NAME} is available!<br><br>"
            f"Current version: <b>v{APP_VERSION}</b><br>"
            f"New version: <b>{new_version}</b><br><br>"
            f"Click <a href='{REPO_URL}'>here</a> to view the release page."
        )

        self.show_info_dialog("Update Available", msg)

    @Slot(tuple)
    def show_error_dialog(self, err_tb):
        """Displays a modal dialog for critical errors from background tasks."""
        exc, tb_str = err_tb
        is_user_warning = hasattr(exc, "hint")
        if is_user_warning:
            log.warning(f"Displaying user warning: {exc}")
        else:
            log.error(f"Displaying critical error: {exc}\nTraceback: {tb_str}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning if is_user_warning else QMessageBox.Critical)
        title = "Action Required" if is_user_warning else "An Error Occurred"
        msg_box.setWindowTitle(title)
        msg_box.setText(str(exc))
        msg_box.setInformativeText(getattr(exc, "hint", f"Please check '{APP_NAME.lower()}.log' for details."))
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def show_info_dialog(self, title: str, message: str):
        """Displays a modal informational dialog with a standard icon and rich text."""
        log.info(f"Displaying info dialog: {title}")
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setTextFormat(Qt.RichText)  # parse HTML
        msg_box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        msg_box.setText(f"<b>{title}</b>")
        msg_box.setInformativeText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()
