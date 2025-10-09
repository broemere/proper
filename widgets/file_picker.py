import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStyle, QWidget, QFileDialog


class FilePickerWidget(QWidget):
    csv_selected = Signal(str)
    video_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        v = QVBoxLayout(self)
        # Top row: CSV picker
        row_csv = QHBoxLayout()
        btn_csv = QPushButton('Import CSV')
        btn_csv.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        btn_csv.clicked.connect(self.choose_csv)
        row_csv.addWidget(btn_csv)
        row_csv.addWidget(QLabel('CSV loaded:'))
        self.file_path = QLabel('')
        self.file_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_csv.addWidget(self.file_path)
        self.file_label = QLabel('')
        self.file_label.setStyleSheet('font-size: 14pt; color: #00a007;')
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_csv.addWidget(self.file_label)
        v.addLayout(row_csv)

        # Second row: Video picker
        row_video = QHBoxLayout()
        btn_video = QPushButton('Import Video')
        btn_video.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        btn_video.clicked.connect(self.choose_video)
        row_video.addWidget(btn_video)
        row_video.addWidget(QLabel('Video loaded:'))
        self.video_path = QLabel('')
        self.video_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_video.addWidget(self.video_path)
        self.video_label = QLabel('')
        self.video_label.setStyleSheet('font-size: 14pt; color: #00a007;')
        self.video_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_video.addWidget(self.video_label)
        v.addLayout(row_video)

    def choose_csv(self):
        last_dir = self.settings.value("last_dir", "") or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", last_dir,
                                               "CSV Files (*.csv);;All Files (*)")
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))
            self.csv_selected.emit(path)

    def choose_video(self):
        last_dir = self.settings.value("last_dir", "") or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select Video file", last_dir,
                                               "Video Files (*.avi *.mkv *.tif *.tiff);;All Files (*)")
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))
            self.video_selected.emit(path)

            # self.pipeline.video = path
            # self._init_video_tab()
            # self._get_frames()

    def set_csv_label(self, path):
        self.file_path.setText(str(Path(path).parent)+os.path.sep)
        self.file_label.setText(Path(path).name)

    def set_video_label(self, path):
        self.video_path.setText(str(Path(path).parent)+os.path.sep)
        self.video_label.setText(Path(path).name)
