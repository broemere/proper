from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar, QPushButton, QStatusBar


class StatusBarWidget(QWidget):
    cancel_clicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)

        status_layout = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setAlignment(Qt.AlignLeft)
        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.cancel_clicked)
        self.btn_cancel.setEnabled(False)
        status_layout.addWidget(self.status)
        status_layout.addStretch()
        status_layout.addWidget(self.btn_cancel)
        v.addLayout(status_layout)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("QProgressBar{border:1px solid gray; border-radius:5px;}"+
                                   "QProgressBar::chunk{background-color:#4CAF50}")
        self.progress.setAlignment(Qt.AlignCenter)
        v.addWidget(self.progress)

    # Public slots for other parts of the app to call
    @Slot(str)
    def update_status(self, text):
        self.status.setText(text)

    @Slot(int)
    def update_progress(self, value):
        self.progress.setValue(value)

    @Slot(bool)
    def set_cancel_enabled(self, enabled):
        self.btn_cancel.setEnabled(enabled)

    @Slot(bool)
    def batch_finished(self):
        self.status.setText("Ready")
        self.progress.setValue(0)