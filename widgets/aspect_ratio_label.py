from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QSize


class AspectRatioLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pixmap = None
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def setPixmap(self, pixmap: QPixmap):
        # store the original
        self._pixmap = pixmap
        super().setPixmap(pixmap)

    def minimumSizeHint(self) -> QSize:
        # allow the layout to shrink us down to 0×0 if it wants
        return QSize(0, 0)

    def resizeEvent(self, event):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
        super().resizeEvent(event)
