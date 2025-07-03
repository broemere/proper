import os
import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui     import QPixmap, QCursor


def resource_path(*path_parts: str) -> str:
    """
    Return the absolute path to a bundled resource:
      - If frozen by PyInstaller, base is sys._MEIPASS
      - Otherwise base is this file’s directory
    """
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(__file__)
    return os.path.join(base, *path_parts)

def load_cursor(name: str, hot_x: int, hot_y: int) -> QCursor:
    """
    Loads “name.png” or “name@2x.png” from resources/cursors,
    picks based on primary screen DPI, and returns a QCursor.
    Must be called after QApplication() exists.
    """
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("Call load_cursor() only after QApplication() is created")

    scale = app.primaryScreen().devicePixelRatio()
    suffix = "@2x" if scale >= 2.0 else ""
    path   = resource_path("..", "resources", f"{name}{suffix}.png")

    pix = QPixmap(path)
    # Only do a smooth resize on non-integer scales (e.g. 1.5, 2.25)
    if not scale.is_integer():
        pix = pix.scaled(
            pix.width()  * scale,
            pix.height() * scale,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    # Tag the pixmap so Qt treats its logical size correctly
    pix.setDevicePixelRatio(scale)
    return QCursor(pix, int(hot_x * scale), int(hot_y * scale))
