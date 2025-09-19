import os
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QCursor


def resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource, works for dev and for PyInstaller.
    The 'relative_path' should be the path to the resource relative to the
    project's root directory.
    """
    if getattr(sys, "frozen", False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        # If the application is run as a script, find the project root
        # by going up one level from this file's directory (processing/).
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return os.path.join(base_path, relative_path)


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

    path_to_resource = os.path.join("resources", f"{name}{suffix}.png")
    absolute_path = resource_path(path_to_resource)

    pix = QPixmap(absolute_path)
    # Only do a smooth resize on non-integer scales (e.g. 1.5, 2.25)
    if not scale.is_integer():
        pix = pix.scaled(
            int(pix.width() * scale),
            int(pix.height() * scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
    # Tag the pixmap so Qt treats its logical size correctly
    pix.setDevicePixelRatio(scale)
    return QCursor(pix, int(hot_x * scale), int(hot_y * scale))
