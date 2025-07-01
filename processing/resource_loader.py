import os
import sys
from pathlib import Path
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

    # Determine scale (1 or 2+)
    scale = int(app.primaryScreen().devicePixelRatio())

    # Pick 32×32 vs 64×64
    suffix = "@2x" if scale > 1 else ""
    fname  = f"{name}{suffix}.png"

    # Build the full path via resource_path
    png_path = resource_path("..", "resources", fname)
    # if your data_loader is in processing/, adjust the .. accordingly

    pix = QPixmap(png_path)
    pix.setDevicePixelRatio(scale)
    return QCursor(pix, hot_x * scale, hot_y * scale)
