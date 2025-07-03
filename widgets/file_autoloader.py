from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QWidget
import logging

log = logging.getLogger(__name__)


def find_and_prompt_for_video(parent: QWidget, csv_path: str) -> str | None:
    """Looks for a matching video file, asks the user, and returns the path or None."""
    video_path = _look_for_video(csv_path)
    if video_path is None:
        return None

    reply = QMessageBox.question(
        parent,
        "Load video?",
        f"Found a matching video file:\n\n{video_path.parent}\n\n{video_path.name}\n\nWould you like to load it?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )

    return str(video_path) if reply == QMessageBox.Yes else None


def find_and_prompt_for_csv(parent: QWidget, video_path: str) -> str | None:
    """Looks for a matching CSV file, asks the user, and returns the path or None."""
    csv_path = _look_for_csv(video_path)
    if csv_path is None:
        return None

    reply = QMessageBox.question(
        parent,
        "Load CSV?",
        f"Found a matching data file:\n\n{csv_path.parent}\n\n{csv_path.name}\n\nWould you like to load it?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )

    return str(csv_path) if reply == QMessageBox.Yes else None


def _look_for_video(path):
    p      = Path(path)
    stems  = [p.stem, p.stem.replace("_pressure", "_video")]
    exts   = [".avi", ".mkv", ".tif"]

    for stem in stems:
        for ext in exts:
            candidate = p.with_stem(stem).with_suffix(ext)
            if candidate.exists():
                log.info(f"Found matching video: {candidate}")
                return candidate

    return None


def _look_for_csv(path):
    p      = Path(path)
    stems  = [p.stem, p.stem.replace("_video", "_pressure")]
    ext   = ".csv"

    for stem in stems:
        candidate = p.with_stem(stem).with_suffix(ext)
        if candidate.exists():
            log.info(f"Found matching csv: {candidate}")
            return candidate

    return None
