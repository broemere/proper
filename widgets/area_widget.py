import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout


class AreaAnalysisWidget(QWidget):
    """A widget to display a pixmap and analyze labeled regions on click."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.display_pixmap = None
        self.labeled_array = None
        self.image_rect = QRect()

    def set_data(self, pixmap: QPixmap, labeled_array: np.ndarray):
        """Sets the visual pixmap and the data array for analysis."""
        self.display_pixmap = pixmap
        self.labeled_array = labeled_array
        self.update()  # Trigger a repaint with the new image

    def paintEvent(self, event):
        """Draws the pixmap centered in the widget."""
        if not self.display_pixmap:
            return

        painter = QPainter(self)
        widget_rect = self.rect()

        # Calculate the centered position for the pixmap
        x = (widget_rect.width() - self.display_pixmap.width()) / 2
        y = (widget_rect.height() - self.display_pixmap.height()) / 2

        # Store the drawing rectangle for use in mouse events
        self.image_rect = QRect(QPoint(x, y), self.display_pixmap.size())

        painter.drawPixmap(self.image_rect, self.display_pixmap)

    def mousePressEvent(self, event):
        """Handles clicks to calculate the area of the clicked segment."""
        # Ensure we have data and the click is a left-click
        if self.labeled_array is None or not self.image_rect.contains(event.pos()):
            return

        # Convert widget coordinates to image coordinates
        image_pos = event.pos() - self.image_rect.topLeft()
        x, y = image_pos.x(), image_pos.y()

        # Get the label ID of the clicked pixel
        blob_id = self.labeled_array[y, x]

        # 0 is the background, so we only care about non-zero labels
        scalar_id = blob_id.item()
        if scalar_id > 0:
            # Use numpy to efficiently count all pixels with the same label ID
            area = np.sum(self.labeled_array == scalar_id)
            print(f"Clicked blob with ID {scalar_id}. Area: {area} pixels")
        else:
            print("Clicked background.")