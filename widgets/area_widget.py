import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QPointF
from PySide6.QtGui import QPainter, QPixmap, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout


class AreaAnalysisWidget(QWidget):
    """A widget to display a pixmap and analyze labeled regions on click."""
    area_clicked = Signal(list)
    def __init__(self, MAX_MARKERS=0, parent=None):
        super().__init__(parent)
        self.display_pixmap = None
        self.labeled_array = None
        self.image_rect = QRect()
        self.marker_positions = []
        self.MAX_MARKERS = MAX_MARKERS

    def set_data(self, pixmap: QPixmap, labeled_array: np.ndarray):
        """Sets the visual pixmap and the data array for analysis."""
        self.display_pixmap = pixmap
        self.labeled_array = labeled_array
        self.marker_positions.clear()
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

        # 1. Draw the base image
        painter.drawPixmap(self.image_rect, self.display_pixmap)

        # 2. Iterate through the list and draw each marker
        painter.setRenderHint(QPainter.Antialiasing)
        for position in self.marker_positions:
            widget_pos = QPointF(self.image_rect.topLeft()) + position
            # Draw white outline
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(Qt.white))
            painter.drawEllipse(widget_pos, 8, 8)

            # Draw green dot
            painter.setBrush(QBrush(Qt.green))
            painter.drawEllipse(widget_pos, 5, 5)

    def mousePressEvent(self, event):
        """Handles clicks to calculate the area of the clicked segment."""
        # Ensure we have data and the click is a left-click
        if self.labeled_array is None or not self.image_rect.contains(event.pos()):
            return

        # Convert widget coordinates to image coordinates
        image_pos = event.pos() - self.image_rect.topLeft()
        x, y = image_pos.x(), image_pos.y()
        blob_id = self.labeled_array[y, x]  # Get the label ID of the clicked pixel
        scalar_id = blob_id.item()

        # A blob_id of 0 is the background, so we only process actual blobs.
        if scalar_id > 0:
            # Find all pixels belonging to the clicked blob
            coords = np.where(self.labeled_array == scalar_id)

            # 1. Calculate the area (total number of pixels)
            area = coords[0].size

            # 2. Calculate the centroid (mean of pixel coordinates)
            # coords[0] contains all y-coordinates (rows)
            # coords[1] contains all x-coordinates (columns)
            centroid_y = np.mean(coords[0])
            centroid_x = np.mean(coords[1])

            # 3. Package the data into a list [area, centroid_x, centroid_y]
            # Note: We cast to float for consistent data types.
            blob_properties = [scalar_id, float(area), float(centroid_x), float(centroid_y)]

            # 4. Emit the signal with the packaged data
            self.area_clicked.emit(blob_properties)

            # If the list is full, remove the oldest marker (at the front)
            if len(self.marker_positions) >= self.MAX_MARKERS:
                self.marker_positions.pop(0)

            # Add the new marker position to the end of the list
            self.marker_positions.append(QPointF(centroid_x, centroid_y))

            self.update()