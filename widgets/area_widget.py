import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QPointF
from PySide6.QtGui import QPainter, QPixmap, QBrush, QFont, QColor, QPen, QPainterPath
from PySide6.QtWidgets import QWidget, QVBoxLayout


class AreaAnalysisWidget(QWidget):
    """A widget to display a pixmap and analyze labeled regions on click."""
    area_clicked = Signal(list) # Emits raw [id, area, cx, cy] on click
    def __init__(self, parent=None):
        super().__init__(parent)
        self.display_pixmap = None
        self.labeled_array = None
        self.image_rect = QRect()
        self.blob_data_to_draw = {} # Will store { (cx, cy): "Label" }

    def set_image_data(self, pixmap: QPixmap, labeled_array: np.ndarray):
        """Sets the base visual pixmap and the data array for analysis."""
        self.display_pixmap = pixmap
        self.labeled_array = labeled_array
        self.setFixedSize(self.display_pixmap.size())
        # Call update to ensure the new base image is drawn immediately
        self.update()

    def update_markers(self, blob_data: dict):
        """Updates only the markers and labels to be drawn on the canvas."""
        self.blob_data_to_draw.clear()
        for blob_id, props in blob_data.items():
            # props format is [area, cx, cy, label]
            cx, cy = props[1], props[2]
            label = props[3]
            self.blob_data_to_draw[(cx, cy)] = label
        # Trigger a repaint to show the new markers
        self.update()

    def paintEvent(self, event):
        """Draws the pixmap, markers, and position labels."""
        if not self.display_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        widget_rect = self.rect()

        # Calculate the centered position for the pixmap
        x = (widget_rect.width() - self.display_pixmap.width()) / 2
        y = (widget_rect.height() - self.display_pixmap.height()) / 2
        self.image_rect = QRect(QPoint(x, y), self.display_pixmap.size())

        # 1. Draw the base image
        painter.drawPixmap(self.image_rect, self.display_pixmap)

        font = QFont("Arial", 12, QFont.Bold)
        for pos_tuple, label in self.blob_data_to_draw.items():
            position = QPointF(pos_tuple[0], pos_tuple[1])
            widget_pos = QPointF(self.image_rect.topLeft()) + position

            # Draw marker dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(Qt.white))
            painter.drawEllipse(widget_pos, 8, 8)
            painter.setBrush(QBrush(Qt.green))
            painter.drawEllipse(widget_pos, 5, 5)

            # Draw label text if it exists
            if label:
                path = QPainterPath()
                text_pos = widget_pos + QPointF(-15, -15)
                path.addText(text_pos, font, label)
                # Draw the white outline (stroke)
                pen = QPen(QColor("white"), 2, Qt.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)

                # Draw the black text (fill)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor("black"))
                painter.drawPath(path)

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
