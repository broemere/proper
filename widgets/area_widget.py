import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal, QPointF
from PySide6.QtGui import QPainter, QPixmap, QBrush, QFont, QColor, QPen, QPainterPath
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
        self.setFixedSize(self.display_pixmap.size())
        self.update()  # Trigger a repaint with the new image

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

        # 2. Draw each marker dot
        for position in self.marker_positions:
            widget_pos = QPointF(self.image_rect.topLeft()) + position
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(Qt.white))
            painter.drawEllipse(widget_pos, 8, 8)
            painter.setBrush(QBrush(Qt.green))
            painter.drawEllipse(widget_pos, 5, 5)

        # 3. If we have 5 markers, calculate and draw position labels
        if len(self.marker_positions) == self.MAX_MARKERS:
            labels = self._calculate_marker_labels(self.marker_positions)
            font = QFont("Arial", 12, QFont.Bold)

            for pos_tuple, label in labels.items():
                position = QPointF(pos_tuple[0], pos_tuple[1])
                widget_pos = QPointF(self.image_rect.topLeft()) + position

                # Use QPainterPath for a clean text outline
                path = QPainterPath()
                # Position the text slightly above the marker dot
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

            # If the list is full, remove the oldest marker (at the front)
            if len(self.marker_positions) >= self.MAX_MARKERS:
                self.marker_positions.pop(0)

            # Add the new marker position to the end of the list
            self.marker_positions.append(QPointF(centroid_x, centroid_y))

            self.update()

    def _calculate_marker_labels(self, positions):
        """
        Calculates marker position labels ("Left", "Right", etc.)
        when exactly 5 markers are present.
        Returns a dictionary mapping a tuple (x,y) to its label.
        """
        if len(positions) != 5:
            return {}

        # Find the markers with extreme centroid values
        left_marker = min(positions, key=lambda p: p.x())
        right_marker = max(positions, key=lambda p: p.x())
        top_marker = min(positions, key=lambda p: p.y())
        bottom_marker = max(positions, key=lambda p: p.y())

        # --- FIXED: Convert QPointF to hashable tuples for set operations ---
        all_markers_set = {(p.x(), p.y()) for p in positions}
        extreme_markers_set = {
            (left_marker.x(), left_marker.y()),
            (right_marker.x(), right_marker.y()),
            (top_marker.x(), top_marker.y()),
            (bottom_marker.x(), bottom_marker.y())
        }

        middle_marker_set = all_markers_set - extreme_markers_set
        middle_marker_tuple = list(middle_marker_set)[0] if middle_marker_set else None

        # --- FIXED: Use tuples as dictionary keys ---
        position_map = {}
        if middle_marker_tuple:
            position_map[middle_marker_tuple] = "Middle"

        position_map[(left_marker.x(), left_marker.y())] = "Left"
        position_map[(right_marker.x(), right_marker.y())] = "Right"
        position_map[(top_marker.x(), top_marker.y())] = "Top"
        position_map[(bottom_marker.x(), bottom_marker.y())] = "Bottom"

        return position_map