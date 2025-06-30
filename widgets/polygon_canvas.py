from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen
from PySide6.QtWidgets import QWidget


class PolygonCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # list of polygons: each {'points': [QPoint,...], 'closed': bool, 'color': QColor or None}
        self.polygons = []
        self._start_new_polygon()
        self.current_pos = None
        self.background_pixmap = None
        # default final color used for polygons closed after toggle
        self.final_color = QColor(Qt.black)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_background(self, pixmap: QPixmap):
        """Set a fixed-size background and reset polygons."""
        self.background_pixmap = pixmap
        self.setFixedSize(pixmap.size())  # enforce 1:1 mapping
        self.polygons = []
        self._start_new_polygon()
        self.update()

    def set_final_color(self, color: QColor):
        """Update the default color for future polygons."""
        self.final_color = color
        self.update()

    def _start_new_polygon(self):
        # initialize a fresh polygon with no color until closed
        self.polygons.append({'points': [], 'closed': False, 'color': None})
        self.current_pos = None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = event.pos()
        poly = self.polygons[-1]
        pts = poly['points']

        # if last polygon is already closed, begin a new one
        if poly['closed']:
            self._start_new_polygon()
            poly = self.polygons[-1]
            pts = poly['points']

        if not pts:
            # first point of new polygon
            pts.append(pos)
        else:
            # subsequent point: either close or append
            first = pts[0]
            if len(pts) >= 3 and (pos - first).manhattanLength() < 10:
                poly['closed'] = True
                # assign the color at closure time
                poly['color'] = QColor(self.final_color)
            else:
                pts.append(pos)
        self.update()

    def mouseMoveEvent(self, event):
        if not self.polygons:
            return
        # update live edge only if current poly not closed
        if not self.polygons[-1]['closed']:
            self.current_pos = event.pos()
            self.update()

    def keyPressEvent(self, event):
        # Esc cancels the in-progress polygon only
        if event.key() == Qt.Key_Escape:
            poly = self.polygons[-1]
            if not poly['closed'] and poly['points']:
                self.polygons.pop()
                self._start_new_polygon()
                self.current_pos = None
                self.update()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # background
        if self.background_pixmap:
            painter.drawPixmap(0, 0, self.background_pixmap)

        # draw each polygon
        for poly in self.polygons:
            pts = poly['points']
            if len(pts) < 2:
                continue
            if poly['closed']:
                # fill and stroke with the polygon's own color
                color = poly['color'] or self.final_color
                painter.setBrush(color)
                painter.setPen(QPen(color, 2))
                painter.drawPolygon(pts)
            else:
                # open polygon: red strokes
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(Qt.red, 2))
                for i in range(1, len(pts)):
                    painter.drawLine(pts[i-1], pts[i])

        # draw live edge for the last polygon
        current = self.polygons[-1]
        pts = current['points']
        if not current['closed'] and pts and self.current_pos:
            first = pts[0]
            near = len(pts) >= 3 and (self.current_pos - first).manhattanLength() < 10
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(Qt.green if near else Qt.red, 2))
            painter.drawLine(pts[-1], self.current_pos)

    def undo_last_polygon(self):
        """
        Remove the most recently closed polygon.
        If none are closed, do nothing.
        Always leave one open polygon ready for new drawing.
        """
        # find the last CLOSED polygon
        for idx in range(len(self.polygons) - 1, -1, -1):
            if self.polygons[idx]['closed']:
                self.polygons.pop(idx)
                break
        else:
            # no closed polygon found → nothing to undo
            return

        # if you’ve removed the very last polygon, or the new last is STILL closed,
        # start a brand-new open polygon
        if not self.polygons or self.polygons[-1]['closed']:
            self._start_new_polygon()

        self.update()
