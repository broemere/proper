from data_pipeline import DataPipeline
from PySide6.QtCore import Qt, Signal, QPointF, QRect, QSize, QTimer, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QDoubleSpinBox, QSizePolicy,
    QStyle, QCheckBox, QLineEdit, QGridLayout, QGridLayout, QButtonGroup, QSlider, QFrame, QScrollArea
)
from PySide6.QtGui import QPalette, QPixmap, QColor, QPainter, QImage, QPen, QCursor, QIcon
import numpy as np
from processing.data_transform import numpy_to_qpixmap
from widgets.polygon_canvas import PolygonCanvas


class CanvasTab(QWidget):


    def __init__(self, pipeline: DataPipeline, left_right = None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self.pipeline.register_observer("threshed", self._refresh_canvas)
        self.left_right = left_right


    def init_ui(self):
        tab_layout = QVBoxLayout(self)

        # Canvas
        self.polygon_canvas = PolygonCanvas()
        self.polygon_canvas.image_flattened.connect(self._flattened_image_received)
        scroll = QScrollArea()
        scroll.setWidget(self.polygon_canvas)
        # self.polygon_layout.addWidget(self.polygon_canvas)
        scroll.setWidget(self.polygon_canvas)
        scroll.setWidgetResizable(False)  # the canvas stays its natural size
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tab_layout.addWidget(scroll, stretch=1)

        # Controls row: Refresh + Final Color
        ctrl_row = QHBoxLayout()
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn.setToolTip("Refresh")
        ctrl_row.addWidget(self.refresh_btn)

        self.tool_btn = QPushButton("Polygon")
        self.tool_btn.clicked.connect(self._toggle_tool)
        self.tool_btn.setToolTip("Tool")
        #self.polygon_canvas.color_changed.connect(self._toggle_tool)
        ctrl_row.addWidget(self.tool_btn)

        self.color_btn = QPushButton("Fill Color")
        # show current final color as background
        self.color_btn.setStyleSheet(
            f"background-color: {self.polygon_canvas.final_color.name()}; color: white;"
        )
        self.color_btn.setToolTip("Fill Color. Keyboard Shortcut: [Space]")
        ctrl_row.addWidget(self.color_btn)
        self.polygon_canvas.color_changed.connect(self._toggle_final_color)
        self.refresh_btn.clicked.connect(self._refresh_canvas)
        # self.color_btn.clicked.connect(
        #     lambda _=None, cvs=self.polygon_canvas, btn=self.color_btn_first: self._toggle_final_color(cvs, btn)
        # )
        self.color_btn.clicked.connect(self._toggle_final_color)
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.polygon_canvas.undo_last_polygon)
        ctrl_row.addWidget(undo_btn)
        tab_layout.addLayout(ctrl_row)


    def _refresh_canvas(self, _):
        if self.left_right == "left":
            self._show_canvas_image(self.pipeline.left_threshed)
        else:
            if self.pipeline.right_threshed is not None:
                self._show_canvas_image(self.pipeline.right_threshed)

    def _show_canvas_image(self, img_array: np.ndarray):
        # convert numpy to QPixmap and set as background
        h, w = img_array.shape[:2]
        bytes_per_line = w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.polygon_canvas.set_background(pix)

    # def _toggle_final_color(self, canvas: PolygonCanvas, btn: QPushButton):
    #     """
    #     Flip the canvas’s final_color (e.g. black ↔ white), and update
    #     the button’s stylesheet so it shows the current color.
    #     """
    #     curr = canvas.final_color
    #     new = QColor(Qt.white) if curr == QColor(Qt.black) else QColor(Qt.black)
    #     canvas.set_final_color(new)
    #
    #     # Now update the button’s background so the user sees the new color
    #     btn.setStyleSheet(f"background-color: {new.name()};")

    def _toggle_final_color(self, new_color):
        """
        Flip the canvas’s final_color (e.g. black ↔ white), and update
        the button’s stylesheet so it shows the current color.
        """
        if not new_color:
            curr = self.polygon_canvas.final_color
            new_color = QColor(Qt.white) if curr == QColor(Qt.black) else QColor(Qt.black)
            self.polygon_canvas.set_final_color(new_color)
        # Now update the button’s background so the user sees the new color
        if new_color.value() == 255:
            text_color = "black"
        else:
            text_color = "white"
        self.color_btn.setStyleSheet(f"background-color: {new_color.name()}; color: {text_color};")

    def _toggle_tool(self, tool):
        if not tool:
            current_tool = self.polygon_canvas.current_tool
            if current_tool == "polygon":
                tool = "lasso"
            else:
                tool = "polygon"
            self.tool_btn.setText(tool.title())
            self.polygon_canvas.set_tool(tool)

    def _flattened_image_received(self, img):
        print(type(img))

