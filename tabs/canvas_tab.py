from data_pipeline import DataPipeline
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStyle, QScrollArea, QMessageBox, \
    QSizePolicy
from PySide6.QtGui import QPixmap, QColor, QImage
import numpy as np
from widgets.polygon_canvas import PolygonCanvas


class CanvasTab(QWidget):
    def __init__(self, pipeline: DataPipeline, left_right = None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self.pipeline.register_observer("threshed", self._refresh_canvas)
        self.left_right = left_right
        self._should_center_canvas = False

    def init_ui(self):
        tab_layout = QVBoxLayout(self)

        # Canvas
        self.polygon_canvas = PolygonCanvas()
        self.polygon_canvas.image_flattened.connect(self._flattened_image_received)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.polygon_canvas)
        # self.polygon_layout.addWidget(self.polygon_canvas)
        self.scroll.setWidgetResizable(False)  # the canvas stays its natural size
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tab_layout.addWidget(self.scroll, stretch=1)

        v_bar = self.scroll.verticalScrollBar()
        v_bar.rangeChanged.connect(self.center_on_range_change)

        # Controls row: Refresh + Tool + Final Color + Help
        ctrl_row = QHBoxLayout()
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn.setToolTip("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_canvas)
        ctrl_row.addWidget(self.refresh_btn)

        self.tool_btn = QPushButton("Polygon")
        self.tool_btn.clicked.connect(self._toggle_tool)
        self.tool_btn.setToolTip("Tool")
        self.polygon_canvas.tool_changed.connect(self._toggle_tool)
        ctrl_row.addWidget(self.tool_btn)

        self.color_btn = QPushButton("Fill Color")
        self.color_btn.setStyleSheet(f"background-color: {self.polygon_canvas.final_color.name()}; color: white;")
        self.color_btn.setToolTip("Fill Color. Keyboard Shortcut: [Space]")
        self.color_btn.clicked.connect(self._toggle_final_color)
        self.polygon_canvas.color_changed.connect(self._toggle_final_color)
        ctrl_row.addWidget(self.color_btn)

        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.polygon_canvas.undo_last_polygon)
        ctrl_row.addWidget(undo_btn)

        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self.show_help)
        self.help_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        ctrl_row.addWidget(self.help_btn)

        tab_layout.addLayout(ctrl_row)


    def _refresh_canvas(self, _):
        if self.left_right == "left":
            self._show_canvas_image(self.pipeline.left_threshed)
        else:
            if self.pipeline.right_threshed is not None:
                self._show_canvas_image(self.pipeline.right_threshed)

    def _show_canvas_image(self, img_array: np.ndarray):
        # convert numpy to QPixmap and set as background
        self._should_center_canvas = True
        h, w = img_array.shape[:2]
        bytes_per_line = w
        qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.polygon_canvas.set_background(pix)

    def _toggle_final_color(self, new_color):
        """
        Flip the canvas’s final_color (e.g. black ↔ white), and update
        the button’s stylesheet so it shows the current color.
        """
        if not new_color:
            curr = self.polygon_canvas.final_color
            new_color = QColor(Qt.white) if curr == QColor(Qt.black) else QColor(Qt.black)
            self.polygon_canvas.set_final_color(new_color)
        if new_color.value() == 255:
            text_color = "black"
        else:
            text_color = "white"
        self.color_btn.setStyleSheet(f"background-color: {new_color.name()}; color: {text_color};")

    def _toggle_tool(self, tool):
        current_tool = self.polygon_canvas.current_tool
        if current_tool == "polygon":
            tool = "lasso"
        else:
            tool = "polygon"
        self.tool_btn.setText(tool.title())
        self.polygon_canvas.set_tool(tool)

    def _flattened_image_received(self, img):
        if self.left_right == "left":
            self.pipeline.left_cleaned = img
        else:
            self.pipeline.right_cleaned = img

    def center_on_range_change(self, min_val, max_val):
        """
        This slot is connected to the scrollbar's rangeChanged signal.
        It centers the view only when the flag is set.
        """
        # Only run this logic if it has been "armed" by a new image load
        if self._should_center_canvas:
            # Disarm the flag so this doesn't run again until the next image load
            self._should_center_canvas = False

            h_bar = self.scroll.horizontalScrollBar()
            v_bar = self.scroll.verticalScrollBar()

            # Set the scrollbars to their center positions
            h_bar.setValue(h_bar.maximum() // 2)
            v_bar.setValue(v_bar.maximum() // 2)

    def show_help(self):
        instructions = (
            "Use this tool to clean up noise in the thresholded images.<br>"
            "The objects of interest should be isolated from any background objects/extra bits.<br><br>"
            "There is a polygon tool and lasso tool available.<br><br>"
            "With the polygon tool, click to start drawing a polygon. "
            "To close the polygon, move the cursor near the first point until it turns green, and click again.<br><br>"
            "With the lasso tool, simply click and drag to draw the shape you want to fill in. "
            "It will automatically close the shape when you let go of the mouse button.<br><br>"
            "Black and white colors can be used. Click the Fill Color to change.<br>"
            "The refresh button will clear all shapes (quick undo all).<br><br>"
            "Keyboard shortcuts:<br>"
            "Space: toggle color (black/white)<br>"
            "Escape: cancel current shape being drawn (polygon or lasso)<br>"
            "Ctrl+Z: undo the last shape, or cancel the current shape if currently being drawn<br>"
            "t: Switch tools"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("Help")
        msg.setTextFormat(Qt.RichText)
        msg.setText(instructions)
        msg.setStandardButtons(QMessageBox.Close)
        msg.setDefaultButton(QMessageBox.Close)
        msg.exec()
