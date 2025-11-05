from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, QMessageBox, QSizePolicy
import numpy as np
import sys
# Assuming your PolygonCanvas is in a 'widgets' sub-directory.
# Adjust the import path if necessary.
from .polygon_canvas import PolygonCanvas


class CanvasWindow(QWidget):
    """
    A standalone window for drawing on a pixmap using PolygonCanvas.
    It encapsulates the canvas and all its associated UI controls.
    """
    # Signal to send the final image back when the user saves.
    drawing_completed = Signal(np.ndarray)

    def __init__(self, background_pixmap: QPixmap, parent=None):
        """
        Initializes the window with a base image to draw on.

        Args:
            base_pixmap (QPixmap): The image to be used as the canvas background.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        # Make this a top-level window instead of being embedded in a parent.
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Drawing Canvas")
        # Set a reasonable default size for the window.
        self.showMaximized()
        self.platform = 'mac' if sys.platform == 'darwin' else 'win'
        self.ctrl_key = "Cmd" if self.platform == "mac" else "Ctrl"
        self.init_ui()

        # Load the provided pixmap into the canvas.
        self.polygon_canvas.set_pixmap(background_pixmap)



    def init_ui(self):
        """
        Builds the user interface, including the canvas and control buttons.
        """
        main_layout = QVBoxLayout(self)

        # The canvas is placed inside a scroll area to handle large images.
        self.polygon_canvas = PolygonCanvas()
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.polygon_canvas)
        self.scroll.setWidgetResizable(False)  # Important for scrolling a fixed-size canvas
        main_layout.addWidget(self.scroll, stretch=1)

        v_bar = self.scroll.verticalScrollBar()
        v_bar.rangeChanged.connect(self.center_canvas)

        # --- Drawing Controls ---
        ctrl_row = QHBoxLayout()
        undo_btn = QPushButton(f"Undo [{self.ctrl_key}+Z]")
        undo_btn.setToolTip("Undo last polygon (Ctrl+Z)")
        undo_btn.clicked.connect(self.polygon_canvas.undo_last_polygon)
        ctrl_row.addWidget(undo_btn)

        self.tool_btn = QPushButton("Lasso [T]")
        self.tool_btn.setToolTip("Change drawing tool (T)")
        self.tool_btn.clicked.connect(self._toggle_tool)
        self.polygon_canvas.tool_changed.connect(self._toggle_tool)
        ctrl_row.addWidget(self.tool_btn)

        self.color_btn = QPushButton()
        self.color_btn.setToolTip("Change fill color (Spacebar)")
        self.color_btn.clicked.connect(self._toggle_final_color)
        self.polygon_canvas.color_changed.connect(self._update_color_button)
        ctrl_row.addWidget(self.color_btn)

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.show_help)
        help_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        ctrl_row.addWidget(help_btn)
        main_layout.addLayout(ctrl_row)

        # --- Window Actions ---
        action_row = QHBoxLayout()
        action_row.addStretch()  # Push buttons to the right

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        action_row.addWidget(cancel_button)

        save_button = QPushButton("Save and Close")
        save_button.setDefault(True)  # Activated on Enter press
        save_button.clicked.connect(self._save_and_close)
        action_row.addWidget(save_button)
        main_layout.addLayout(action_row)

        # Set initial state for buttons
        self._update_color_button(self.polygon_canvas.final_color)
        self.tool_btn.setText(self.polygon_canvas.current_tool.title() + " [T]")

    def _toggle_tool(self):
        """Switches between 'polygon' and 'lasso' tools."""
        new_tool = 'polygon' if self.polygon_canvas.current_tool == 'lasso' else 'lasso'
        self.polygon_canvas.set_tool(new_tool)
        self.tool_btn.setText(new_tool.title() + " [T]")

    def _toggle_final_color(self):
        """Switches between black and white fill colors."""
        new_color = QColor(Qt.white) if self.polygon_canvas.final_color == QColor(Qt.black) else QColor(Qt.black)
        self.polygon_canvas.set_final_color(new_color)
        self._update_color_button(new_color)

    def _update_color_button(self, color: QColor):
        """Updates the color button's appearance."""
        fill_color = "white" if color.value() > 127 else "black"
        text_color = "black" if color.value() > 127 else "white"
        self.color_btn.setText(f"Fill: {fill_color.title()} [Space]")
        self.color_btn.setStyleSheet(f"background-color: {color.name()}; color: {text_color};")

    def _save_and_close(self):
        """Retrieves the final drawing, emits it, and closes the window."""
        final_image_array = self.polygon_canvas.get_flattened_image()
        if final_image_array is not None:
            self.drawing_completed.emit(final_image_array)
        self.close()

    def center_canvas(self, min_val, max_val):
        """
        This slot is connected to the scrollbar's rangeChanged signal.
        It centers the view only when the flag is set.
        """
        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()

        # Set the scrollbars to their center positions
        h_bar.setValue(h_bar.maximum() // 2)
        v_bar.setValue(v_bar.maximum() // 2)

    def show_help(self):
        """Displays a help message box with instructions."""
        instructions = (
            "<h3>Drawing Controls</h3>"
            "<ul>"
            "<li><b>Lasso Tool:</b> Click and drag to draw a freeform shape. Release to close.</li>"
            "<li><b>Polygon Tool:</b> Click to place points. Click near the start point to close.</li>"
            "<li><b>Toggle Tool (T):</b> Switch between Lasso and Polygon tools.</li>"
            "<li><b>Toggle Color (Spacebar):</b> Switch fill color between black and white.</li>"
            "<li><b>Undo (Ctrl+Z):</b> Remove the last completed shape.</li>"
            "<li><b>Cancel Drawing (Esc):</b> Cancel the shape you are currently drawing.</li>"
            "</ul>"
        )
        QMessageBox.information(self, "Help", instructions)
