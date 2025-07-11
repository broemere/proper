from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QFrame, QVBoxLayout, QPushButton


class AutoResizeImage(QFrame):
    """
    A QFrame that automatically resizes a QPixmap to fit its size while maintaining
    aspect ratio. It can optionally display a button overlaid in the top-right corner.
    """
    # Signal emitted when the overlay button is clicked
    button_clicked = Signal()

    def __init__(self, text: str = None, show_button: bool = False, parent=None):
        """
        Args:
            text (str, optional): Initial text to display. Defaults to None.
            show_button (bool, optional): Whether to show the overlay button. Defaults to False.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(0, 0)
        self.show_button = show_button

        # The internal label will hold the scaled pixmap
        self.image_label = QLabel(text)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Use a layout to make the label fill the frame automatically
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        # Create the button if requested, parenting it to this widget for manual positioning
        self.action_button = None
        if show_button:
            self.action_button = QPushButton("Draw", self)
            self.action_button.setStyleSheet(
                "QPushButton { "
                "  background-color: rgba(255, 255, 255, 150); "
                "  color: black; "
                "  font-weight: bold; "
                "  border: 1px solid black;"
                "  border-radius: 5px; "
                "  padding: 5px; "
                "} "
                "QPushButton:hover { "
                "  background-color: rgba(255, 255, 255, 200); "
                "}"
            )
            self.action_button.setToolTip("Perform action on this image")
            # Connect the button's click to our custom signal
            self.action_button.clicked.connect(self.button_clicked.emit)
            self.action_button.setVisible(False)

    def setPixmap(self, pixmap: QPixmap):
        """
        Sets the pixmap to be displayed. The image will be scaled to fit.

        Args:
            pixmap (QPixmap): The full-resolution pixmap.
        """
        self._pixmap = pixmap
        if self.show_button:
            self.action_button.setVisible(True)
        self._update_scaled_pixmap()

    def pixmap(self) -> QPixmap | None:
        """
        Returns the original, unscaled QPixmap.

        Returns:
            QPixmap | None: The original pixmap, or None if not set.
        """
        return self._pixmap

    def _update_scaled_pixmap(self):
        """
        Internal method to scale the pixmap and set it on the label.
        """
        if self._pixmap:
            # Scale the pixmap to the current size of this QFrame widget
            scaled_pixmap = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """
        Handles resizing of the widget to rescale the image and reposition the button.
        """
        super().resizeEvent(event)
        self._update_scaled_pixmap()

        # Reposition the button to keep it in the top-right corner
        if self.action_button:
            margin = 5  # 5px margin from the edges
            button_size = self.action_button.size()
            x_pos = self.width() - button_size.width() - margin
            y_pos = margin
            self.action_button.move(x_pos, y_pos)