import logging
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget, QPushButton, QStyle
from data_pipeline import DataPipeline
from widgets.error_bus import user_error

log = logging.getLogger(__name__)


class StiffnessTab(QWidget):
    """
    A widget tab for visualizing the effect of spline smoothing on data.

    This tab displays the original 'zeroed' data alongside a version smoothed
    using a B-spline representation. A slider and spinbox allow for interactive
    adjustment of the spline's smoothing factor 's'.
    """

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.t_data = None  # To store time data (in seconds)
        self.p_data = None  # To store pressure data
        self.pending_user_error = None
        self.last_plotted_version = -1

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initializes the user interface of the tab."""
        pg.setConfigOptions(antialias=True)
        layout = QVBoxLayout(self)
        self.plot_widget = self._create_plot_widget("<b>Spline Smoothing Transformation</b>")
        self.plot_widget.addLegend()

        # --- Curve Styling ---
        original_color = self.palette().color(QPalette.WindowText)
        original_color.setAlpha(100)
        spline_color = QColor("#E74C3C")  # reddish-orange

        # --- Plot Curves ---
        self.original_curve = self.plot_widget.plot(
            [],
            [],
            pen=pg.mkPen(original_color, width=1.5),
            name="Original Smoothed",
        )
        self.spline_curve = self.plot_widget.plot([], [], pen=pg.mkPen(spline_color, width=2.5), name="Spline")
        self.original_curve.setZValue(0)
        self.spline_curve.setZValue(1)  # spline curve is always drawn on top of the original
        self.plot_widget.setLabel("left", "<b>Stress [kPa]</b>")
        self.plot_widget.setLabel("bottom", "<b>Stretch [-]</b>")
        layout.addWidget(self.plot_widget)

        # --- Interactive Controls ---
        controls_layout = QHBoxLayout()
        self.refresh_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "")
        self.refresh_btn.setToolTip("Reload graph view.")
        self.refresh_btn.clicked.connect(self._refresh_data)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(QLabel("Smoothing Factor (s):"))
        # Slider to control the 's' parameter
        self.s_slider = QSlider(Qt.Horizontal)
        self.s_slider.setMinimum(0)
        self.s_slider.setMaximum(500000)  # 's' can be large, this is a starting range
        self.s_slider.setValue(10000)  # default
        self.s_slider.setTickInterval(50000)
        self.s_slider.setTickPosition(QSlider.TicksBelow)
        controls_layout.addWidget(self.s_slider)
        # SpinBox linked to the slider
        self.s_spinbox = QSpinBox()
        self.s_spinbox.setMinimum(0)
        self.s_spinbox.setMaximum(500000)
        self.s_spinbox.setValue(10000)
        self.s_spinbox.setFixedWidth(80)  # Accommodate larger numbers
        controls_layout.addWidget(self.s_spinbox)
        layout.addLayout(controls_layout)

    def connect_signals(self):
        # Connect signals for synchronization and updates
        self.s_slider.valueChanged.connect(self._slider_value_changed)
        self.s_spinbox.valueChanged.connect(self._spinbox_value_changed)

    def showEvent(self, event):
        """Called when the widget becomes visible."""
        super().showEvent(event)

        if self.last_plotted_version != self.pipeline.data_version:
            log.info("Data is stale. Refreshing StiffnessTab.")
            self._refresh_data()

        if self.pending_user_error:
            QTimer.singleShot(0, self._show_pending_user_error)  # Hold the error until the tab has rendered

    def _show_pending_user_error(self):
        user_error(self.pending_user_error[0], self.pending_user_error[1])
        self.original_curve.clear()
        self.spline_curve.clear()

    def _create_plot_widget(self, title: str) -> pg.PlotWidget:
        """Helper to create a plot widget consistent with the app's theme."""
        fg_color = self.palette().color(QPalette.WindowText).name()
        bg_color = self.palette().color(QPalette.Window).name()
        plot = pg.PlotWidget(title=title, color=fg_color, size="12pt")
        plot.setBackground(bg_color)
        plot.showGrid(x=False, y=True, alpha=0.3)
        for axis in ("bottom", "left"):
            ax = plot.getAxis(axis)
            ax.setPen(pg.mkPen(fg_color))
            ax.setTextPen(fg_color)
        # Add zero-lines for reference
        pen = pg.mkPen(color=fg_color, style=Qt.DotLine)
        plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pen))
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pen))
        return plot

    def run_spline_transform(self):
        """Performs the spline calculation using the current 's' value and updates the transformed data curve."""
        s_value = self.s_slider.value()
        log.debug(f"Requesting spline transform with s={s_value}")
        spline_y_values = self.pipeline.calculate_spline(s_value)
        if spline_y_values is not None:
            self.spline_curve.setData(self.pipeline.stretch, spline_y_values)
        else:
            self.spline_curve.clear()

    def _slider_value_changed(self, value: int):
        """Synchronizes the spinbox to the slider and triggers a recalculation."""
        self.s_spinbox.blockSignals(True)
        self.s_spinbox.setValue(value)
        self.s_spinbox.blockSignals(False)
        self.run_spline_transform()

    def _spinbox_value_changed(self, value: int):
        """Synchronizes the slider to the spinbox and triggers a recalculation."""
        self.s_slider.blockSignals(True)
        self.s_slider.setValue(value)
        self.s_slider.blockSignals(False)
        self.run_spline_transform()

    def _refresh_data(self):
        self.pending_user_error = None
        if self.pipeline.conversion_factor == 0:
            self.pending_user_error = ("Missing conversion factor",
                           "Go to the Scale tab and ensure the conversion factor is set.")
            return
        if len(self.pipeline.area_data_left)+len(self.pipeline.area_data_right) < 10:
            self.pending_user_error = ("Area analysis incomplete",
                           "Go to the Area tab and ensure all 5 blobs have been selected for each frame.")
            return
        if len(self.pipeline.thickness_data) == 0:
            self.pending_user_error = ("Thickness analysis incomplete",
                           "Go to the Thickness tab and ensure the thickness has been measured at least once.")
            return

        self.pipeline.get_stress_stretch()
        self._update_plot_and_controls()
        self.last_plotted_version = self.pipeline.data_version

    def _update_plot_and_controls(self):
        # Handle cases with no data
        if self.pipeline.stretch.size <= 1:
            log.warning("SplineTab received empty data array.")
            self.original_curve.clear()
            self.spline_curve.clear()
            return

        self._configure_y_axis()
        self._configure_controls()
        self._configure_original_plot()
        self.run_spline_transform()  # Re-run the spline calculation with the new data

    def _configure_y_axis(self):
        """Calculates and sets custom major ticks for the Y-axis (Stress)."""
        y_axis = self.plot_widget.getAxis('left')
        y_min, y_max = np.min(self.pipeline.stress), np.max(self.pipeline.stress)
        # Find the first multiple of 5 at or above the minimum value
        start_tick = np.ceil(y_min / 5) * 5
        tick_values = np.arange(start_tick, y_max, 5)
        # Format for setTicks: a list of (value, label) tuples
        major_ticks = [(tick, f"{int(tick)}") for tick in tick_values]
        y_axis.setTicks([major_ticks])

    def _configure_controls(self):
        """Updates the range and value of the smoothing slider and spinbox."""
        new_max = self.pipeline.stretch.size
        log.info(f"Setting slider/spinbox maximum to new data length: {new_max}")
        self.s_slider.blockSignals(True)  # Block signals to prevent the UI from firing redundant updates
        self.s_spinbox.blockSignals(True)
        try:
            self.s_slider.setMaximum(new_max)
            self.s_spinbox.setMaximum(new_max)
            if self.last_plotted_version == -1:
                default_value = int(round(new_max / 10))
                self.s_slider.setValue(default_value)
                self.s_spinbox.setValue(default_value)
            tick_interval = new_max // 10
            self.s_slider.setTickInterval(max(1, tick_interval))  # update the slider's visual tick interval
        finally:
            self.s_slider.blockSignals(False)
            self.s_spinbox.blockSignals(False)

    def _configure_original_plot(self):
        """Sets the data for the original plot curves and adjusts the X-axis range."""
        x_data, y_data = self.pipeline.stretch, self.pipeline.stress
        self.original_curve.setData(x_data, y_data)
        # Set the x-range manually, leaving the y-axis to autorange
        min_x, max_x = x_data.min(), x_data.max()
        padding = (max_x - min_x) * 0.05
        self.plot_widget.setXRange(min_x - padding, max_x + padding, padding=0)
        self.plot_widget.enableAutoRange(axis='y')
