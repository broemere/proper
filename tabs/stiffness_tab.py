import logging
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget
from scipy.interpolate import make_splrep, splev
from data_pipeline import DataPipeline

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

        self.init_ui()

        # Register to receive the 'zeroed' data whenever it's updated
        # The "transformed" signal from the pipeline carries [zeroed_data, smoothed_data]
        self.pipeline.register_observer("transformed", self._on_data_updated)

    def init_ui(self):
        """Initializes the user interface of the tab."""
        # --- Basic Plot Setup ---
        pg.setConfigOptions(antialias=True)
        layout = QVBoxLayout(self)
        self.plot_widget = self._create_plot_widget(
            "<b>Spline Smoothing Transformation</b>"
        )
        self.plot_widget.addLegend()

        # --- Curve Styling ---
        # A semi-transparent color for the original data curve
        original_color = self.palette().color(QPalette.WindowText)
        original_color.setAlpha(100)
        # A distinct, vibrant color for the transformed spline curve
        spline_color = QColor("#E74C3C")  # A nice reddish-orange

        # --- Plot Curves ---
        self.original_curve = self.plot_widget.plot(
            [],
            [],
            pen=pg.mkPen(original_color, width=1.5),
            name="Original Smoothed",
        )
        self.spline_curve = self.plot_widget.plot(
            [], [], pen=pg.mkPen(spline_color, width=2.5), name="Spline Smoothed"
        )

        # Ensure the spline curve is always drawn on top of the original
        self.original_curve.setZValue(0)
        self.spline_curve.setZValue(1)

        self.plot_widget.setLabel("left", "<b>Pressure [mmHg]</b>")
        self.plot_widget.setLabel("bottom", "<b>Time [min]</b>")
        layout.addWidget(self.plot_widget)

        # --- Interactive Controls ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Smoothing Factor (s):"))

        # Slider to control the 's' parameter
        self.s_slider = QSlider(Qt.Horizontal)
        self.s_slider.setMinimum(0)
        self.s_slider.setMaximum(500000)  # 's' can be large, this is a starting range
        self.s_slider.setValue(10000)  # A sensible default
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

        # Connect signals for synchronization and updates
        self.s_slider.valueChanged.connect(self._slider_value_changed)
        self.s_spinbox.valueChanged.connect(self._spinbox_value_changed)

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

    def _on_data_updated(self, data: list):
        """Slot to receive new data from the pipeline and update the plot."""
        smoothed_data = data[1]  # Unpack the zeroed data from the signal
        self.t_data = np.array(smoothed_data.get("t", []))
        self.p_data = np.array(smoothed_data.get("p", []))

        # Handle cases with no data
        if self.t_data.size == 0:
            log.warning("SplineTab received empty data array.")
            self.original_curve.clear()
            self.spline_curve.clear()
            return

        y_axis = self.plot_widget.getAxis('left')
        if self.p_data.size > 1:
            y_min = np.min(self.p_data)
            y_max = np.max(self.p_data)

            # Find the first multiple of 5 at or above the minimum value
            start_tick = np.ceil(y_min / 5) * 5

            # Generate tick values every 5 units up to the max value
            tick_values = np.arange(start_tick, y_max, 5)

            # Format for setTicks: a list containing one list of (value, label) tuples
            major_ticks = [(tick, f"{int(tick)}") for tick in tick_values]
            y_axis.setTicks([major_ticks])

        new_max = self.t_data.size
        log.info(f"Setting slider/spinbox maximum to new data length: {new_max}")

        tick_interval = new_max // 10
        self.s_slider.setTickInterval(max(1, tick_interval))

        # Block signals to prevent valueChanged from firing prematurely
        self.s_slider.blockSignals(True)
        self.s_spinbox.blockSignals(True)
        try:
            # Set the new maximum for both widgets
            self.s_slider.setMaximum(new_max)
            self.s_spinbox.setMaximum(new_max)
            self.s_spinbox.setValue(int(round(new_max/10)))
            self.s_slider.setValue(int(round(new_max/10)))
            # The widgets will automatically clamp the current value if it's
            # now out of the new [min, max] range.
        finally:
            # Always unblock signals, even if an error occurs
            self.s_slider.blockSignals(False)
            self.s_spinbox.blockSignals(False)

        log.info(f"SplineTab received new data with {self.t_data.size} points.")

        # Plot data with time axis in minutes
        t_minutes = self.t_data / 60.0
        self.original_curve.setData(t_minutes, self.p_data)
        self.plot_widget.autoRange() # Ensure plot is visible

        # Re-run the spline calculation with the new data
        self.run_spline_transform()

    def run_spline_transform(self):
        """
        Performs the spline calculation using the current 's' value and
        updates the transformed data curve.
        """
        # Scipy's make_splrep requires at least k+1 (i.e., 4) data points for a cubic spline
        if self.t_data is None or self.t_data.size < 4:
            self.spline_curve.clear()
            return

        s_value = self.s_slider.value()
        log.debug(f"Running spline transform with s={s_value}")

        try:
            # make_splrep requires the x-data (time) to be strictly increasing
            # This should already be true for time-series data.
            tck = make_splrep(self.t_data, self.p_data, k=3, s=s_value)
            p_spline = splev(self.t_data, tck)

            t_minutes = self.t_data / 60.0
            self.spline_curve.setData(t_minutes, p_spline)
            self.pipeline.p_spline = p_spline
        except Exception as e:
            # Catch potential errors from the spline algorithm
            log.error(f"Error during spline calculation: {e}")
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