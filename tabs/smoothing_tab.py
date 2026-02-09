import logging
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget, QPushButton, QStyle, QCheckBox
from data_pipeline import DataPipeline
from widgets.error_bus import user_error
from widgets.user_messages import HELP_CONTENT

log = logging.getLogger(__name__)


class SmoothingTab(QWidget):
    """
    A widget tab for visualizing the effect of spline smoothing on data.

    This tab displays the original 'zeroed' data alongside a version smoothed
    using a B-spline representation. A slider and spinbox allow for interactive
    adjustment of the spline's smoothing factor 's'.
    """

    help_requested = Signal(str, str)

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
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
        self.interest_points = self.plot_widget.plot(
            [], [],
            pen=None,  # No connecting line
            symbol='o',  # Use circles as markers
            symbolBrush=None,  # Fill color of the circles
            symbolPen='k',  # Outline color of the circles (black)
            symbolSize=7,
            name="Pressures of Interest"
        )
        self.original_curve.setZValue(1)
        self.spline_curve.setZValue(2)  # spline curve is always drawn on top of the original
        self.interest_points.setZValue(3)  # Ensure points are drawn on top

        self.tangent_lines = []
        max_tangents = 10  # Cap at 10 to be safe
        tangent_pen = pg.mkPen('k', width=2)
        for _ in range(max_tangents):
            line = pg.PlotCurveItem(pen=tangent_pen)
            line.setZValue(0)  # Above everything else
            line.setVisible(False)
            self.plot_widget.addItem(line)
            self.tangent_lines.append(line)


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

        # --- Toggles ---
        self.show_spline_cb = QCheckBox("Spline")
        self.show_spline_cb.setChecked(True)
        self.show_tangents_cb = QCheckBox("Tangents")
        self.show_tangents_cb.setChecked(True)

        controls_layout.addWidget(self.show_spline_cb)
        controls_layout.addWidget(self.show_tangents_cb)

        self.help_btn = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion), "")
        self.help_btn.setToolTip("Smoothing tab info")
        title, msg = HELP_CONTENT["smoothing"]
        self.help_btn.clicked.connect(lambda: self.help_requested.emit(title, msg))
        controls_layout.addWidget(self.help_btn)
        layout.addLayout(controls_layout)

    def connect_signals(self):
        self.s_slider.valueChanged.connect(self._slider_value_changed)
        self.s_spinbox.valueChanged.connect(self._spinbox_value_changed)
        self.show_spline_cb.stateChanged.connect(self.run_spline_transform)
        self.show_tangents_cb.stateChanged.connect(self.run_spline_transform)

    def showEvent(self, event):
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
        self.interest_points.clear()

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
        #log.info(f"Requesting spline transform with s={s_value}")
        spline_y_values = self.pipeline.calculate_spline(s_value)

        show_spline = self.show_spline_cb.isChecked()
        self.spline_curve.setVisible(show_spline)

        for line in self.tangent_lines:
            line.setVisible(False)

        if spline_y_values is not None:
            if show_spline:
                self.spline_curve.setData(self.pipeline.stretch, spline_y_values)
            points = self.pipeline.get_interest_points_on_spline()
            show_tangents = self.show_tangents_cb.isChecked()
            if points:
                x_coords, y_coords, slopes = points
                self.interest_points.setData(x_coords, y_coords)

                if show_tangents:

                    x_range = self.pipeline.stretch.max() - self.pipeline.stretch.min()
                    y_range = spline_y_values.max() - spline_y_values.min()
                    k = y_range / x_range if x_range != 0 else 1

                    # 2. Desired visual length (relative to X-axis)
                    L = 0.05

                    for i, (x, y, m) in enumerate(zip(x_coords, y_coords, slopes)):
                        if i < len(self.tangent_lines):
                            # 3. Calculate dx using the scaled slope (m/k)
                            # This treats the plot as if it were a 1:1 square
                            m_scaled = m / k
                            dx_unit = L / np.sqrt(1 + m_scaled ** 2)
                            dy_unit = m * (dx_unit / k)  # Apply scaling back to dy

                            # Note: we use dx_unit for X, but dy_unit for Y
                            x_data = np.array([x - dx_unit, x + dx_unit])
                            y_data = np.array([y - (m * dx_unit), y + (m * dx_unit)])

                            self.tangent_lines[i].setData(x_data, y_data)
                            self.tangent_lines[i].setVisible(True)


            else:
                self.interest_points.clear()
        else:
            self.spline_curve.clear()
            self.interest_points.clear()

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
        validation_error = self.pipeline.validate_for_stress_stretch()
        if validation_error:
            self.pending_user_error = validation_error
            return

        self.pipeline.get_stress_stretch()
        self._update_plot_and_controls()
        self.last_plotted_version = self.pipeline.data_version

    def _update_plot_and_controls(self):
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
