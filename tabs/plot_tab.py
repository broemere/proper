from PySide6.QtCore import Qt, QEvent, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox
from PySide6.QtGui import QColor
import pyqtgraph as pg
from data_pipeline import DataPipeline
import logging
log = logging.getLogger(__name__)

class PlotTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._is_state_synced = False
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        tab_layout = QVBoxLayout(self)

        self.data_plot = self._create_plot_widget("<b>Trimmed and Transformed Data</b>")
        self.data_plot.setLabel('left', "<b>Pressure [mmHg]</b>")
        self.data_plot.setLabel('bottom', "<b>Time [min]</b>")
        self.data_plot.addLegend()
        zeroed_color = QColor("#16A085")  # A nice green shade
        unsmoothed_color = QColor(self.pipeline._fg_color())
        unsmoothed_color.setAlpha(100)  # semi-transparent foreground color for the background curves
        smoothed_color = QColor(self.pipeline._fg_color())
        smoothed_color.setAlpha(200)  # Makes it less prominent
        pg.setConfigOptions(antialias=True)

        self.trim_curve = self.data_plot.plot([], [], pen=pg.mkPen(unsmoothed_color, width=1.5), name="Trimmed")
        self.smooth_curve = self.data_plot.plot([], [], pen=pg.mkPen(smoothed_color, width=1.5), name="Smoothed")
        self.zero_curve = self.data_plot.plot([], [],pen=pg.mkPen(zeroed_color, width=2.5), name="Zeroed")
        self.trim_curve.setZValue(0)  # Draw this curve first (in the back)
        self.smooth_curve.setZValue(1)  # Draw this curve next
        self.zero_curve.setZValue(2)  # Draw this curve last (on top)
        tab_layout.addWidget(self.data_plot)

        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignCenter)

        # Trim controls
        trim_layout = QHBoxLayout()
        trim_layout.setAlignment(Qt.AlignCenter)
        trim_layout.addWidget(QLabel('Trim Data:'))
        self.spin_start = QSpinBox(minimum=1)
        self.spin_stop = QSpinBox(minimum=1)
        self.spin_start.valueChanged.connect(self._apply_trim)  # Initialize limits after loading data
        self.spin_stop.valueChanged.connect(self._apply_trim)
        trim_layout.addWidget(QLabel('Start'))
        trim_layout.addWidget(self.spin_start)
        trim_layout.addWidget(QLabel('Stop'))
        trim_layout.addWidget(self.spin_stop)
        btn_reset = QPushButton('Reset')
        btn_reset.clicked.connect(self._reset_trim)
        trim_layout.addWidget(btn_reset)
        controls_layout.addLayout(trim_layout)

        # Zero controls
        zero_methods = ["None", "First", "Min", "Median"]
        zero_layout = QHBoxLayout()
        zero_layout.setAlignment(Qt.AlignCenter)
        zero_layout.addWidget(QLabel('Zero Method:'))
        self.cb_zero = QComboBox()
        self.cb_zero.addItems(zero_methods)
        self.cb_zero.setCurrentText(self.pipeline.zeroing_method)
        self.cb_zero.currentTextChanged.connect(self._apply_zeroing)
        zero_layout.addWidget(self.cb_zero)
        self.spin_zero_window = QSpinBox(minimum=0, maximum=999, value=7)
        self.spin_zero_window.installEventFilter(self)
        self.spin_zero_window.valueChanged.connect(self._apply_zeroing)
        zero_layout.addWidget(QLabel('Window'))
        zero_layout.addWidget(self.spin_zero_window)
        controls_layout.addLayout(zero_layout)

        # Smooth controls
        smooth_methods = ["None", "Min", "Double Min", "Moving Avg", "Median", "Gaussian"]
        smooth_layout = QHBoxLayout()
        smooth_layout.setAlignment(Qt.AlignCenter)
        smooth_layout.addWidget(QLabel('Smoothing Method:'))
        self.cb_smooth = QComboBox()
        self.cb_smooth.addItems(smooth_methods)
        self.cb_smooth.setCurrentText(self.pipeline.smoothing_method)
        self.cb_smooth.currentTextChanged.connect(self._apply_smoothing)
        smooth_layout.addWidget(self.cb_smooth)
        self.spin_smooth_window = QSpinBox(minimum=0, maximum=999, value=100)
        self.spin_smooth_window.installEventFilter(self)
        self.spin_smooth_window.valueChanged.connect(self._apply_smoothing)
        smooth_layout.addWidget(QLabel('Window'))
        smooth_layout.addWidget(self.spin_smooth_window)
        controls_layout.addLayout(smooth_layout)

        tab_layout.addWidget(controls_widget)

        self._hover_source = None
        self._hover_region = None

    def connect_signals(self):
        self.pipeline.new_data.connect(self._new_data_loaded)
        self.pipeline.transformed_data.connect(self._data_transformed)
        self.pipeline.trimming_data.connect(self._sync_ui_to_pipeline)
        self.pipeline.state_loaded.connect(self._on_state_loaded)

    def showEvent(self, event: QEvent):
        """This Qt event fires every time the widget is shown."""
        # First, let the parent class do its thing
        super().showEvent(event)
        # If the widget is being shown and its UI is out of sync, update it now.
        if self.isVisible() and not self._is_state_synced:
            self._sync_ui_to_pipeline()

    @Slot()
    def _on_state_loaded(self):
        """
        Slot for the 'state_loaded' signal from the pipeline.
        Marks the UI as dirty and triggers a sync if the tab is already visible.
        """
        log.info("PlotTab: Received 'state_loaded' notification.")
        self._is_state_synced = False
        # If the tab is already visible when the state is loaded, update immediately.
        # Otherwise, showEvent will handle it when the user clicks on the tab.
        if self.isVisible():
            self._sync_ui_to_pipeline()

    def _sync_ui_to_pipeline(self, *args):
        """Pulls the current state from the pipeline and updates all UI controls."""
        log.info("PlotTab: Synchronizing UI controls to pipeline state.")

        # Block signals to prevent feedback loops while we set values
        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.cb_zero.blockSignals(True)
        self.spin_zero_window.blockSignals(True)
        self.cb_smooth.blockSignals(True)
        self.spin_smooth_window.blockSignals(True)

        try:
            # Update trim controls
            self.spin_start.setMaximum(max(0, self.pipeline.working_length-1))
            self.spin_stop.setMaximum(max(0, self.pipeline.working_length-1))
            self.spin_start.setValue(self.pipeline.trim_start)
            self.spin_stop.setValue(self.pipeline.trim_stop)

            # Update zeroing controls
            self.cb_zero.setCurrentText(self.pipeline.zeroing_method)
            self.spin_zero_window.setValue(self.pipeline.zeroing_window)

            # Update smoothing controls
            self.cb_smooth.setCurrentText(self.pipeline.smoothing_method)
            self.spin_smooth_window.setValue(self.pipeline.smoothing_window)

            self._is_state_synced = True

        finally:
            self.spin_start.blockSignals(False)
            self.spin_stop.blockSignals(False)
            self.cb_zero.blockSignals(False)
            self.spin_zero_window.blockSignals(False)
            self.cb_smooth.blockSignals(False)
            self.spin_smooth_window.blockSignals(False)

    def _create_plot_widget(self, title: str) -> pg.PlotWidget:
        """Helper to create a theme-aware plot with zero lines"""
        plot = pg.PlotWidget(title=title, color=self.pipeline._fg_color(), size='12pt')
        plot.setBackground(self.pipeline._bg_color())
        for axis in ('bottom', 'left'):  # Configure axes
            ax = plot.getAxis(axis)
            ax.setPen(pg.mkPen(self.pipeline._fg_color()))
            ax.setTextPen(self.pipeline._fg_color())
        # Zero lines
        plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=self.pipeline._fg_color(), style=Qt.DotLine)))
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color=self.pipeline._fg_color(), style=Qt.DotLine)))
        return plot

    def _reset_trim(self):
        self.spin_start.setValue(0)
        self.spin_stop.setValue(self.spin_stop.maximum())
        self._apply_trim()

    def _apply_trim(self):
        start = self.spin_start.value()
        stop = self.spin_stop.value()
        print("Setting trim", start, stop)
        self.pipeline.set_trimming(start, stop)

    def _apply_zeroing(self):
        method = self.cb_zero.currentText()
        window = self.spin_zero_window.value()
        self.pipeline.set_zeroing(method, window)

    def _apply_smoothing(self):
        method = self.cb_smooth.currentText()
        window = self.spin_smooth_window.value()
        self.pipeline.set_smoothing(method, window)

    def _update_zero_plot(self, data: dict):
        t = [x / 60 for x in data.get('t', [])]
        p = data.get('p', [])
        self.zero_curve.setData(t, p)

    def _update_smoothed_plot(self, data: dict):
        t = [x / 60 for x in data.get('t', [])]
        p = data.get('p', [])
        self.smooth_curve.setData(t, p)

    def _update_trimmed_plot(self, data: dict):
        t = [x / 60 for x in data.get('t', [])]
        p = data.get('p', [])
        self.trim_curve.setData(t, p)

    def _new_data_loaded(self):
        self.spin_start.setMaximum(self.pipeline.working_length-1)
        self.spin_start.setValue(0)
        self.spin_stop.setMaximum(self.pipeline.working_length-1)
        self.spin_stop.setValue(self.pipeline.working_length-1)
        log.info("New data")

    def _data_transformed(self, data: list):
        trimmed_data, smoothed_data, zeroed_data = data
        self._update_zero_plot(zeroed_data)
        self._update_smoothed_plot(smoothed_data)
        self._update_trimmed_plot(trimmed_data)

    def eventFilter(self, obj, event):
        smooth = getattr(self, 'spin_smooth_window', None)
        zero = getattr(self, 'spin_zero_window', None)
        if obj is smooth or obj is zero:
            if event.type() == QEvent.Enter:
                self._hover_source = 'smooth' if obj is smooth else 'zero'
                self._show_hover_region()
            elif event.type() == QEvent.Leave:
                self._hover_source = None
                self._hide_hover_region()
        return super().eventFilter(obj, event)

    def _show_hover_region(self):
        # Remove existing region
        if self._hover_region:
            self.data_plot.removeItem(self._hover_region)
        # Determine window value based on hover source
        if self._hover_source == 'smooth':
            window = self.spin_smooth_window.value()
        elif self._hover_source == 'zero':
            window = self.spin_zero_window.value()
        else:
            return
        color = QColor(self.pipeline._fg_color())
        color.setAlpha(50)

        t = self.pipeline.zeroed_data.get('t', [])
        try:
            upper_limit = max(t[0:window])/60
        except:
            upper_limit = 1

        region = pg.LinearRegionItem(values=[0, upper_limit], brush=color, movable=False)
        region.setZValue(-10)
        self.data_plot.addItem(region)
        self._hover_region = region

    def _hide_hover_region(self):
        if self._hover_region:
            self.data_plot.removeItem(self._hover_region)
            self._hover_region = None
