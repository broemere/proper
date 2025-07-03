from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QApplication
from PySide6.QtGui import QPalette, QColor
import pyqtgraph as pg
from data_pipeline import DataPipeline

class PlotTab(QWidget):

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()
        self.pipeline.register_observer("raw", self._new_data_loaded)
        self.pipeline.register_observer("transformed", self._data_transformed)

    def init_ui(self):
        tab_layout = QVBoxLayout(self)
        # Transformed Data Plot
        self.data_plot = self._create_plot_widget("Zeroed and Smoothed Data")
        self.data_plot.addLegend()
        self.zero_curve = self.data_plot.plot([], [], pen=pg.mkPen(self._fg_color(), width=1), symbol='o', name="Zeroed")
        self.smooth_curve = self.data_plot.plot([], [], pen=pg.mkPen(self._fg_color(), width=2), name="Smoothed")
        self.data_plot.setLabel('left', 'Pressure [mmHg]')
        self.data_plot.setLabel('bottom', 'Time [min]')
        tab_layout.addWidget(self.data_plot)

        # --- Controls centered below charts ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignCenter)

        # Trim controls
        trim_layout = QHBoxLayout()
        trim_layout.setAlignment(Qt.AlignCenter)
        trim_layout.addWidget(QLabel('Trim Data:'))
        self.spin_start = QSpinBox(minimum=0)
        self.spin_stop = QSpinBox(minimum=0)
        # Initialize limits after loading data
        self.spin_start.valueChanged.connect(self._apply_trim)
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


    def _create_plot_widget(self, title: str) -> pg.PlotWidget:
        """Helper to create a theme-aware plot with zero lines"""
        plot = pg.PlotWidget(title=title)
        plot.setBackground(self._bg_color())
        # Configure axes
        for axis in ('bottom', 'left'):
            ax = plot.getAxis(axis)
            ax.setPen(pg.mkPen(self._fg_color()))
            ax.setTextPen(self._fg_color())
        # Zero lines
        plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(self._fg_color())))
        plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(self._fg_color())))
        return plot

    def _bg_color(self) -> str:
        return QApplication.instance().palette().color(QPalette.Window).name()

    def _fg_color(self) -> str:
        return QApplication.instance().palette().color(QPalette.WindowText).name()

    def _reset_trim(self):
        self.spin_start.setValue(0)
        self.spin_stop.setValue(self.spin_stop.maximum())
        self._apply_trim()

    def _apply_trim(self):
        start = self.spin_start.value()
        stop = self.spin_stop.value()
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

    def _new_data_loaded(self, data):
        self.spin_start.setMaximum(self.pipeline.length-1)
        self.spin_stop.setMaximum(self.pipeline.length-1)
        self.spin_stop.setValue(self.pipeline.length-1)
        self._apply_trim()
        # self._apply_zeroing()
        # zeroed_data = self.pipeline.get_data("zeroed")
        # self._update_zero_plot(zeroed_data)
        # self._apply_smoothing()
        # smoothed_data = self.pipeline.get_data("smoothed")
        # self._update_smoothed_plot(smoothed_data)

    def _data_transformed(self, data: list):
        zeroed_data, smoothed_data = data
        self._update_zero_plot(zeroed_data)
        self._update_smoothed_plot(smoothed_data)


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
        color = QColor(self._fg_color())
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
