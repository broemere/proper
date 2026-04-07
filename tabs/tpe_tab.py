from PySide6.QtCore import Qt, QEvent, Slot, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox
from PySide6.QtGui import QColor
import pyqtgraph as pg
from data_pipeline import DataPipeline
import logging
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
log = logging.getLogger(__name__)

class TPETab(QWidget):
    wave_data_updated = Signal(dict)

    def __init__(self, pipeline: DataPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._is_state_synced = False
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        tab_layout = QVBoxLayout(self)

        # --- Plot 1: Pressure Data ---
        self.data_plot = self._create_plot_widget("<b>Trimmed Data & Wave Isolation</b>")
        self.data_plot.setLabel('left', "<b>Pressure [mmHg]</b>")
        self.data_plot.addLegend(offset=(10, 10))
        self.data_plot.showGrid(x=True, y=True, alpha=0.333)

        unsmoothed_color = QColor(self.pipeline._fg_color())
        unsmoothed_color.setAlpha(100)

        pg.setConfigOptions(antialias=True)

        self.trim_curve = self.data_plot.plot([], [], pen=pg.mkPen(unsmoothed_color, width=1.5),
                                              name="Original Data")
        self.lowpass_curve = self.data_plot.plot([], [], pen=pg.mkPen('#FFA500', width=2.5), name="Low-Pass (Basal Tone)")
        self.highpass_curve = self.data_plot.plot([], [], pen=pg.mkPen('#00d2ff', width=1.5), name="High-Pass (Active Contraction)")

        self.highpass_curve.setZValue(2)

        # NEW: Explicit ScatterPlotItem to allow clicking and per-point data
        self.peak_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('w'),
            hoverable=True,
            hoverSymbol='s',  # Turns into a square on hover so the user knows it's clickable
            hoverSize=12,
            name="Detected Peaks"
        )
        self.peak_scatter.setZValue(3)  # Ensure dots draw ON TOP of all the lines
        self.data_plot.addItem(self.peak_scatter)

        # Dummy plots strictly to force the InfiniteLines into the legend
        self.data_plot.plot([], [], pen=pg.mkPen('#ff0000', width=2, style=Qt.DashLine), name="Analysis Start")
        self.data_plot.plot([], [], pen=pg.mkPen('#b500ff', width=2, style=Qt.DashLine), name="Analysis End")

        # Interactive Start Marker (Red)
        self.onset_line_main = pg.InfiniteLine(angle=90, movable=True,
                                               pen=pg.mkPen(color='#ff0000', width=2, style=Qt.DashLine))
        self.data_plot.addItem(self.onset_line_main)

        # Interactive End Marker (Purple)
        self.offset_line_main = pg.InfiniteLine(angle=90, movable=True,
                                                pen=pg.mkPen(color='#b500ff', width=2, style=Qt.DashLine))
        self.data_plot.addItem(self.offset_line_main)

        tab_layout.addWidget(self.data_plot, stretch=2)

        # --- 3. Wave Analysis & Controls ---
        metrics_group = QGroupBox("Wave Statistics (Analyzed between Bounds)")
        metrics_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 5px; }")

        group_layout = QVBoxLayout()
        metrics_group.setLayout(group_layout)

        # A. Control Row (Spin Boxes)
        control_layout = QHBoxLayout()

        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.1, 20.0)
        self.spin_dist.setSingleStep(0.5)
        self.spin_dist.setValue(self.pipeline.tpe_distance)  # Default 4 seconds
        self.spin_dist.setSuffix(" s")

        self.spin_prom = QDoubleSpinBox()
        self.spin_prom.setRange(0.01, 5.0)
        self.spin_prom.setSingleStep(0.1)
        self.spin_prom.setValue(self.pipeline.tpe_prominence)  # Default 0.5x STD
        self.spin_prom.setPrefix("x")

        control_layout.addWidget(QLabel("Min Peak Distance:"))
        control_layout.addWidget(self.spin_dist)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Prominence Threshold (x STD):"))
        control_layout.addWidget(self.spin_prom)
        control_layout.addStretch()
        self.pressure_range = QLabel("Pressure Range: -- mmHg")
        control_layout.addWidget(self.pressure_range)

        group_layout.addLayout(control_layout)

        # B. Metrics Grid (4x2)
        metrics_grid = QGridLayout()
        group_layout.addLayout(metrics_grid)

        self.lbl_count = QLabel("Wave Count: --")
        self.lbl_freq = QLabel("Median Frequency: -- Hz")
        self.lbl_rate = QLabel("Median Rate: -- waves/min")
        self.lbl_period = QLabel("Median Period: -- s")
        self.lbl_amp_mean = QLabel("Mean Peak-to-Peak Amp: -- mmHg")
        self.lbl_amp_med = QLabel("Median Peak-to-Peak Amp: -- mmHg")
        self.lbl_slope_mean = QLabel("Mean Leading Slope: -- mmHg/s")
        self.lbl_slope_med = QLabel("Median Leading Slope: -- mmHg/s")

        labels = [self.lbl_count, self.lbl_freq, self.lbl_rate, self.lbl_period,
                  self.lbl_amp_mean, self.lbl_amp_med, self.lbl_slope_mean, self.lbl_slope_med]
        for lbl in labels:
            lbl.setStyleSheet("font-size: 13px; padding: 3px;")

        # Row 0: Counts, Rates, and Total Energy
        metrics_grid.addWidget(self.lbl_count, 0, 0)
        metrics_grid.addWidget(self.lbl_freq, 1, 1)

        # Row 1-3: Conversions, Amplitudes, Slopes
        metrics_grid.addWidget(self.lbl_rate, 0, 1)
        metrics_grid.addWidget(self.lbl_period, 1, 0)

        metrics_grid.addWidget(self.lbl_amp_mean, 0, 2)
        metrics_grid.addWidget(self.lbl_amp_med, 1, 2)

        metrics_grid.addWidget(self.lbl_slope_mean, 0, 3)
        metrics_grid.addWidget(self.lbl_slope_med, 1, 3)

        tab_layout.addWidget(metrics_group, stretch=0)

    def connect_signals(self):
        self.pipeline.transformed_data.connect(self._data_transformed)
        self.onset_line_main.sigPositionChangeFinished.connect(self._on_bounds_dragged)
        self.offset_line_main.sigPositionChangeFinished.connect(self._on_bounds_dragged)
        self.spin_dist.valueChanged.connect(self._on_params_changed)
        self.spin_prom.valueChanged.connect(self._on_params_changed)
        self.peak_scatter.sigClicked.connect(self._on_peak_clicked)

    def _on_peak_clicked(self, plot, points):
        """NEW: Toggles a wave's exclusion status when clicked"""
        if not points:
            return

        clicked_point = points[0]
        peak_id = clicked_point.data()  # Get the underlying index we assigned

        if peak_id is None:
            return

        if peak_id in self.pipeline.excluded_waves:
            self.pipeline.excluded_waves.remove(peak_id)
        else:
            self.pipeline.excluded_waves.add(peak_id)

        # Re-run metrics to apply the filter and update dot colors immediately
        self._calculate_wave_metrics(self.onset_line_main.value(), self.offset_line_main.value())

    def _on_params_changed(self):
        """Emits new parameters and recalculates metrics based on current bounds."""
        self.pipeline.excluded_waves.clear()  # Clear state on param change
        dist_val = self.spin_dist.value()
        prom_val = self.spin_prom.value()
        # 1. Emit to the pipeline
        self.pipeline.tpe_distance = dist_val
        self.pipeline.tpe_prominence = prom_val
        self._calculate_wave_metrics(self.onset_line_main.value(), self.offset_line_main.value())

    def _on_bounds_dragged(self):
        """Called when the user finishes dragging either the start or end line."""
        start_t = self.onset_line_main.value()
        end_t = self.offset_line_main.value()

        # 1. Enforce logic: Start cannot be dragged past End
        if start_t >= end_t:
            start_t = end_t - 0.1
            self.onset_line_main.setValue(start_t)

        # 3. Emit the updated bounds back to the pipeline/backend
        self.pipeline.excluded_waves.clear()  # Clear state on bound drag
        print(f"User manually updated bounds: Start={start_t:.2f}m, End={end_t:.2f}m")
        self._calculate_wave_metrics(start_t, end_t)

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

    def _update_trimmed_plot(self, data: dict):
        raw_t = np.array(data.get('t', []))
        p = np.array(data.get('p', []))

        if len(raw_t) < 10:
            return

        t_mins = raw_t / 60.0
        dt = np.mean(np.diff(raw_t))
        fs = 1.0 / dt if dt > 0 else 10.0

        cutoff = 0.05
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0: normal_cutoff = 0.99

        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        p_lowpass = filtfilt(b, a, p)
        p_highpass = p - p_lowpass

        onset_time = self.pipeline.tpe_onset_time

        # Set the end time to the absolute final frame of the data
        if self.pipeline.tpe_end_time == -1:
            end_time = t_mins[-1]
        else:
            if self.pipeline.tpe_end_time > max(t_mins):
                self.pipeline.tpe_end_time = t_mins[-1]
            end_time = self.pipeline.tpe_end_time

        # 1. Plot all the continuous curves
        self.trim_curve.setData(t_mins, p)
        self.lowpass_curve.setData(t_mins, p_lowpass)
        self.highpass_curve.setData(t_mins, p_highpass)

        # 2. Position the interactive lines
        self.onset_line_main.setValue(onset_time)
        self.offset_line_main.setValue(end_time)

        # 3. Emit the initial guess to the pipeline immediately
        self.pipeline.excluded_waves.clear()  # Clear state on new data
        print(f"Initial bounds automatically emitted: Start={onset_time:.2f}m, End={end_time:.2f}m")
        self._calculate_wave_metrics(onset_time, end_time)

    def _data_transformed(self, data: list):
        trimmed_data, smoothed_data, zeroed_data = data
        self._update_trimmed_plot(trimmed_data)

    def _calculate_wave_metrics(self, start_t_min, end_t_min):
        """Slices the high-pass data between the user bounds and extracts wave metrics."""
        from scipy.signal import find_peaks
        import numpy as np

        self.pipeline.tpe_onset_time = start_t_min
        self.pipeline.tpe_end_time = end_t_min

        # 1. Grab the currently plotted data from the high-pass curve
        t_mins, p_wave = self.highpass_curve.getData()
        _, p_lowpass = self.lowpass_curve.getData()  # Time axes are identical

        if t_mins is None or len(t_mins) == 0:
            return

        # 2. Slice the data to strictly between the red and purple lines
        mask = (t_mins >= start_t_min) & (t_mins <= end_t_min)
        t_sliced_min = t_mins[mask]
        p_sliced = p_wave[mask]
        p_lowpass_sliced = p_lowpass[mask]

        if len(p_sliced) < 10:
            return

        start_p_val = p_lowpass_sliced[0]
        end_p_val = p_lowpass_sliced[-1]

        self.pressure_range.setText(f"Pressure Range: {start_p_val:.2f} to {end_p_val:.2f} mmHg")

        # Convert time to seconds for standard Hz and dP/dt units
        t_sliced_sec = t_sliced_min * 60.0

        # 3. Find Peaks and Troughs
        dt = np.mean(np.diff(t_sliced_sec))
        fs = 1.0 / dt if dt > 0 else 10.0

        dist_sec = self.spin_dist.value()
        prom_mult = self.spin_prom.value()

        min_distance_idx = max(1, int(dist_sec * fs))
        std_p = np.std(p_sliced)

        # 1. Find ALL peaks
        peaks, _ = find_peaks(
            p_sliced,
            prominence=(prom_mult * std_p),
            distance=min_distance_idx
        )

        total_detected_count = len(peaks)

        # 2. Assign colors based on exclusion state and draw all dots
        if total_detected_count > 0:
            t_peaks = t_sliced_min[peaks]
            p_peaks = p_sliced[peaks] + 0.75

            brushes = []
            for i in range(total_detected_count):
                if i in self.pipeline.excluded_waves:
                    brushes.append(pg.mkBrush(color=(128, 128, 128, 200)))  # Grayed out
                else:
                    brushes.append(pg.mkBrush('#ff007f'))  # Active pink

            # Inject the array index 'i' into the 'data' kwarg so we can retrieve it on click
            self.peak_scatter.setData(t_peaks, p_peaks, brush=brushes, data=np.arange(total_detected_count))
        else:
            self.peak_scatter.setData([], [])

        # 3. Create the valid subset for math
        valid_peaks = [p for i, p in enumerate(peaks) if i not in self.pipeline.excluded_waves]
        wave_count = len(valid_peaks)

        if wave_count < 2:
            self.lbl_count.setText("Wave Count: Not enough valid waves")
            self.lbl_freq.setText("Frequency: -- Hz")
            self.lbl_rate.setText("Rate: -- waves/min")
            self.lbl_period.setText("Period: -- s")
            return

        # 4. Math applies strictly to valid_peaks
        peak_intervals_sec = np.diff(t_sliced_sec[valid_peaks])
        period_sec = float(np.median(peak_intervals_sec))
        freq_hz = 1.0 / period_sec if period_sec > 0 else 0.0
        waves_per_min = freq_hz * 60.0

        amplitudes = []
        max_slopes = []

        for i in range(len(valid_peaks)):
            peak_idx = valid_peaks[i]

            # Searching back to the previous VALID peak ensures we find the global trough
            # even if there's a noisy bump we excluded between them.
            start_search = valid_peaks[i - 1] if i > 0 else 0

            if start_search == peak_idx:
                continue

            trough_rel_idx = np.argmin(p_sliced[start_search:peak_idx])
            trough_idx = start_search + trough_rel_idx

            amp = p_sliced[peak_idx] - p_sliced[trough_idx]
            amplitudes.append(amp)

            trough_val = p_sliced[trough_idx]
            thresh_20 = trough_val + 0.20 * amp
            thresh_80 = trough_val + 0.80 * amp

            t_edge = t_sliced_sec[trough_idx:peak_idx + 1]
            p_edge = p_sliced[trough_idx:peak_idx + 1]

            window_mask = (p_edge >= thresh_20) & (p_edge <= thresh_80)
            t_window = t_edge[window_mask]
            p_window = p_edge[window_mask]

            if len(t_window) > 1:
                slope, intercept = np.polyfit(t_window, p_window, 1)
                max_slopes.append(slope)
            elif len(t_edge) > 1:
                fallback_slope = (p_edge[-1] - p_edge[0]) / (t_edge[-1] - t_edge[0])
                max_slopes.append(fallback_slope)

        self.lbl_count.setText(f"Wave Count: {wave_count}")
        self.lbl_freq.setText(f"Median Frequency: {freq_hz:.3f} Hz")
        self.lbl_rate.setText(f"Median Rate: {waves_per_min:.2f} waves/min")
        self.lbl_period.setText(f"Median Period: {period_sec:.2f} s")

        if amplitudes:
            self.lbl_amp_mean.setText(f"Mean Peak-to-Peak Amp: {np.mean(amplitudes):.2f} mmHg")
            self.lbl_amp_med.setText(f"Median Peak-to-Peak Amp: {np.median(amplitudes):.2f} mmHg")

        if max_slopes:
            self.lbl_slope_mean.setText(f"Mean Leading Slope: {np.mean(max_slopes):.2f} mmHg/s")
            self.lbl_slope_med.setText(f"Median Leading Slope: {np.median(max_slopes):.2f} mmHg/s")

        wave_data = {
            "mean_p2p_amp(mmHg)": np.mean(amplitudes),
            "median_p2p_amp(mmHg)": np.median(amplitudes),
            "mean_leading_slope(mmHg/s)": np.mean(max_slopes),
            "median_leading_slope(mmHg/s)": np.median(max_slopes),
            "freq(Hz)": freq_hz,
            "waves_per_min": waves_per_min,
            "period(s)": period_sec,
            "wave_count": wave_count,
        }
        self.pipeline.wave_data = wave_data