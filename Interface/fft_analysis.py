"""
fft_analysis.py

Live FFT / PSD Analysis window for Behavior Recording
=======================================================
A standalone module, kept separate from BehaviorRecording.py on purpose
so the main application file doesn't grow further and so this window
can be tested/reused independently. Opened from BehaviorRecording's
Analysis > FFT... menu action, which imports FFTAnalysisWindow from
here.

Shows a LIVE power spectral density (Welch's method -- same approach
used in the offline MATLAB analysis scripts elsewhere in this project)
of the currently streaming signal, recomputed on a timer over a rolling
window of the most recent N seconds. Useful for watching the spectrum
change in real time -- e.g. while adjusting mechanical isolation, or
hunting for a specific vibration frequency, without having to stop,
save, and analyze offline first.

This window reads directly from the parent BehaviorRecording window's
own live buffers (signal_time / signal_value / signal_processed_value)
-- it does NOT open its own serial connection or duplicate any
acquisition logic, so there is exactly one source of truth for the data.

REQUIREMENTS: numpy, scipy, pyqtgraph, PyQt5 (all already required by
BehaviorRecording.py itself).

AUTHOR: Flavio Mourao (mourao.fg@gmail.com)
Started: 07/2026
"""

import numpy as np
from scipy.signal import welch, detrend
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore


class FFTAnalysisWindow(QtWidgets.QDialog):
    """
    Live FFT / power spectral density viewer.

    Parameters exposed to the examiner, each with a sensible default
    matching the offline MATLAB PSD workflow used elsewhere in this
    project:

      Source          -- "Raw" or "Processed" (whichever signal the
                          main window is currently streaming/logging).
                          Default: Raw. Curve color follows the same
                          convention as the main window's plot (gray for
                          Raw, red for Processed).
      Window (s)       -- how much of the most recent signal history to
                          analyze on each update (i.e. how many segments
                          get averaged together -- more history means a
                          smoother, less noisy PSD estimate at the SAME
                          frequency resolution). Default: 5 s.
      Update (ms)      -- how often to recompute/redraw. Default: 500 ms.
      Segment (s)      -- length of each FFT segment inside Welch's
                          method (nperseg, in seconds) -- this is what
                          actually sets the FREQUENCY RESOLUTION
                          (\u0394f \u2248 1/Segment(s)), and is DIFFERENT
                          from "Window (s)" above. Longer segment = finer
                          resolution but a noisier estimate (fewer
                          segments to average) and needs "Window (s)" to
                          be at least this long; shorter segment =
                          coarser resolution but smoother/faster.
                          Default: 3 s.
      Spectral window  -- the WINDOW FUNCTION applied to each segment
                          before its FFT in Welch's method (distinct from
                          "Window (s)" above). Default: Hamming, matching
                          MATLAB's pwelch() default -- scipy's own
                          welch() defaults to Hann instead, which is
                          similar but not identical; this selector keeps
                          results directly comparable to the offline
                          MATLAB analysis scripts used elsewhere in this
                          project.
      Detrend          -- remove the analyzed window's own linear trend
                          before computing the PSD, so slow drift
                          doesn't inflate the lowest-frequency bins (the
                          same reasoning applied in the MATLAB scripts).
                          Default: ON.
      Log Y            -- logarithmic Y axis, matching the MATLAB PSD
                          plots. Default: ON.

    Nothing here is saved to disk -- this is a live viewing tool only,
    independent of whatever is or isn't being recorded at the time.
    """

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Live FFT / PSD Analysis")
        self.resize(720, 520)

        layout = QtWidgets.QVBoxLayout(self)

        # -- Controls --
        controls = QtWidgets.QHBoxLayout()

        controls.addWidget(QtWidgets.QLabel("Source:"))
        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.addItems(["Raw", "Processed"])
        self.combo_source.setToolTip(
            "Raw: the always-unfiltered signal (Value column).\n"
            "Processed: whatever real-time High-pass/Low-pass/Moving-\n"
            "average chain is currently active in the main window.")
        self.combo_source.currentTextChanged.connect(self._apply_source_color)
        controls.addWidget(self.combo_source)

        controls.addWidget(QtWidgets.QLabel("Window (s):"))
        self.spin_window = QtWidgets.QDoubleSpinBox()
        self.spin_window.setRange(1.0, 120.0)
        self.spin_window.setSingleStep(1.0)
        self.spin_window.setValue(5.0)
        controls.addWidget(self.spin_window)

        controls.addWidget(QtWidgets.QLabel("Update (ms):"))
        self.spin_update_ms = QtWidgets.QSpinBox()
        self.spin_update_ms.setRange(100, 5000)
        self.spin_update_ms.setSingleStep(100)
        self.spin_update_ms.setValue(500)
        self.spin_update_ms.valueChanged.connect(self._on_update_interval_changed)
        controls.addWidget(self.spin_update_ms)

        controls.addWidget(QtWidgets.QLabel("Segment (s):"))
        self.spin_segment_s = QtWidgets.QDoubleSpinBox()
        self.spin_segment_s.setRange(0.5, 30.0)
        self.spin_segment_s.setSingleStep(0.5)
        self.spin_segment_s.setValue(3.0)
        self.spin_segment_s.setToolTip(
            "Length of each FFT segment inside Welch's method (nperseg, "
            "expressed in seconds) -- this is what actually sets the "
            "FREQUENCY RESOLUTION: \u0394f \u2248 1 / Segment(s). It is "
            "DIFFERENT from 'Window (s)' above, which only controls how "
            "much history feeds in (i.e. how many segments get averaged "
            "together for a smoother estimate). Longer segment = finer "
            "frequency resolution but a noisier estimate (fewer segments "
            "to average) and needs 'Window (s)' to be at least this long; "
            "shorter segment = coarser resolution but a smoother, faster-"
            "responding estimate.")
        controls.addWidget(self.spin_segment_s)

        controls.addWidget(QtWidgets.QLabel("Spectral window:"))
        self.combo_window_fn = QtWidgets.QComboBox()
        self.combo_window_fn.addItems(["hamming", "hann", "blackman", "boxcar"])
        self.combo_window_fn.setToolTip(
            "The WINDOW FUNCTION applied to each segment before its FFT "
            "in Welch's method -- not the same as 'Window (s)' above, "
            "which is how much time history is analyzed. Default here "
            "is Hamming, matching MATLAB's pwelch() default (scipy's own "
            "welch() default is Hann instead -- these are similar but not "
            "identical, hence this selector, so results are directly "
            "comparable to the offline MATLAB analysis if desired).")
        controls.addWidget(self.combo_window_fn)

        self.chk_detrend = QtWidgets.QCheckBox("Detrend")
        self.chk_detrend.setChecked(True)
        self.chk_detrend.setToolTip(
            "Removes the analyzed window's own linear trend before "
            "computing the PSD -- avoids slow drift inflating the "
            "lowest frequency bins.")
        controls.addWidget(self.chk_detrend)

        self.chk_log_scale = QtWidgets.QCheckBox("Log Y")
        self.chk_log_scale.setChecked(True)
        self.chk_log_scale.stateChanged.connect(self._apply_log_scale)
        controls.addWidget(self.chk_log_scale)

        controls.addStretch()
        layout.addLayout(controls)

        # -- Plot --
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#141414')
        self.plot.setLabel('bottom', 'Frequency', 'Hz')
        self.plot.setLabel('left', 'PSD')
        # Same reasoning as BehaviorRecording's own signal plot: disable
        # pyqtgraph's automatic SI-prefix axis scaling, which otherwise
        # stacks a confusing "(x10^-n)" multiplier onto the axis label.
        self.plot.getAxis('left').enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        # Curve color tracks the selected Source, matching the same
        # convention as the main window's own plot (gray=raw, red=processed).
        self.SOURCE_COLORS = {"Raw": "#c22017", "Processed": "#44aaff"}
        self.curve = self.plot.plot(pen=pg.mkPen(self.SOURCE_COLORS["Raw"], width=2))
        layout.addWidget(self.plot, 1)

        self.lbl_status = QtWidgets.QLabel("Waiting for data...")
        layout.addWidget(self.lbl_status)

        self._apply_log_scale()

        # -- Update timer: recomputes the PSD on a fixed cadence, reading
        # whatever is currently in the parent window's live buffers. --
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.spin_update_ms.value())

    def _on_update_interval_changed(self, value):
        self.timer.setInterval(value)

    def _apply_source_color(self, source_text):
        """Recolors the curve to match the selected Source, following
        the main window's own convention (gray=raw, red=processed)."""
        color = self.SOURCE_COLORS.get(source_text, "#44aaff")
        self.curve.setPen(pg.mkPen(color, width=2))

    def _apply_log_scale(self):
        self.plot.setLogMode(x=False, y=self.chk_log_scale.isChecked())

    def update_plot(self):
        """
        Pulls the most recent `window_s` seconds of data from the
        parent's live buffers, computes its PSD via Welch's method, and
        redraws the curve. Does nothing (just updates the status label)
        if there isn't enough data yet -- e.g. the signal isn't
        connected, or fewer samples have arrived than the window needs.
        """
        source = self.combo_source.currentText()
        t_all = self.main_window.signal_time
        v_all = (self.main_window.signal_value if source == "Raw"
                 else self.main_window.signal_processed_value)

        if len(t_all) < 10 or len(v_all) < 10:
            self.lbl_status.setText("Waiting for data...")
            return

        window_s = self.spin_window.value()
        t_arr = np.asarray(t_all)
        v_arr = np.asarray(v_all[:len(t_arr)])   # guard against a rare 1-sample race

        t_end = t_arr[-1]
        mask  = t_arr >= (t_end - window_s)
        t_win = t_arr[mask]
        v_win = v_arr[mask]

        if len(t_win) < 10:
            self.lbl_status.setText("Waiting for enough samples in the window...")
            return

        fs = 1.0 / np.mean(np.diff(t_win))

        if self.chk_detrend.isChecked():
            v_win = detrend(v_win, type='linear')

        # Segment length (nperseg) comes directly from the "Segment (s)"
        # control, converted to samples at this window's actual fs --
        # capped to the available samples so Welch stays well-defined
        # even early on / with short "Window (s)" settings.
        nperseg = min(int(round(self.spin_segment_s.value() * fs)), len(v_win))
        nperseg = max(nperseg, 4)   # scipy needs at least a few samples per segment
        window_fn = self.combo_window_fn.currentText()
        freq, pxx = welch(v_win, fs=fs, window=window_fn, nperseg=nperseg)

        self.curve.setData(freq, pxx)

        peak_idx = int(np.argmax(pxx))
        delta_f = fs / nperseg
        self.lbl_status.setText(
            f"{source} — fs≈{fs:.1f} Hz, N={len(v_win)} samples, "
            f"segment={nperseg} samples (\u0394f≈{delta_f:.3f} Hz), "
            f"peak at {freq[peak_idx]:.2f} Hz")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)