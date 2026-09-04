"""
BehaviorRecording.py

Behavior Recording
===================
Multi-camera video acquisition and load cell recording for behavioral conditioning setups.

DESCRIPTION:
    Provides live preview and independent recording of up to 4 USB
    cameras simultaneously, arranged in an adaptive grid layout,
    alongside real-time acquisition and plotting of a serial signal
    (currently a single HX711 load-cell channel read from an Arduino
    Nano; designed to extend to up to 4 independent channels).

FEATURES:
    - Live preview of 1 to 4 USB cameras in an adaptive grid:
        1 camera  -> full panel
        2 cameras -> side by side
        3 cameras -> two on top, one on bottom (full width)
        4 cameras -> 2x2 grid
      The grid updates live as the camera count is changed, before
      streaming even starts.
    - Each camera runs on its own independent capture thread, so one
      camera failing, disconnecting, or not being found does not affect
      the others — only that camera's grid slot shows "unavailable".
    - Independent video recording per camera (separate file per camera,
      same session timestamp), with a choice of output format:
        - AVI / MJPG  (most broadly compatible across operating systems)
        - MP4 / mp4v  (smaller files; codec availability is OS-dependent)
    - Recording uses real-time frame pacing: the number of frames written
      is kept in sync with actual elapsed time rather than simply one
      frame per camera callback, so the recorded file's duration matches
      the real session duration regardless of the camera's true delivery
      rate.
    - Decoupled recording: "Start Recording" works with whatever is
      currently active — camera(s) only, the load cell signal only, or
      both together. Nothing is required to be running before you can
      record; each active source is written to its own file, and the
      load cell log is skipped entirely for video-only sessions (no
      empty file left behind).
    - Synchronized recording: when both are active, clicking
      "Start Recording" logs the load cell signal for the same session,
      sharing the same zero-point in time as the video frame pacing. The
      log is saved as a .csv file with a header documenting the channel
      name, serial port, baud rate, sample count, duration, and the
      measured average sample rate — written once the session ends, once
      the true rate is known.
    - Real-time serial signal acquisition: reads a load-cell value from
      an Arduino Nano (HX711 amplifier) over a serial connection, on its
      own background thread. Non-numeric lines (startup banner, tare
      confirmations) are ignored automatically. A "Tare" button sends the
      zeroing command to the device. The live plot can show either a
      scrolling time window (oscilloscope-style) or the full session
      history. An optional moving-average smoothing can be applied to
      the live plot for readability — this is display-only and never
      affects the raw values written to the recording log.
    - Event marking: an "Events" button (enabled only while recording)
      draws a vertical line on the live plot and flags the nearest
      sample in the CSV. mark_event(code) is also public API, so another
      application can trigger it directly (e.g. Conditioning Setup marks
      each stimulus onset via window.mark_event(code)). Codes: 1=Sound,
      2=Light, 3=Shock, 4=Trigger 1, 5=Trigger 2, 6=this window's own
      manual button. Each code gets its OWN 0/1 column in the CSV
      (Sound, Light, Shock, Trigger1, Trigger2, Manual) rather than one
      shared column, so simultaneous events (e.g. Sound and Light onset
      at the same instant) don't overwrite each other.
    - Settings > Video...: a live camera adjustment panel (Brightness/
      Contrast/Sharpness sliders applying in real time, plus Resolution
      via an explicit Apply button) affecting every currently active
      camera at once. Support for these properties is driver/OS-
      dependent — some cameras or backends (notably AVFoundation on
      macOS) silently ignore brightness/contrast/sharpness; resolution
      tends to be the most reliably supported of the four.
    - Signal displayed and logged in millivolts (mV), not raw ADC
      counts: converted per the HX711 datasheet's own full-scale
      formula, so the value represents the load cell's actual
      differential output voltage (at its E+/E- terminals) — a
      property of the sensor/electronics, independent of any
      grams calibration. See counts_to_mv() / AVDD_VOLTS below.
    - Gain control: a Gain combo (128 default / 64) sends a serial
      command to the Arduino to change the HX711's Channel A PGA gain
      at runtime, re-taring automatically afterward (the zero-offset is
      gain-dependent). Only 128/64 are offered — 32 belongs to Channel
      B, a separate physical input not wired in this design.
    - Real-time signal processing: an optional High-pass -> Low-pass ->
      Moving-average chain (each stage independently toggleable) is
      applied to the raw signal as it streams in, producing a
      "Processed" value plotted as a second curve and saved as its own
      CSV column, alongside the always-unfiltered raw "Value (mV)"
      column. This is CAUSAL (real-time) filtering, not the zero-phase
      filtfilt used for offline MATLAB analysis, so it has some phase
      delay — use the raw column, not the processed one, for precise
      event-timing analysis.

PLANNED (not yet implemented):
    - A second, third, and fourth independent signal channel (multiple
      load cells), mirroring the camera grid's approach.
    - Per-sample / per-frame timestamp logging, to allow precise post-hoc
      alignment between cameras and the signal channel(s).

WORKFLOW:
    1. Set "Number of cameras" (1 to 4). Camera 1 uses device index 0,
       Camera 2 uses index 1, and so on.
    2. Click "Start Cameras" and/or connect the signal (see step 5) —
       either one, or both, can be active before recording.
    3. Click "Start Recording": whatever is currently active (video,
       signal, or both) gets recorded. Choose an output folder and
       format under "Recording" first if you haven't already.
    4. Click "Stop Recording" to finalize the files, then "Stop Cameras"
       / disconnect the signal as needed (stopping cameras while a
       recording is active also stops the recording).
    5. For the signal: select the serial port and baud rate (115200 to
       match the provided Arduino sketch), click "Connect". Use "Tare"
       to re-zero the load cell at any time. Choose "Scrolling window" or
       "Full history" to change how the live plot is displayed.

REQUIREMENTS:
    Python >= 3.8
    PyQt5, pyqtgraph, opencv-python, numpy, pyserial, scipy

AUTHOR:
    Flavio Mourao (mourao.fg@gmail.com)

Started:     04/2026
Last update: 07/2026
"""

import sys
import os
import time
import threading
from collections import deque
import numpy as np
import cv2
import serial
import serial.tools.list_ports
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from scipy.signal import butter, sosfilt, sosfilt_zi

# Theme
BG        = "#141414"
BG_PANEL  = "#141414"
TEXT      = "#c8c8c8"
BORDER    = "#2a2a2a"
DIM       = "#666666"

GLOBAL_STYLESHEET = f"""
    QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
    QGroupBox {{ background: {BG_PANEL}; border: 1px solid {BORDER}; margin-top: 10px; }}
    QGroupBox::title {{
        color: {TEXT};
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
    }}
    QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
"""

# Cameras
MAX_CAMERAS = 4

VIDEO_FORMATS = {
    "AVI (MJPG)": (".avi", "MJPG"),
    "MP4 (mp4v)": (".mp4", "mp4v"),
}

# Numeric codes identifying each event type. Each code gets its own 0/1
# column in the CSV (see EVENT_COLUMNS below), and its own vertical-line
# color on the live plot (see EVENT_COLORS). 1-5 are meant to be
# triggered externally (e.g. by Conditioning Setup, one per stimulus
# type); 6 is the manual "Events" button in this window. Colors 1-5
# match Conditioning Setup's own stimulus color scheme (ACC_BLUE/
# ACC_YELL/ACC_RED/ACC_TRG1/ACC_TRG2) for visual consistency between the
# two apps.
EVENT_LABELS = {
    1: "Sound",
    2: "Light",
    3: "Shock",
    4: "Trigger 1",
    5: "Trigger 2",
    6: "Manual",
}
EVENT_COLORS = {
    1: (68, 170, 255, 160),    # blue
    2: (255, 170, 68, 160),    # yellow/orange
    3: (238, 68, 68, 160),     # red
    4: (170, 170, 170, 160),   # light grey
    5: (85, 85, 85, 160),      # dark grey
    6: (150, 150, 150, 150),   # neutral grey (manual button press)
}
# CSV column names (space-free) for the per-event-type Event columns, in
# the order they're written. Using one 0/1 column per event type (instead
# of a single categorical column) means simultaneous events at the same
# sample no longer collide/overwrite each other.
EVENT_COLUMNS = {
    1: "Sound",
    2: "Light",
    3: "Shock",
    4: "Trigger1",
    5: "Trigger2",
    6: "Manual",
}

# =============================================================================
# HX711 -> VOLTAGE CONVERSION
# Converts raw ADC counts to the load cell's actual differential output
# voltage (millivolts, at its E+/E- terminals), per the HX711 datasheet's
# own full-scale-range formula:
#     full_scale_V = 0.5 * (AVDD_VOLTS / gain)
#     counts_to_mV = (full_scale_V / 2**23) * 1000
#
# This characterizes the ELECTRICAL signal itself -- a property of the
# sensor/ADC/gain setting, independent of which load cell is attached or
# how it's calibrated in grams. Useful for studying electrical noise in
# an absolute, comparable-across-setups unit.
#
# The conversion depends on gain, which can now change at runtime (see
# Settings > Video... sibling control, the Gain combo in the signal
# panel) -- counts_to_mV(gain) recomputes it accordingly. Only 128 and
# 64 are ever passed in: those are HX711 Channel A's two gain options
# (matching this design's A+/A- wiring). Gain 32 belongs to Channel B --
# a SEPARATE physical input, not wired here -- so it is deliberately
# never offered; selecting it would silently read an unconnected input,
# not "the same signal at lower gain".
#
# ASSUMPTIONS -- adjust to match your actual hardware:
#   AVDD_VOLTS -- the HX711's actual regulated excitation voltage. 5.0V
#                 is the datasheet's reference value, but many modules
#                 regulate slightly lower in practice (commonly closer
#                 to ~4.3V). For best accuracy, measure the AVDD pin
#                 directly with a multimeter and update this value --
#                 it directly scales every mV value this app shows/saves.
# =============================================================================
HX711_GAIN_DEFAULT = 128
AVDD_VOLTS          = 5.0

# HX711 RATE pin is a hardware jumper on this board (80 SPS "shorted",
# 10 SPS otherwise) -- not software-controllable, so this is fixed here
# to match. Used only to DESIGN the real-time filters below (their
# cutoff-to-Nyquist ratio depends on it); it is NOT used to change the
# actual acquisition rate.
NOMINAL_SAMPLE_RATE_HZ = 80.0


def counts_to_mv(gain):
    """mV per raw ADC count, for the given HX711 Channel A gain (128 or 64)."""
    full_scale_v = 0.5 * (AVDD_VOLTS / gain)
    return (full_scale_v / (2 ** 23)) * 1000.0


# Fixed Y-axis display units for the live plot. Everything is always
# stored/logged internally (and in the CSV) in mV regardless of this
# setting -- these only rescale what's DRAWN, replacing pyqtgraph's
# automatic SI-prefix axis scaling (which was producing a confusing
# stacked-unit label, e.g. "mmV", with a "(x10^-3)"-style multiplier).
DISPLAY_UNIT_SCALES = {
    "mV": 1.0,
    "\u00b5V": 1000.0,   # microvolts
    "V":  0.001,
}


class RealtimeFilter:
    """
    A causal (real-time-safe) IIR Butterworth filter, processed one
    sample at a time using scipy's second-order-sections form with
    persistent state (zi) carried across calls -- this statefulness is
    what makes it usable in a live stream at all, unlike MATLAB's
    filtfilt (zero-phase, but requires the entire signal already in
    memory to run both forward and backward).

    TRADE-OFF: a causal filter has real phase delay (the filtered
    output lags the true input by some amount that depends on filter
    order and how close the signal is to the cutoff frequency -- see
    the discussion in chat for estimated numbers, roughly ~15-30ms per
    low-pass stage at order 1-2, less for a high-pass well above its
    own cutoff). filtfilt, used for all the offline MATLAB analysis,
    has none. This is an inherent property of real-time filtering, not
    a bug -- there is no such thing as a zero-phase live filter.
    """
    def __init__(self, kind, cutoff_hz, fs_hz, order=2):
        """kind: 'highpass' or 'lowpass'."""
        self.kind      = kind
        self.cutoff_hz = cutoff_hz
        self.fs_hz     = fs_hz
        self.order     = order
        self._design()

    def _design(self):
        nyquist    = self.fs_hz / 2.0
        normalized = min(max(self.cutoff_hz / nyquist, 1e-6), 0.999)
        self.sos = butter(self.order, normalized, btype=self.kind, output='sos')
        self.zi  = sosfilt_zi(self.sos)
        self._initialized = False

    def reset(self):
        """Clears the filter's internal state -- call this on a fresh
        connection, or whenever cutoff/order changes, so stale history
        from a previous signal doesn't bleed into new samples."""
        self.zi = sosfilt_zi(self.sos)
        self._initialized = False

    def process(self, x):
        """Filters a single new sample and returns the filtered value."""
        if not self._initialized:
            # Prime the filter's initial state to the first sample's own
            # level, rather than zero -- avoids a startup ramp transient
            # that a causal filter would otherwise show for the first
            # several samples after connecting.
            self.zi = self.zi * x
            self._initialized = True
        y, self.zi = sosfilt(self.sos, [x], zi=self.zi)
        return float(y[0])


class CameraWorker(QtCore.QThread):
    """
    Continuously captures frames from a single camera on a background
    thread and emits them (tagged with this camera's device index) via a
    Qt signal for the main thread to display and/or record.

    The main GUI thread never calls cv2.VideoCapture directly: all camera
    I/O happens here. This keeps the interface responsive and — on
    macOS in particular — avoids capture calls stalling when made
    synchronously from within the Qt event loop. The default OpenCV
    backend is used (no backend forced), for the broadest camera
    compatibility across operating systems.

    Each camera gets its own independent CameraWorker instance/thread,
    so cameras never block or interfere with one another.
    """
    frame_ready = QtCore.pyqtSignal(int, np.ndarray)   # camera index, BGR frame
    error       = QtCore.pyqtSignal(int, str)          # camera index, message

    def __init__(self, index=0, parent=None):
        super().__init__(parent)
        self.index    = index
        self._running = False
        self.cap        = None              # set once opened in run(); guarded by _prop_lock
        self._prop_lock = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.error.emit(self.index, f"Could not open camera {self.index}.")
            return

        self.cap      = cap
        self._running = True
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.error.emit(
                    self.index,
                    f"Failed to grab frame from camera {self.index} "
                    "(disconnected?).")
                break
            self.frame_ready.emit(self.index, frame)

        cap.release()
        self.cap = None

    def set_property(self, prop_id, value):
        """
        Thread-safe: request a cv2.VideoCapture property change (e.g.
        brightness, resolution) from the main thread while this worker's
        run() loop is reading frames on its own thread.

        Support for these properties is driver/OS-dependent — some
        cameras or backends silently ignore unsupported properties
        (cv2 won't raise an error either way).
        """
        with self._prop_lock:
            if self.cap is not None:
                try:
                    self.cap.set(prop_id, value)
                except Exception:
                    pass

    def get_property(self, prop_id):
        """Thread-safe read of a current camera property value, or None
        if unavailable."""
        with self._prop_lock:
            if self.cap is not None:
                try:
                    return self.cap.get(prop_id)
                except Exception:
                    return None
        return None

    def stop(self):
        """Ask the run() loop to exit; it releases the camera on its way out."""
        self._running = False


class SerialWorker(QtCore.QThread):
    """
    Reads lines from a serial port on a background thread and emits each
    successfully parsed numeric sample as (timestamp, value). Follows the
    same pattern as CameraWorker: all I/O happens here, the main thread
    only ever receives data via signals.

    The Arduino sketch this is designed for mixes plain numeric lines
    (the actual signal) with occasional human-readable text lines (a
    startup banner, tare confirmation messages). Any line that fails to
    parse as a number is silently ignored — this is expected, not an
    error condition.

    Timestamps are generated on arrival (time.perf_counter() since the
    connection was opened), not read from the device, since the sketch
    does not send one. This also means the true sampling rate is
    whatever is empirically observed, which may differ from a rate
    "requested" in the firmware (e.g. an HX711 load-cell amplifier in
    its default configuration converts at ~10 samples/sec natively,
    regardless of how often the sketch checks for new data).

    NOTE: the value emitted here is the RAW ADC count from the Arduino
    (as printed by the sketch) -- the mV conversion happens in the main
    window's on_sample_ready(), not here, keeping this class focused
    purely on serial I/O.
    """
    sample_ready = QtCore.pyqtSignal(float, float)   # elapsed seconds, raw ADC count
    error        = QtCore.pyqtSignal(str)
    connected    = QtCore.pyqtSignal()

    def __init__(self, port, baud=115200, parent=None):
        super().__init__(parent)
        self.port  = port
        self.baud  = baud
        self._running    = False
        self._ser        = None
        self._start_time = None
        self._write_lock = threading.Lock()

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
        except Exception as e:
            self.error.emit(f"Could not open {self.port}: {e}")
            return

        self._start_time = time.perf_counter()
        self.connected.emit()
        self._running = True

        while self._running:
            try:
                raw = self._ser.readline()
            except Exception as e:
                self.error.emit(f"Serial read error: {e}")
                break

            if not raw:
                continue   # read timeout with no data; loop back and re-check _running

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                value = float(line)
            except ValueError:
                continue   # non-numeric line (banner / tare confirmation text) — ignore

            t = time.perf_counter() - self._start_time
            self.sample_ready.emit(t, value)

        if self._ser is not None:
            self._ser.close()

    def send_tare(self):
        """Send the 't' tare command to the Arduino. Safe to call from
        the main thread while run() is reading on its own thread."""
        if self._ser is not None and self._ser.is_open:
            with self._write_lock:
                try:
                    self._ser.write(b't')
                except Exception:
                    pass

    def send_gain(self, gain):
        """
        Send the gain-change command to the Arduino: 'h' for 128
        (Channel A, default/"high"), 'm' for 64 (Channel A, "medium").
        Gain 32 (Channel B) is never sent -- that's a separate physical
        input not wired in this design. The firmware re-tares
        automatically after changing gain, since the zero-offset is
        gain-dependent.
        """
        if self._ser is None or not self._ser.is_open:
            return
        command = {128: b'h', 64: b'm'}.get(gain)
        if command is None:
            return
        with self._write_lock:
            try:
                self._ser.write(command)
            except Exception:
                pass

    def stop(self):
        """Ask the run() loop to exit; it closes the port on its way out."""
        self._running = False


class VideoSettingsDialog(QtWidgets.QDialog):
    """
    Live camera adjustment panel reachable from Settings > Video...
    Brightness/Contrast/Sharpness apply in real time as each slider
    moves; Resolution applies via an explicit button (changing
    resolution mid-stream can cause a brief visual hiccup, so it isn't
    tied to a slider).

    The SAME value is applied to every currently active camera at once
    (not per-camera individually).

    IMPORTANT: actual hardware/driver support for these properties
    varies a lot by OS and camera. Some backends (notably AVFoundation
    on macOS) accept the cv2.VideoCapture.set() call without error but
    do not actually change anything for brightness/contrast/sharpness.
    Resolution tends to be the most reliably supported of the four.
    """
    RESOLUTIONS = ["640x480", "1280x720", "1920x1080"]

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Video Settings")
        self.setMinimumWidth(380)

        layout = QtWidgets.QVBoxLayout(self)

        note = QtWidgets.QLabel(
            "Applies to all active cameras at once. Support for these "
            "controls depends on your camera/driver — some settings may "
            "not have any visible effect on certain cameras.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(note)

        self._add_slider(layout, "Brightness", cv2.CAP_PROP_BRIGHTNESS)
        self._add_slider(layout, "Contrast",   cv2.CAP_PROP_CONTRAST)
        self._add_slider(layout, "Sharpness",  cv2.CAP_PROP_SHARPNESS)

        # -- Resolution --
        res_layout = QtWidgets.QHBoxLayout()
        res_layout.addWidget(QtWidgets.QLabel("Resolution:"))
        self.combo_resolution = QtWidgets.QComboBox()
        self.combo_resolution.addItems(self.RESOLUTIONS)
        res_layout.addWidget(self.combo_resolution)

        btn_apply_res = QtWidgets.QPushButton("Apply Resolution")
        btn_apply_res.clicked.connect(self.apply_resolution)
        res_layout.addWidget(btn_apply_res)
        layout.addLayout(res_layout)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

    def _add_slider(self, layout, label_text, prop_id):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label_text + ":"))

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 100)

        # Best-effort initial value: query the first active camera; if
        # the driver doesn't report a usable value, default to the
        # middle of the range (50).
        first_worker = next(iter(self.main_window.cam_workers.values()))
        current = first_worker.get_property(prop_id)
        initial = int(current) if current is not None and 0 <= current <= 100 else 50
        slider.setValue(initial)

        spin = QtWidgets.QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(initial)

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda v, p=prop_id: self._apply_to_all(p, v))

        row.addWidget(slider, 1)
        row.addWidget(spin)
        layout.addLayout(row)

    def _apply_to_all(self, prop_id, value):
        for worker in self.main_window.cam_workers.values():
            worker.set_property(prop_id, value)

    def apply_resolution(self):
        w_str, h_str = self.combo_resolution.currentText().split("x")
        w, h = int(w_str), int(h_str)
        for worker in self.main_window.cam_workers.values():
            worker.set_property(cv2.CAP_PROP_FRAME_WIDTH, w)
            worker.set_property(cv2.CAP_PROP_FRAME_HEIGHT, h)


class BehaviorRecording(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Behavior Recording")
        self.setStyleSheet(GLOBAL_STYLESHEET)
        
        self.resize(1300, 700)

        self.cam_workers   = {}    # camera index -> CameraWorker
        self.active_indices = []   # camera indices currently streaming, in grid order
        self.is_streaming  = False

        self.is_recording   = False
        self.video_writers  = {}    # camera index -> cv2.VideoWriter (opened lazily)
        self.frames_written = {}    # camera index -> count of frames written so far
        self.output_dir     = None
        self.record_start_time = None
        self.session_timestamp = None   # shared across video files + signal log, per session
        self.signal_log_rows   = []     # (elapsed_s_since_record_start, raw_mV, processed_mV), logged while recording
        self.event_times_rec   = []     # (elapsed_s_since_record_start, code) per mark_event() call -- own vs external
        self.event_lines       = []     # pg.InfiniteLine items drawn on the live plot for each Events click

        self.rec_clock_timer = QtCore.QTimer()
        self.rec_clock_timer.timeout.connect(self.update_recording_clock)

        self.serial_worker      = None
        self.is_signal_connected = False
        self.signal_time  = []   # elapsed seconds since connection, per sample (for the live plot)
        self.signal_value = []   # display value per sample, in mV (raw, or smoothed if enabled)
        self.display_scale = 1000.0   # multiplies mV for the PLOT only; default unit = uV
        self.display_unit  = "\u00b5V"
        self.smooth_deque = deque(maxlen=5)   # rolling buffer for the optional DISPLAY-ONLY moving average

        # -- HX711 gain (Channel A: 128 or 64 only -- see counts_to_mv() note) --
        self.current_gain        = HX711_GAIN_DEFAULT
        self.counts_to_mv_factor = counts_to_mv(self.current_gain)

        # -- Real-time signal processing chain: High-pass -> Low-pass ->
        # Moving average, each stage independently toggleable. Unlike
        # smooth_deque above (display-only), this chain's output IS
        # saved -- as a "Processed (mV)" column alongside the raw
        # "Value (mV)" column -- and is also drawn as a second curve on
        # the live plot. Causal (real phase delay), not zero-phase --
        # see RealtimeFilter's docstring. Filter objects are built lazily
        # in rebuild_processing_chain(), called once controls exist.
        self.hp_filter = None
        self.lp_filter = None
        self.proc_ma_deque = deque(maxlen=5)
        self.signal_processed_value = []   # mirrors signal_value, but processed

        self.setup_gui()

    # ---------------------------------------------------------------
    # GUI SETUP

    def setup_gui(self):
        # -- Menu bar --
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        action_video = QtWidgets.QAction("Video...", self)
        action_video.triggered.connect(self.open_video_settings)
        settings_menu.addAction(action_video)

        # Analysis menu: live tools kept in separate module files (not
        # written into this one) so this file doesn't keep growing --
        # each is imported lazily, on first use, from the same folder.
        analysis_menu = menubar.addMenu("Analysis")
        action_fft = QtWidgets.QAction("FFT...", self)
        action_fft.triggered.connect(self.open_fft_analysis)
        analysis_menu.addAction(action_fft)
        self.fft_window = None   # created on first use; reused afterward

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # -------------------------------------------------------
        # LEFT PANEL: Cameras
        cam_group  = QtWidgets.QWidget()
        cam_layout = QtWidgets.QVBoxLayout(cam_group)
        cam_layout.setContentsMargins(0, 0, 0, 0)

        # -- Video grid (1-4 adaptive), boxed on its own --
        video_group  = QtWidgets.QGroupBox()
        video_layout = QtWidgets.QVBoxLayout()

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(2)
        self.grid_widget.setMinimumSize(560, 400)

        self.video_labels = []
        for i in range(MAX_CAMERAS):
            lbl = QtWidgets.QLabel(f"Camera {i + 1}")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("background-color: black; color: #888;")
            self.video_labels.append(lbl)
        self.rebuild_grid_layout(1)

        video_layout.addWidget(self.grid_widget)
        video_group.setLayout(video_layout)

        # -- Number of cameras: automatically assigns device indices
        #    0..N-1, in order, to grid slots 1..N --
        num_layout = QtWidgets.QHBoxLayout()
        num_layout.addWidget(QtWidgets.QLabel("Number of cameras:"))
        self.spin_num_cameras = QtWidgets.QSpinBox()
        self.spin_num_cameras.setRange(1, MAX_CAMERAS)
        self.spin_num_cameras.setValue(1)
        self.spin_num_cameras.valueChanged.connect(self.on_camera_selection_changed)
        num_layout.addWidget(self.spin_num_cameras)
        num_layout.addStretch()

        # -- Start / Stop --
        self.btn_start = QtWidgets.QPushButton("Start Cameras")
        self.btn_start.clicked.connect(self.start_cameras)

        self.btn_stop = QtWidgets.QPushButton("Stop Cameras")
        self.btn_stop.clicked.connect(self.stop_cameras)
        self.btn_stop.setEnabled(False)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        self.lbl_status = QtWidgets.QLabel("Status: idle")

        cam_layout.addWidget(video_group, 1)
        cam_layout.addLayout(num_layout)
        cam_layout.addLayout(btn_layout)
        cam_layout.addWidget(self.lbl_status)

        # -------------------------------------------------------
        # RIGHT PANEL: Signal (serial acquisition)
        plot_group  = QtWidgets.QGroupBox()
        plot_layout = QtWidgets.QVBoxLayout()

        self.plot_signal = pg.PlotWidget(title="")
        self.plot_signal.setBackground('#141414')
        self.plot_signal.setLabel('bottom', 'Time', 's')
        # NOTE: pyqtgraph auto-applies its own SI-prefix scaling when a
        # "units" string is passed as the 3rd argument here, and expects
        # a BASE unit (e.g. "V") to do that -- passing an already-prefixed
        # unit like "mV" makes it stack another prefix on top (producing
        # a nonsensical "mmV" label and multiplying the displayed axis
        # values by 1000). Embedding "(mV)" directly in the text instead
        # disables that auto-scaling, so the axis shows the real mV values.
        self.plot_signal.setLabel('left', 'Signal (\u00b5V)')
        # pyqtgraph's Y axis auto-applies its own SI-prefix scaling
        # (e.g. showing "(x0.001)" next to the label) based on the data
        # range, REGARDLESS of what text/units we pass to setLabel above
        # -- it's a separate feature of the axis itself. Disabling it
        # here is what actually stops that confusing multiplier
        # annotation from appearing; changing the label text alone
        # (as tried before) does not.
        self.plot_signal.getAxis('left').enableAutoSIPrefix(False)
        #self.curve_signal = self.plot_signal.plot(pen=pg.mkPen('#0072BD', width=2)) # BLUE
        #self.curve_signal = self.plot_signal.plot(pen=pg.mkPen('#D95319', width=2)) # ORANGE
        self.curve_signal = self.plot_signal.plot(pen=pg.mkPen('#c22017', width=2)) # RED (raw)
        self.curve_processed = self.plot_signal.plot(pen=pg.mkPen('#44aaff', width=2))  # BLUE (processed)
        
        # -- Display mode: scrolling window vs. full growing history --
        display_layout = QtWidgets.QHBoxLayout()
        display_layout.addWidget(QtWidgets.QLabel("View:"))
        self.combo_display_mode = QtWidgets.QComboBox()
        self.combo_display_mode.addItems(["Scrolling window", "Full history"])
        self.combo_display_mode.currentIndexChanged.connect(self.on_display_mode_changed)
        display_layout.addWidget(self.combo_display_mode)

        display_layout.addWidget(QtWidgets.QLabel("Window (s):"))
        self.spin_window_seconds = QtWidgets.QSpinBox()
        self.spin_window_seconds.setRange(1, 300)
        self.spin_window_seconds.setValue(10)
        display_layout.addWidget(self.spin_window_seconds)

        # -- Fixed Y-axis unit: replaces pyqtgraph's automatic SI-prefix
        #    scaling (which produced a confusing stacked-unit label) with
        #    an explicit choice. Internally always mV; this only rescales
        #    what's drawn on the plot. --
        display_layout.addWidget(QtWidgets.QLabel("Y unit:"))
        self.combo_display_unit = QtWidgets.QComboBox()
        self.combo_display_unit.addItems(list(DISPLAY_UNIT_SCALES.keys()))
        self.combo_display_unit.setCurrentText("\u00b5V")
        self.combo_display_unit.currentTextChanged.connect(self.on_display_unit_changed)
        display_layout.addWidget(self.combo_display_unit)

        # -- Curve visibility: colors are always fixed (raw=red,
        #    processed=blue) -- but when no processing stage is active,
        #    the processed curve is IDENTICAL to the raw one and, being
        #    drawn on top, visually hides it (looks like "everything
        #    turned blue", though the red curve is still there underneath
        #    unchanged). This selector lets you show just one at a time
        #    to avoid that overlap confusion, or both together. --
        display_layout.addWidget(QtWidgets.QLabel("Show:"))
        self.combo_curve_visibility = QtWidgets.QComboBox()
        self.combo_curve_visibility.addItems(["Both", "Raw only", "Processed only"])
        self.combo_curve_visibility.currentTextChanged.connect(self.on_curve_visibility_changed)
        display_layout.addWidget(self.combo_curve_visibility)

        display_layout.addStretch()

        # -- Optional smoothing: affects the LIVE PLOT only. The raw,
        #    unaveraged signal is always what gets written to the
        #    recording's .csv log — smoothing here is purely a display
        #    convenience and is never applied to the saved data. --
        smooth_layout = QtWidgets.QHBoxLayout()
        self.chk_smooth = QtWidgets.QCheckBox("Smooth plot (moving average)")
        self.chk_smooth.stateChanged.connect(self.on_smoothing_changed)
        smooth_layout.addWidget(self.chk_smooth)

        smooth_layout.addWidget(QtWidgets.QLabel("Window (samples):"))
        self.spin_smooth_window = QtWidgets.QSpinBox()
        self.spin_smooth_window.setRange(2, 100)
        self.spin_smooth_window.setValue(5)
        self.spin_smooth_window.setEnabled(False)   # matches chk_smooth starting unchecked
        self.spin_smooth_window.valueChanged.connect(self.on_smoothing_changed)
        smooth_layout.addWidget(self.spin_smooth_window)
        smooth_layout.addStretch()

        # -- Port / refresh / connect (single toggle) / baud rate --
        port_layout = QtWidgets.QHBoxLayout()
        port_layout.addWidget(QtWidgets.QLabel("Port:"))
        self.combo_serial_port = QtWidgets.QComboBox()
        self.combo_serial_port.setMinimumWidth(140)
        port_layout.addWidget(self.combo_serial_port)

        self.btn_refresh_ports = QtWidgets.QPushButton("\u27f3")   # small refresh icon (↻)
        self.btn_refresh_ports.setFixedWidth(28)
        self.btn_refresh_ports.setToolTip("Refresh port list")
        self.btn_refresh_ports.clicked.connect(self.populate_serial_ports)
        port_layout.addWidget(self.btn_refresh_ports)

        self.btn_conn_toggle = QtWidgets.QPushButton("Connect")
        self.btn_conn_toggle.clicked.connect(self.toggle_signal_connection)
        port_layout.addWidget(self.btn_conn_toggle)

        port_layout.addWidget(QtWidgets.QLabel("Baud:"))
        self.combo_baud = QtWidgets.QComboBox()
        self.combo_baud.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.combo_baud.setCurrentText("115200")   # matches the Arduino sketch
        self.combo_baud.setMinimumWidth(80)
        port_layout.addWidget(self.combo_baud)
        port_layout.addStretch()

        # -- Tare / Events / Gain --
        tare_layout = QtWidgets.QHBoxLayout()
        self.btn_tare = QtWidgets.QPushButton("Tare")
        self.btn_tare.clicked.connect(self.tare_signal)
        self.btn_tare.setEnabled(False)
        tare_layout.addWidget(self.btn_tare)

        self.btn_events = QtWidgets.QPushButton("Events")
        self.btn_events.setToolTip(
            "Marks the current moment as a manual event: draws a line on "
            "the live plot and sets a 1 in the CSV's 'Manual' column at "
            "the nearest sample. Enabled only while recording.")
        self.btn_events.clicked.connect(lambda checked=False: self.mark_event(6))
        self.btn_events.setEnabled(False)
        tare_layout.addWidget(self.btn_events)

        tare_layout.addWidget(QtWidgets.QLabel("Gain:"))
        self.combo_gain = QtWidgets.QComboBox()
        self.combo_gain.addItems(["128 (default)", "64"])
        self.combo_gain.setToolTip(
            "HX711 Channel A gain. Only 128/64 are offered -- gain 32 is "
            "Channel B, a separate physical input not wired in this "
            "design. Changing gain re-tares automatically on the Arduino.")
        self.combo_gain.setEnabled(False)
        self.combo_gain.currentIndexChanged.connect(self.on_gain_changed)
        tare_layout.addWidget(self.combo_gain)
        tare_layout.addStretch()

        # -- Real-time signal processing chain: High-pass -> Low-pass ->
        # Moving average. Each stage is independently toggleable and
        # feeds a SEPARATE "Processed (mV)" column in the CSV (the raw
        # "Value (mV)" column is always saved unfiltered, regardless of
        # these settings) plus a second curve on the live plot. This is
        # causal/real-time filtering (not zero-phase like the offline
        # MATLAB filtfilt analysis), so it introduces some phase delay --
        # see RealtimeFilter's docstring for estimated magnitudes.
        proc_group  = QtWidgets.QGroupBox("Signal Processing (saved as 'Processed')")
        proc_layout = QtWidgets.QHBoxLayout()

        self.chk_hp = QtWidgets.QCheckBox("High-pass")
        self.chk_hp.stateChanged.connect(self.rebuild_processing_chain)
        proc_layout.addWidget(self.chk_hp)
        self.spin_hp_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_hp_cutoff.setRange(0.01, 30.0)
        self.spin_hp_cutoff.setValue(0.5)
        self.spin_hp_cutoff.setSuffix(" Hz")
        self.spin_hp_cutoff.valueChanged.connect(self.rebuild_processing_chain)
        proc_layout.addWidget(self.spin_hp_cutoff)

        self.chk_lp = QtWidgets.QCheckBox("Low-pass")
        self.chk_lp.stateChanged.connect(self.rebuild_processing_chain)
        proc_layout.addWidget(self.chk_lp)
        self.spin_lp_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_lp_cutoff.setRange(0.5, 40.0)
        self.spin_lp_cutoff.setValue(10.0)
        self.spin_lp_cutoff.setSuffix(" Hz")
        self.spin_lp_cutoff.valueChanged.connect(self.rebuild_processing_chain)
        proc_layout.addWidget(self.spin_lp_cutoff)

        self.chk_proc_ma = QtWidgets.QCheckBox("Moving avg.")
        self.chk_proc_ma.stateChanged.connect(self.rebuild_processing_chain)
        proc_layout.addWidget(self.chk_proc_ma)
        self.spin_proc_ma_window = QtWidgets.QSpinBox()
        self.spin_proc_ma_window.setRange(2, 100)
        self.spin_proc_ma_window.setValue(5)
        self.spin_proc_ma_window.setSuffix(" samples")
        self.spin_proc_ma_window.valueChanged.connect(self.rebuild_processing_chain)
        proc_layout.addWidget(self.spin_proc_ma_window)

        proc_layout.addStretch()
        proc_group.setLayout(proc_layout)

        self.lbl_signal_status = QtWidgets.QLabel("Status: idle")

        plot_layout.addWidget(self.plot_signal, 1)
        plot_layout.addLayout(display_layout)
        plot_layout.addLayout(smooth_layout)
        plot_layout.addSpacing(12)
        plot_layout.addWidget(proc_group)
        plot_layout.addLayout(tare_layout)
        plot_layout.addLayout(port_layout)
        plot_layout.addWidget(self.lbl_signal_status)
        plot_group.setLayout(plot_layout)

        self.populate_serial_ports()

        # -------------------------------------------------------
        main_layout.addWidget(cam_group, 1)
        main_layout.addWidget(plot_group, 1)

        # -------------------------------------------------------
        # RECORDING panel: governs camera video + load cell log together.
        # Placed at the bottom of the LEFT (Cameras) column, not spanning
        # the full window width, but visually its own box (separate from
        # the camera controls above it) since "Start Recording" now
        # synchronizes camera video and the load cell log together.
        rec_group  = QtWidgets.QGroupBox()
        rec_layout = QtWidgets.QVBoxLayout()

        title_output = QtWidgets.QLabel("Recording")
        title_output.setStyleSheet("font-weight: bold;")
        folder_layout = QtWidgets.QHBoxLayout()
        self.lbl_output_dir = QtWidgets.QLabel("Output folder: (none selected)")
        self.lbl_output_dir.setWordWrap(True)
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self.choose_output_dir)
        folder_layout.addWidget(self.lbl_output_dir, 1)
        folder_layout.addWidget(btn_browse)

        format_layout = QtWidgets.QHBoxLayout()
        format_layout.addWidget(QtWidgets.QLabel("Video Format:"))
        self.combo_format = QtWidgets.QComboBox()
        self.combo_format.addItems(list(VIDEO_FORMATS.keys()))
        format_layout.addWidget(self.combo_format)
        format_layout.addWidget(QtWidgets.QLabel("FPS:"))
        self.spin_record_fps = QtWidgets.QSpinBox()
        self.spin_record_fps.setRange(1, 120)
        self.spin_record_fps.setValue(30)
        format_layout.addWidget(self.spin_record_fps)
        format_layout.addStretch()

        rec_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_start_rec = QtWidgets.QPushButton("● Start Recording")
        self.btn_start_rec.clicked.connect(self.start_recording)
        self.btn_start_rec.setEnabled(False)

        self.btn_stop_rec = QtWidgets.QPushButton("Stop Recording")
        self.btn_stop_rec.clicked.connect(self.stop_recording)
        self.btn_stop_rec.setEnabled(False)

        rec_btn_layout.addWidget(self.btn_start_rec)
        rec_btn_layout.addWidget(self.btn_stop_rec)
        rec_btn_layout.addStretch()

        self.lbl_rec_status = QtWidgets.QLabel("Not recording")

        rec_layout.addWidget(title_output)
        rec_layout.addLayout(folder_layout)
        rec_layout.addLayout(format_layout)
        rec_layout.addLayout(rec_btn_layout)
        rec_layout.addWidget(self.lbl_rec_status)
        rec_group.setLayout(rec_layout)

        cam_layout.addWidget(rec_group)

    def rebuild_grid_layout(self, n):
        """
        Re-arrange the video grid for exactly `n` active cameras (1-4).
        Reuses the same persistent QLabel widgets; only their position/
        span inside the QGridLayout changes, and unused labels are hidden.
        """
        n = max(1, min(n, MAX_CAMERAS))

        for lbl in self.video_labels:
            self.grid_layout.removeWidget(lbl)
            lbl.setVisible(False)

        if n == 1:
            self.grid_layout.addWidget(self.video_labels[0], 0, 0, 2, 2)
        elif n == 2:
            self.grid_layout.addWidget(self.video_labels[0], 0, 0, 2, 1)
            self.grid_layout.addWidget(self.video_labels[1], 0, 1, 2, 1)
        elif n == 3:
            self.grid_layout.addWidget(self.video_labels[0], 0, 0)
            self.grid_layout.addWidget(self.video_labels[1], 0, 1)
            self.grid_layout.addWidget(self.video_labels[2], 1, 0, 1, 2)
        else:  # n == 4
            self.grid_layout.addWidget(self.video_labels[0], 0, 0)
            self.grid_layout.addWidget(self.video_labels[1], 0, 1)
            self.grid_layout.addWidget(self.video_labels[2], 1, 0)
            self.grid_layout.addWidget(self.video_labels[3], 1, 1)

        for i in range(n):
            self.video_labels[i].setVisible(True)

    def on_camera_selection_changed(self):
        """
        Live preview: as soon as the camera count changes, rearrange the
        grid to match — no need to click Start first to see the layout.
        Has no effect while actually streaming (the control is disabled
        then anyway).
        """
        if self.is_streaming:
            return
        self.rebuild_grid_layout(self.spin_num_cameras.value())

    def open_video_settings(self):
        """Settings > Video... — opens the live brightness/contrast/
        sharpness/resolution adjustment panel for the active camera(s)."""
        if not self.cam_workers:
            QtWidgets.QMessageBox.information(
                self, "No Camera Active",
                "Start at least one camera before adjusting video settings.")
            return
        dialog = VideoSettingsDialog(self, parent=self)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.show()

    def open_fft_analysis(self):
        """
        Analysis > FFT... — opens the live PSD viewer, implemented in
        the separate fft_analysis.py module (kept in the same folder).
        Reuses the same window instance on subsequent clicks rather than
        creating a new one each time, just raising/focusing it if it's
        already open.
        """
        try:
            from fft_analysis import FFTAnalysisWindow
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "FFT Analysis Not Found",
                "fft_analysis.py was not found alongside this script.\n\n"
                "Place both files in the same folder to enable this feature.")
            return

        if self.fft_window is None:
            self.fft_window = FFTAnalysisWindow(self, parent=self)
        self.fft_window.show()
        self.fft_window.raise_()
        self.fft_window.activateWindow()

    # ---------------------------------------------------------------
    # CAMERA CONTROL

    def start_cameras(self):
        if self.is_streaming:
            return

        n = self.spin_num_cameras.value()
        self.active_indices = list(range(n))   # Camera 1=index 0, Camera 2=index 1, ...

        self.rebuild_grid_layout(n)
        for i, lbl in enumerate(self.video_labels[:n]):
            lbl.clear()
            lbl.setText(f"Opening Camera {i + 1}...")

        self.lbl_status.setText(f"Status: opening cameras 0-{n - 1}...")

        self.cam_workers = {}
        for idx in self.active_indices:
            worker = CameraWorker(idx)
            worker.frame_ready.connect(self.on_frame_ready)
            worker.error.connect(self.on_camera_error)
            self.cam_workers[idx] = worker
            worker.start()

        self.is_streaming = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.spin_num_cameras.setEnabled(False)
        self.update_recording_availability()
        self.lbl_status.setText(f"Status: streaming cameras 0-{n - 1}")

    def stop_cameras(self):
        if self.is_recording:
            self.stop_recording()

        for idx, worker in self.cam_workers.items():
            # Disconnect first: a frame emitted right as the thread was
            # stopping can still be queued for delivery even after wait()
            # returns. Disconnecting prevents a late frame from being
            # drawn over the placeholder text set below.
            try:
                worker.frame_ready.disconnect(self.on_frame_ready)
            except TypeError:
                pass
            worker.stop()
        for worker in self.cam_workers.values():
            worker.wait(2000)
        self.cam_workers = {}
        self.active_indices = []

        self.is_streaming = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.spin_num_cameras.setEnabled(True)
        self.update_recording_availability()
        self.lbl_status.setText("Status: idle")

        for i, lbl in enumerate(self.video_labels):
            lbl.clear()
            lbl.setText(f"Camera {i + 1}")

    def on_frame_ready(self, idx, frame):
        """Slot: runs on the main thread (Qt marshals the cross-thread
        signal automatically), safe to touch the QLabel here."""
        if idx not in self.active_indices:
            return
        pos = self.active_indices.index(idx)
        if pos >= len(self.video_labels):
            return
        label = self.video_labels[pos]

        if self.is_recording:
            self.write_frame_to_disk(idx, pos, frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = frame_rgb.shape
        q_img     = QtGui.QImage(
            frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        label.setPixmap(
            QtGui.QPixmap.fromImage(q_img).scaled(
                label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))

    def on_camera_error(self, idx, message):
        """
        A single camera failed (not found / disconnected). Stop and clean
        up only that camera — the others keep streaming undisturbed.
        """
        pos = self.active_indices.index(idx) if idx in self.active_indices else idx
        QtWidgets.QMessageBox.warning(
            self, "Camera Error",
            f"Camera {pos + 1} was not found or could not be opened.\n\n"
            "Check that it's connected and not in use by another app.")

        worker = self.cam_workers.pop(idx, None)
        if worker is not None:
            try:
                worker.frame_ready.disconnect(self.on_frame_ready)
            except TypeError:
                pass
            worker.stop()
            worker.wait(2000)

        # Mark that camera's grid slot as unavailable, but keep
        # active_indices unchanged so the OTHER cameras' grid positions
        # (computed via active_indices.index(their_idx)) don't shift.
        if idx in self.active_indices:
            slot_pos = self.active_indices.index(idx)
            if slot_pos < len(self.video_labels):
                lbl = self.video_labels[slot_pos]
                lbl.clear()
                lbl.setText(f"Camera {slot_pos + 1}\nunavailable")

        # Release this camera's recording file (if any) — the other
        # cameras keep recording undisturbed.
        writer = self.video_writers.pop(idx, None)
        if writer is not None:
            writer.release()

        if not self.cam_workers:
            # No camera left running at all -> fully reset the UI.
            # If a recording was in progress, stop_recording() also saves
            # the load cell log collected so far (nothing is lost).
            if self.is_recording:
                self.stop_recording()
            self.is_streaming   = False
            self.active_indices = []
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.spin_num_cameras.setEnabled(True)
            self.update_recording_availability()
            self.lbl_status.setText("Status: idle")
        else:
            still_running = [i for i in self.active_indices if i in self.cam_workers]
            self.lbl_status.setText(
                f"Status: streaming {still_running} (camera index {idx} failed)")

    # ---------------------------------------------------------------
    # SERIAL SIGNAL

    def populate_serial_ports(self):
        """Refresh the list of available serial ports."""
        self.combo_serial_port.clear()
        ports = serial.tools.list_ports.comports()
        if ports:
            self.combo_serial_port.addItems([p.device for p in ports])
        else:
            self.combo_serial_port.addItem("No ports found")

    def toggle_signal_connection(self):
        """Single button that connects when idle, disconnects when
        already connected — its label is updated accordingly in
        connect_signal()/disconnect_signal()."""
        if self.is_signal_connected:
            self.disconnect_signal()
        else:
            self.connect_signal()

    def connect_signal(self):
        if self.is_signal_connected:
            return

        port = self.combo_serial_port.currentText()
        if not port or "No ports" in port:
            QtWidgets.QMessageBox.warning(
                self, "No Port Selected",
                "No serial port available. Connect the Arduino and click "
                "Refresh.")
            return

        baud = int(self.combo_baud.currentText())

        self.signal_time  = []
        self.signal_value = []
        self.signal_processed_value = []
        self.smooth_deque.clear()
        self.curve_signal.setData([], [])
        self.curve_processed.setData([], [])

        # Gain resets to the Arduino's own default (128) on every power-up
        # / reconnect, since the firmware always starts at gain=128 in
        # setup() -- keep the combo and our own tracked gain in sync.
        self.current_gain        = HX711_GAIN_DEFAULT
        self.counts_to_mv_factor = counts_to_mv(self.current_gain)
        self.combo_gain.blockSignals(True)
        self.combo_gain.setCurrentIndex(0)
        self.combo_gain.blockSignals(False)

        self.rebuild_processing_chain()

        # Clear any event markers drawn during a previous connection.
        for line in self.event_lines:
            self.plot_signal.removeItem(line)
        self.event_lines = []

        self.lbl_signal_status.setText(f"Status: connecting to {port}...")

        self.serial_worker = SerialWorker(port, baud)
        self.serial_worker.sample_ready.connect(self.on_sample_ready)
        self.serial_worker.error.connect(self.on_signal_error)
        self.serial_worker.start()

        self.is_signal_connected = True
        self.btn_conn_toggle.setText("Disconnect")
        self.btn_tare.setEnabled(True)
        self.combo_gain.setEnabled(True)
        self.combo_serial_port.setEnabled(False)
        self.combo_baud.setEnabled(False)
        self.update_recording_availability()
        self.lbl_signal_status.setText(f"Status: connected to {port} @ {baud} baud")

    def disconnect_signal(self):
        if self.serial_worker is not None:
            # Disconnect first: a sample emitted right as the thread was
            # stopping can still be queued for delivery even after wait()
            # returns.
            try:
                self.serial_worker.sample_ready.disconnect(self.on_sample_ready)
            except TypeError:
                pass
            self.serial_worker.stop()
            self.serial_worker.wait(2000)
            self.serial_worker = None

        self.is_signal_connected = False
        self.btn_conn_toggle.setText("Connect")
        self.btn_tare.setEnabled(False)
        self.combo_gain.setEnabled(False)
        self.combo_serial_port.setEnabled(True)
        self.combo_baud.setEnabled(True)
        self.update_recording_availability()
        self.lbl_signal_status.setText("Status: idle")

    def on_gain_changed(self):
        """
        Sends the gain-change command to the Arduino (128 or 64 only --
        see combo_gain's tooltip) and updates the mV conversion factor to
        match. The firmware re-tares automatically after a gain change,
        so the signal will jump momentarily as the new zero settles --
        this is expected, not an error.
        """
        gain = 128 if self.combo_gain.currentIndex() == 0 else 64
        self.current_gain        = gain
        self.counts_to_mv_factor = counts_to_mv(gain)
        if self.serial_worker is not None:
            self.serial_worker.send_gain(gain)
        self.lbl_signal_status.setText(f"Status: gain set to {gain} (re-taring on device)")

    def rebuild_processing_chain(self):
        """
        (Re)builds the High-pass/Low-pass filter objects to match the
        current checkbox/cutoff settings, and resets the moving-average
        buffer -- called on any change to the processing controls, and
        once at connect time. Resetting on any parameter change avoids
        mixing filter state computed under old settings into new output.
        """
        if self.chk_hp.isChecked():
            self.hp_filter = RealtimeFilter(
                'highpass', self.spin_hp_cutoff.value(), NOMINAL_SAMPLE_RATE_HZ)
        else:
            self.hp_filter = None

        if self.chk_lp.isChecked():
            self.lp_filter = RealtimeFilter(
                'lowpass', self.spin_lp_cutoff.value(), NOMINAL_SAMPLE_RATE_HZ)
        else:
            self.lp_filter = None

        self.spin_hp_cutoff.setEnabled(self.chk_hp.isChecked())
        self.spin_lp_cutoff.setEnabled(self.chk_lp.isChecked())
        self.spin_proc_ma_window.setEnabled(self.chk_proc_ma.isChecked())

        self.proc_ma_deque = deque(maxlen=self.spin_proc_ma_window.value())

    def tare_signal(self):
        if self.serial_worker is not None:
            self.serial_worker.send_tare()
            self.lbl_signal_status.setText("Status: connected (tare sent)")

    def mark_event(self, code=6):
        """
        Marks "now" as an event of the given numeric code (see EVENT_LABELS/
        EVENT_COLORS): 1=Sound, 2=Light, 3=Shock, 4=Trigger 1, 5=Trigger 2,
        6=manual (this window's own "Events" button). Draws a vertical
        line on the live plot in that code's color, and records
        (elapsed_rec, code) — elapsed_rec is in the recording's own
        elapsed-time reference, the same one used by signal_log_rows — so
        write_signal_log() can flag the nearest sample in that code's own
        CSV column (see EVENT_COLUMNS: each event type gets an
        independent 0/1 column, so simultaneous events of different
        codes landing on the same sample don't overwrite each other).

        This can be called from outside this window (e.g. by
        Conditioning Setup, once per stimulus onset, via
        window.mark_event(code)) to log external stimulus events
        alongside the load cell signal.

        Only meaningful while recording — the manual button is disabled
        otherwise, and external callers get a silent no-op — since
        outside a recording session there is no elapsed_rec clock
        (self.record_start_time) to log against.
        """
        if not self.is_recording or self.record_start_time is None:
            # Give explicit feedback instead of silently doing nothing —
            # this is the manual "Events" button's own click path (code
            # defaults to 6), so a status message here is meaningful;
            # external callers (e.g. Conditioning Setup) get the same
            # message reflected in this window's status label.
            self.lbl_signal_status.setText(
                "Status: event ignored — not currently recording")
            return

        color = EVENT_COLORS.get(code, (150, 150, 150, 150))

        # Visual line: placed at the most recent point on the live
        # plot's own timeline (elapsed seconds since the signal was
        # connected — not the same clock as the recording's elapsed_rec,
        # but visually correct since it matches whatever's on screen).
        if self.signal_time:
            line = pg.InfiniteLine(
                pos=self.signal_time[-1], angle=90, movable=False,
                pen=pg.mkPen(color=color, width=2))
            self.plot_signal.addItem(line)
            self.event_lines.append(line)

        # CSV logging: elapsed time since the recording session started —
        # the same reference frame as signal_log_rows entries.
        elapsed_rec = time.perf_counter() - self.record_start_time
        self.event_times_rec.append((elapsed_rec, code))
        label = EVENT_LABELS.get(code, f"code {code}")
        self.lbl_signal_status.setText(
            f"Status: connected — event marked ({label}, "
            f"{len(self.event_times_rec)} total)")

    def on_sample_ready(self, t, raw_count):
        """Slot: runs on the main thread. Converts the raw ADC count to
        millivolts (using the CURRENT gain's factor -- see
        counts_to_mv()), appends the new sample, runs the optional
        real-time processing chain (High-pass -> Low-pass -> Moving
        average), and refreshes both plot curves.

        IMPORTANT: the display-only smoothing (chk_smooth) affects only
        the RAW curve's on-screen value, same as before -- it never
        touches saved data. The separate processing chain (chk_hp/
        chk_lp/chk_proc_ma) runs on the raw mV value and produces the
        "Processed" value that DOES get saved (signal_log_rows) and
        plotted as the second (blue) curve.
        """
        value_mV = raw_count * self.counts_to_mv_factor

        display_value = value_mV
        if self.chk_smooth.isChecked():
            self.smooth_deque.append(value_mV)
            display_value = sum(self.smooth_deque) / len(self.smooth_deque)

        # -- Real-time processing chain: High-pass -> Low-pass -> Moving average --
        processed_value = value_mV
        if self.hp_filter is not None:
            processed_value = self.hp_filter.process(processed_value)
        if self.lp_filter is not None:
            processed_value = self.lp_filter.process(processed_value)
        if self.chk_proc_ma.isChecked():
            self.proc_ma_deque.append(processed_value)
            processed_value = sum(self.proc_ma_deque) / len(self.proc_ma_deque)

        self.signal_time.append(t)
        self.signal_value.append(display_value)
        self.signal_processed_value.append(processed_value)

        # If a recording session is active, also log this sample with a
        # timestamp relative to the SAME start time used for video frame
        # pacing (self.record_start_time) — this is what keeps the load
        # cell log synchronized with the recorded video files. Raw
        # (unsmoothed) mV value AND the processed value are both saved,
        # regardless of the live-plot smoothing setting above.
        if self.is_recording and self.record_start_time is not None:
            elapsed_rec = time.perf_counter() - self.record_start_time
            self.signal_log_rows.append((elapsed_rec, value_mV, processed_value))

        # Soft cap so a very long session doesn't grow memory forever.
        max_points = 20000
        if len(self.signal_time) > max_points:
            self.signal_time            = self.signal_time[-max_points:]
            self.signal_value           = self.signal_value[-max_points:]
            self.signal_processed_value = self.signal_processed_value[-max_points:]

        self.curve_signal.setData(
            self.signal_time, np.array(self.signal_value) * self.display_scale)
        self.curve_processed.setData(
            self.signal_time, np.array(self.signal_processed_value) * self.display_scale)

        if self.combo_display_mode.currentIndex() == 0:   # Scrolling window
            window = self.spin_window_seconds.value()
            self.plot_signal.setXRange(max(0, t - window), max(t, window), padding=0)
        else:                                              # Full history
            self.plot_signal.setXRange(0, max(t, 1), padding=0.02)

        last_display = value_mV * self.display_scale
        self.lbl_signal_status.setText(
            f"Status: connected — {len(self.signal_time)} samples, "
            f"last={last_display:.6g} {self.display_unit}")

    def on_signal_error(self, message):
        QtWidgets.QMessageBox.warning(self, "Signal Connection Error", message)
        self.disconnect_signal()

    def on_display_mode_changed(self):
        """Scrolling-window mode is the only one that uses the window
        size, so keep that control's enabled state in sync."""
        self.spin_window_seconds.setEnabled(
            self.combo_display_mode.currentIndex() == 0)

    def on_display_unit_changed(self, unit_text):
        """
        Changes the FIXED Y-axis display unit/scale for the live plot
        (mV, uV, or V) -- this replaces pyqtgraph's automatic SI-prefix
        axis scaling, which was stacking an extra prefix on top of "mV"
        and showing a confusing multiplier. self.signal_value is always
        stored in mV regardless of this choice; only what's DRAWN (and
        this status label) is rescaled -- the saved CSV is never
        affected, same principle as the smoothing option above.
        """
        self.display_scale = DISPLAY_UNIT_SCALES.get(unit_text, 1.0)
        self.display_unit  = unit_text
        self.plot_signal.setLabel('left', f'Signal ({unit_text})')

        if self.signal_time:
            scaled = np.array(self.signal_value) * self.display_scale
            self.curve_signal.setData(self.signal_time, scaled)
            scaled_proc = np.array(self.signal_processed_value) * self.display_scale
            self.curve_processed.setData(self.signal_time, scaled_proc)

    def on_curve_visibility_changed(self, choice_text):
        """
        Shows/hides the raw (red) and processed (blue) curves per the
        "Show:" combo. Colors are always fixed regardless of this choice
        -- raw is always red, processed is always blue -- this only
        controls which of them are currently drawn, so an identical
        processed-equals-raw overlap (when no processing stage is
        active) doesn't visually read as "the curve changed color".
        """
        show_raw       = choice_text in ("Both", "Raw only")
        show_processed = choice_text in ("Both", "Processed only")
        self.curve_signal.setVisible(show_raw)
        self.curve_processed.setVisible(show_processed)

    def on_smoothing_changed(self):
        """
        Re-size (and reset) the moving-average buffer whenever smoothing
        is toggled or its window size changes. Resetting avoids mixing
        samples collected under a different window size into the
        average, which would otherwise bias the next few displayed
        points after a change.
        """
        self.smooth_deque = deque(maxlen=self.spin_smooth_window.value())
        self.spin_smooth_window.setEnabled(self.chk_smooth.isChecked())

    # ---------------------------------------------------------------
    # RECORDING

    def choose_output_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose Output Folder")
        if folder:
            self.output_dir = folder
            self.lbl_output_dir.setText(f"Output folder: {folder}")

    def _confirm_partial_recording(self, title, message):
        """
        Show an Ok/Cancel confirmation when only one of video or signal
        is active (not both). Returns True if the user chose to proceed
        (Ok), False if they cancelled — in which case start_recording()
        aborts without starting anything.
        """
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message + "\n\nContinue?")
        msg_box.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.Ok)
        return msg_box.exec_() == QtWidgets.QMessageBox.Ok

    def start_recording(self):
        if self.is_recording:
            return

        if not self.is_streaming and not self.is_signal_connected:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Record",
                "Neither a camera nor the load cell signal is currently "
                "active. Start at least one of them before recording.")
            return

        if not self.output_dir:
            self.choose_output_dir()
            if not self.output_dir:
                return   # user cancelled the folder picker

        if self.is_streaming and not self.is_signal_connected:
            if not self._confirm_partial_recording(
                "Signal Not Connected",
                "The load cell signal is not connected — this recording "
                "will contain video only."):
                return
        elif self.is_signal_connected and not self.is_streaming:
            if not self._confirm_partial_recording(
                "No Camera Active",
                "No camera is currently streaming — this recording will "
                "contain the load cell log only, no video."):
                return

        # Shared by every file written in this session (all camera videos
        # + the load cell log), so they're trivially easy to group later.
        self.session_timestamp = time.strftime("%Y%m%d_%H%M%S")

        self.video_writers   = {}   # opened lazily per camera, on its first frame
        self.frames_written  = {}   # camera index -> frames written so far
        self.signal_log_rows = []   # (elapsed_s, value_mV) logged by on_sample_ready()
        self.event_times_rec = []   # reset for this session's own elapsed_rec clock
        self.is_recording     = True
        self.record_start_time = time.perf_counter()

        self.btn_start_rec.setEnabled(False)
        self.btn_stop_rec.setEnabled(True)
        self.combo_format.setEnabled(False)
        self.spin_record_fps.setEnabled(False)
        self.btn_events.setEnabled(True)
        self.lbl_rec_status.setText("Recording... 00:00")
        self.rec_clock_timer.start(200)

    def stop_recording(self):
        self.rec_clock_timer.stop()
        self.is_recording = False

        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}

        self.write_signal_log()

        self.update_recording_availability()
        self.btn_stop_rec.setEnabled(False)
        self.combo_format.setEnabled(True)
        self.spin_record_fps.setEnabled(True)
        self.btn_events.setEnabled(False)
        self.lbl_rec_status.setText("Not recording")

    def update_recording_availability(self):
        """
        "Start Recording" should be usable whenever there's at least ONE
        active source (camera streaming and/or signal connected) —
        recording no longer requires both. Whatever is actually active
        at the moment Start is pressed gets recorded: video only,
        signal only, or both. Call this after anything that changes
        streaming/connection state.
        """
        if self.is_recording:
            self.btn_start_rec.setEnabled(False)
            return
        self.btn_start_rec.setEnabled(self.is_streaming or self.is_signal_connected)

    def write_signal_log(self):
        """
        Save the load cell samples collected during the just-finished
        recording session to a .csv file (plain text, '#'-prefixed
        header lines followed by a standard CSV table — readable in any
        text editor and directly loadable with e.g. pandas.read_csv
        using comment='#').

        Timestamps are seconds elapsed since self.record_start_time —
        the SAME zero-point used for the video files' frame pacing — so
        this file is directly synchronized with the recorded videos.

        Two signal columns are written:
          Value (mV)     -- ALWAYS the raw, unfiltered mV-converted
                             signal (per the current HX711 gain -- see
                             counts_to_mv()). This is the reference
                             column for any precise timing/synchrony
                             analysis (e.g. aligning to Conditioning
                             Setup's stimulus event markers).
          Processed (mV) -- the same signal after the real-time
                             High-pass/Low-pass/Moving-average chain
                             (whichever stages were enabled -- see the
                             header comment below for exactly which).
                             This is CAUSAL filtering (real phase delay,
                             unlike MATLAB's zero-phase filtfilt used for
                             offline analysis) -- good for inspecting
                             signal dynamics, not for precise event
                             timing.
        """
        if not self.output_dir or self.session_timestamp is None:
            return

        # Video-only session (signal never connected, nothing logged) —
        # skip writing an empty/misleading .csv file.
        if not self.signal_log_rows and self.serial_worker is None:
            return

        n = len(self.signal_log_rows)
        if n > 0:
            duration = self.signal_log_rows[-1][0]
            avg_rate = (n / duration) if duration > 0 else 0.0
        else:
            duration = 0.0
            avg_rate = 0.0

        # Build one independent 0/1 column per event type (Sound, Light,
        # Shock, Trigger1, Trigger2, Manual) instead of a single
        # categorical column. This is what lets simultaneous events (e.g.
        # Sound and Light onset at the same instant) both be flagged on
        # the same sample without one overwriting the other.
        #
        # Each column still uses nearest-neighbor matching against the
        # sample timestamps, since an event lands at an arbitrary time
        # between two samples, not exactly on one.
        sample_times = [row[0] for row in self.signal_log_rows]
        event_columns = {code: [0] * n for code in EVENT_COLUMNS}
        for et, code in self.event_times_rec:
            if not sample_times:
                break
            nearest_idx = min(
                range(len(sample_times)),
                key=lambda i: abs(sample_times[i] - et))
            event_columns.setdefault(code, [0] * n)[nearest_idx] = 1

        filename = f"LoadCell1_{self.session_timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        col_codes = list(EVENT_COLUMNS.keys())
        col_header = ",".join(EVENT_COLUMNS[c] for c in col_codes)

        # Describe exactly which processing stages were active, so the
        # "Processed" column is reproducible/interpretable later without
        # having to guess the settings used at the time.
        proc_parts = []
        if self.chk_hp.isChecked():
            proc_parts.append(f"high-pass {self.spin_hp_cutoff.value():.2f}Hz")
        if self.chk_lp.isChecked():
            proc_parts.append(f"low-pass {self.spin_lp_cutoff.value():.2f}Hz")
        if self.chk_proc_ma.isChecked():
            proc_parts.append(f"moving-avg {self.spin_proc_ma_window.value()} samples")
        proc_desc = " -> ".join(proc_parts) if proc_parts else "none (passthrough of raw)"

        try:
            with open(filepath, "w") as f:
                f.write("# Behavior Recording - Load Cell Log\n")
                f.write("# Channel: Load Cell 1\n")
                if self.serial_worker is not None:
                    f.write(f"# Serial port: {self.serial_worker.port}\n")
                    f.write(f"# Baud rate: {self.serial_worker.baud}\n")
                else:
                    f.write("# Serial port: (signal was not connected during this recording)\n")
                f.write(f"# Samples: {n}\n")
                f.write(f"# Duration (s): {duration:.3f}\n")
                f.write(f"# Average sample rate (Hz): {avg_rate:.2f}\n")
                f.write(f"# Events marked: {len(self.event_times_rec)}\n")
                f.write(f"# Value (mV) units: HX711 Channel A gain={self.current_gain}, "
                        f"assumed AVDD={AVDD_VOLTS}V, {self.counts_to_mv_factor:.9g} mV/count. "
                        f"NOT raw ADC counts, NOT a grams calibration.\n")
                f.write(f"# Processed (mV): real-time (causal) chain -- {proc_desc}. "
                        f"Has phase delay vs. Value (mV); use Value (mV) for precise timing.\n")
                f.write(f"Time (s),Value (mV),Processed (mV),{col_header}\n")
                for i, (t, v, p) in enumerate(self.signal_log_rows):
                    flags = ",".join(str(event_columns[c][i]) for c in col_codes)
                    f.write(f"{t:.4f},{v:.6g},{p:.6g},{flags}\n")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Signal Log Error",
                f"Could not save the load cell log:\n{e}")

    def write_frame_to_disk(self, idx, pos, frame):
        """
        Write frame(s) for camera `idx`, paced against real elapsed time
        rather than one write per camera callback.

        A video file stores duration implicitly as
        (frame_count / declared_fps). Since a camera's actual delivery
        rate rarely matches the FPS declared in the file header exactly,
        writing one frame per callback can make the recorded file play
        back shorter or longer than the real session (e.g. a real 10s
        recording turning into an 8s file).

        To avoid this, we track how many frames SHOULD exist by now given
        the declared FPS and real elapsed time, and duplicate the latest
        frame as many times as needed to catch up. This keeps the final
        file's duration accurate to the real elapsed recording time
        regardless of how the camera's actual frame rate fluctuates.
        """
        writer = self.video_writers.get(idx)
        if writer is None:
            ext, fourcc_str = VIDEO_FORMATS[self.combo_format.currentText()]
            fourcc    = cv2.VideoWriter_fourcc(*fourcc_str)
            fps       = self.spin_record_fps.value()
            h, w      = frame.shape[:2]
            # Shared timestamp (set once in start_recording) so all video
            # files and the load cell log from this session line up by
            # filename.
            filename  = f"Camera{pos + 1}_idx{idx}_{self.session_timestamp}{ext}"
            filepath  = os.path.join(self.output_dir, filename)

            writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
            if not writer.isOpened():
                QtWidgets.QMessageBox.critical(
                    self, "Recording Error",
                    f"Could not create video file for Camera {pos + 1}:\n{filepath}\n\n"
                    "Try a different format (e.g. AVI/MJPG).")
                self.stop_recording()
                return
            self.video_writers[idx]  = writer
            self.frames_written[idx] = 0

        target_fps = self.spin_record_fps.value()
        elapsed    = time.perf_counter() - self.record_start_time
        expected   = int(elapsed * target_fps)

        while self.frames_written[idx] < expected:
            writer.write(frame)
            self.frames_written[idx] += 1

    def update_recording_clock(self):
        if self.record_start_time is None:
            return
        elapsed = time.perf_counter() - self.record_start_time
        mm = int(elapsed // 60)
        ss = elapsed - mm * 60
        self.lbl_rec_status.setText(f"Recording... {mm:02d}:{ss:04.1f}")

    # ---------------------------------------------------------------
    # CLEAN SHUTDOWN

    def closeEvent(self, event):
        self.stop_cameras()
        self.disconnect_signal()
        super().closeEvent(event)


# ---------------------------------------------------------------
# ENTRY POINT

if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = BehaviorRecording()
    window.show()

    app.exec_()
