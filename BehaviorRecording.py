"""
BehaviorRecording.py

Behavior Recording
===================
Multi-camera video acquisition end load cell recording for behavioral conditioning setups.

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

PLANNED (not yet implemented):
    - A second, third, and fourth independent signal channel (multiple
      load cells), mirroring the camera grid's approach.
    - Event marking and behavioral epoch analysis.
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
    PyQt5, pyqtgraph, opencv-python, numpy, pyserial

AUTHOR:
    Flavio Mourao (mourao.fg@gmail.com)

Started: 07/2026
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

MAX_CAMERAS = 4

VIDEO_FORMATS = {
    "AVI (MJPG)": (".avi", "MJPG"),
    "MP4 (mp4v)": (".mp4", "mp4v"),
}


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

    def run(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.error.emit(self.index, f"Could not open camera {self.index}.")
            return

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
    """
    sample_ready = QtCore.pyqtSignal(float, float)   # elapsed seconds, value
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

    def stop(self):
        """Ask the run() loop to exit; it closes the port on its way out."""
        self._running = False


class BehaviorRecording(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Behavior Recording")
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
        self.signal_log_rows   = []     # (elapsed_s_since_record_start, value), logged while recording

        self.rec_clock_timer = QtCore.QTimer()
        self.rec_clock_timer.timeout.connect(self.update_recording_clock)

        self.serial_worker      = None
        self.is_signal_connected = False
        self.signal_time  = []   # elapsed seconds since connection, per sample (for the live plot)
        self.signal_value = []   # display value per sample (raw, or smoothed if enabled)
        self.smooth_deque = deque(maxlen=5)   # rolling buffer for the optional moving average

        self.setup_gui()

    # ---------------------------------------------------------------
    # GUI SETUP

    def setup_gui(self):
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
        self.plot_signal.setBackground('w')
        self.plot_signal.setLabel('bottom', 'Time', 's')
        self.curve_signal = self.plot_signal.plot(pen=pg.mkPen('#0072BD', width=1))

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
        port_layout.addWidget(self.combo_baud)
        port_layout.addStretch()

        # -- Tare --
        tare_layout = QtWidgets.QHBoxLayout()
        self.btn_tare = QtWidgets.QPushButton("Tare")
        self.btn_tare.clicked.connect(self.tare_signal)
        self.btn_tare.setEnabled(False)
        tare_layout.addWidget(self.btn_tare)
        tare_layout.addStretch()

        self.lbl_signal_status = QtWidgets.QLabel("Status: idle")

        plot_layout.addWidget(self.plot_signal, 1)
        plot_layout.addLayout(display_layout)
        plot_layout.addLayout(smooth_layout)
        plot_layout.addLayout(port_layout)
        plot_layout.addLayout(tare_layout)
        plot_layout.addWidget(self.lbl_signal_status)
        plot_group.setLayout(plot_layout)

        self.populate_serial_ports()

        # -------------------------------------------------------
        main_layout.addWidget(cam_group, 1)
        main_layout.addWidget(plot_group, 1)

        # -------------------------------------------------------
        # BOTTOM BAR: Recording (governs camera video + load cell log
        # together — kept visually separate from the Cameras panel above
        # since "Start Recording" synchronizes both).
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
        self.smooth_deque.clear()
        self.curve_signal.setData([], [])

        self.lbl_signal_status.setText(f"Status: connecting to {port}...")

        self.serial_worker = SerialWorker(port, baud)
        self.serial_worker.sample_ready.connect(self.on_sample_ready)
        self.serial_worker.error.connect(self.on_signal_error)
        self.serial_worker.start()

        self.is_signal_connected = True
        self.btn_conn_toggle.setText("Disconnect")
        self.btn_tare.setEnabled(True)
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
        self.combo_serial_port.setEnabled(True)
        self.combo_baud.setEnabled(True)
        self.update_recording_availability()
        self.lbl_signal_status.setText("Status: idle")

    def tare_signal(self):
        if self.serial_worker is not None:
            self.serial_worker.send_tare()
            self.lbl_signal_status.setText("Status: connected (tare sent)")

    def on_sample_ready(self, t, value):
        """Slot: runs on the main thread. Appends the new sample and
        refreshes the plot according to the selected display mode.

        IMPORTANT: smoothing (if enabled) is applied only to the value
        stored for the live plot. The value logged for recording
        (signal_log_rows, below) always uses the raw, unaveraged `value`
        — smoothing never touches the saved data.
        """
        display_value = value
        if self.chk_smooth.isChecked():
            self.smooth_deque.append(value)
            display_value = sum(self.smooth_deque) / len(self.smooth_deque)

        self.signal_time.append(t)
        self.signal_value.append(display_value)

        # If a recording session is active, also log this sample with a
        # timestamp relative to the SAME start time used for video frame
        # pacing (self.record_start_time) — this is what keeps the load
        # cell log synchronized with the recorded video files. Always the
        # raw value, regardless of the live-plot smoothing setting above.
        if self.is_recording and self.record_start_time is not None:
            elapsed_rec = time.perf_counter() - self.record_start_time
            self.signal_log_rows.append((elapsed_rec, value))

        # Soft cap so a very long session doesn't grow memory forever.
        max_points = 20000
        if len(self.signal_time) > max_points:
            self.signal_time  = self.signal_time[-max_points:]
            self.signal_value = self.signal_value[-max_points:]

        self.curve_signal.setData(self.signal_time, self.signal_value)

        if self.combo_display_mode.currentIndex() == 0:   # Scrolling window
            window = self.spin_window_seconds.value()
            self.plot_signal.setXRange(max(0, t - window), max(t, window), padding=0)
        else:                                              # Full history
            self.plot_signal.setXRange(0, max(t, 1), padding=0.02)

        self.lbl_signal_status.setText(
            f"Status: connected — {len(self.signal_time)} samples, last={value:.0f}")

    def on_signal_error(self, message):
        QtWidgets.QMessageBox.warning(self, "Signal Connection Error", message)
        self.disconnect_signal()

    def on_display_mode_changed(self):
        """Scrolling-window mode is the only one that uses the window
        size, so keep that control's enabled state in sync."""
        self.spin_window_seconds.setEnabled(
            self.combo_display_mode.currentIndex() == 0)

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
        self.signal_log_rows = []   # (elapsed_s, value) logged by on_sample_ready()
        self.is_recording     = True
        self.record_start_time = time.perf_counter()

        self.btn_start_rec.setEnabled(False)
        self.btn_stop_rec.setEnabled(True)
        self.combo_format.setEnabled(False)
        self.spin_record_fps.setEnabled(False)
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

        filename = f"LoadCell1_{self.session_timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

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
                f.write("Time (s),Value\n")
                for t, v in self.signal_log_rows:
                    f.write(f"{t:.4f},{v:.6g}\n")
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