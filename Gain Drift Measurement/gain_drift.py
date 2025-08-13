
"""
SDR Gain Drift Measurement Script

This script performs automated gain drift measurements using various SDRs including HackRF, PlutoSDR, RTL-SDR, LimeSDR, and USRP. The system continuously
monitors received power over time to characterize gain stability and drift characteristics.

For detailed setup instructions and measurement procedures, refer to the accompanying
gain_drift_measurement_guide.md documentation in this folder.
"""


from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import soapy
import sip
import threading
import os
import datetime
import time
from plot_util import plot_data

# ===========================================================
# Parameters
# ===========================================================

MEASUREMENT_DURATION_MIN = 15  # Total measurement duration in minutes
MEASUREMENT_INTERVAL_S = 4     # Time between measurement points in seconds
BUFFERTIME_BEFORE_MEASURING = 60  # Settling time for moving average filter 

# SDR device configuration
SDR_TYPE = "hackrf"  # Supported types: plutosdr, rtlsdr, hackrf, lime, usrp
GAIN = 60  # SDR receiver gain setting in dB

class GainDriftMeasurement(gr.top_block, Qt.QWidget):
    """
    GNU Radio flowgraph class for SDR gain drift measurements.
    
    Implements a power measurement system that continuously monitors received power
    over time to characterize SDR gain stability and drift characteristics.
    """

    def __init__(self):
        """
        Initialize GNU Radio flowgraph and Qt GUI for gain drift measurement.
        
        Sets up SDR source, signal processing blocks, GUI elements, and measurement
        parameters for automated power monitoring over time.
        """
        gr.top_block.__init__(self, f'{SDR_TYPE} Gain={GAIN}db', catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle(f'{SDR_TYPE} Gain={GAIN}db')
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "gain_drift_measurement")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        # Flowgraph variables - SDR and measurement parameters
        self.SDR_TYPE = SDR_TYPE 
        self.samp_rate = samp_rate = 2e6  # Sample rate: 2 MHz
        self.offset = offset = 0  # Power offset correction
        self.gain = GAIN  # SDR gain setting
        self.freq_correction = freq_correction = 22  
        self.channel_freq = channel_freq = 1e9  # Center frequency: 1 GHz
        self.measurement_interval_s = MEASUREMENT_INTERVAL_S
        self.measurement_duration_min = MEASUREMENT_DURATION_MIN
        self.total_measurements = int((self.measurement_duration_min * 60) / self.measurement_interval_s)
        self.avg_samples = int(samp_rate * self.measurement_interval_s)  # Samples per measurement
        self.AVG = AVG = self.avg_samples

        # GNU Radio signal processing blocks
        # SDR source configuration based on device type
        if SDR_TYPE == "usrp":
            # USRP configuration using UHD driver
            from gnuradio import uhd
            self.source = uhd.usrp_source(
                ",".join(("", '')),
                uhd.stream_args(cpu_format="fc32", channels=[0]),
            )
            self.source.set_samp_rate(samp_rate)
            self.source.set_center_freq(channel_freq)
            self.source.set_gain(self.gain)
            self.source.set_bandwidth(samp_rate)
            self.source.set_antenna("TX/RX", 0)
        elif SDR_TYPE in ["plutosdr", "rtlsdr", "hackrf", "lime"]:
            # SoapySDR configuration for other SDR devices
            dev = f"driver={SDR_TYPE}"
            dev_args = ""
            stream_args = ""
            tune_args = [""]
            other_settings = [""]

            self.source = soapy.source(dev, "fc32", 1, dev_args, stream_args, tune_args, other_settings)
            self.source.set_sample_rate(0, samp_rate)
            self.source.set_frequency(0, channel_freq)
            self.source.set_bandwidth(0, samp_rate)
            self.source.set_gain(0, self.gain)

            # Device-specific settings
            if SDR_TYPE == "plutosdr":
                self.source.set_gain_mode(0, False)  # Manual gain control
            if SDR_TYPE == "rtlsdr":
                self.source.set_frequency_correction(0, freq_correction)  # PPM correction
        else:
            raise ValueError(f"Unsupported SDR_TYPE: {SDR_TYPE}")
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_HORIZ,
            1,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title("measured power")

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_0.set_min(i, -1)
            self.qtgui_number_sink_0.set_max(i, 1)
            self.qtgui_number_sink_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels[i])
            self.qtgui_number_sink_0.set_unit(i, units[i])
            self.qtgui_number_sink_0.set_factor(i, factor[i])

        self.qtgui_number_sink_0.enable_autoscale(False)
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            channel_freq, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, 1, 0)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(AVG, (1/AVG), 4000, 1)
        # self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*1, '/Users/danielschreiber/Documents/Uni/Semester/SS25/SIERRA/Radio Astronomy/ra100-receiver-characterisation/02_SDR_dbfs/test.csv', False)
        # self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff((-self.gain-offset))


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        # self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.qtgui_number_sink_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.source, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.source, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.source, 0), (self.qtgui_time_sink_x_0, 0))

        # Prepare output file for live data logging
        now = datetime.datetime.now()
        self.timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(".", SDR_TYPE)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"{self.timestamp_str}.csv")

        try:
            self.output_file = open(self.output_path, "w", buffering=1)  # line-buffered
            self.output_file.write("timestamp,measured_power_dBm,standard_deviation\n")
        except Exception as e:
            print(f"Error opening file: {e}")
            self.output_file = None

        # Probe block for measurement data extraction
        self.probe = blocks.probe_signal_f()
        self.connect((self.blocks_moving_average_xx_0, 0), (self.probe, 0))

        # Start background thread for measurement execution
        self.measurement_thread = threading.Thread(target=self._run_measurement_loop)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
    def _run_measurement_loop(self):
        """
        Execute continuous power measurement loop for gain drift analysis.
        
        Collects power measurements at regular intervals over the specified duration,
        calculates statistics, and saves data to CSV file for drift analysis.
        """
        print(f"Waiting for settling time of {BUFFERTIME_BEFORE_MEASURING} seconds...")
        time.sleep(BUFFERTIME_BEFORE_MEASURING)  # Settling time for moving average filter
        print("Measurement started.")

        start_time = time.time()  # Reference time for relative timestamps

        for i in range(self.total_measurements):
            samples = []
            t_start = time.time()
            # Collect samples over measurement interval
            while time.time() - t_start < self.measurement_interval_s:
                value = self.probe.level()
                samples.append(value)
                time.sleep(0.05)  # ~20 Hz sampling rate

            elapsed_seconds = time.time() - start_time

            # Calculate mean and standard deviation for this measurement period
            if samples:
                mean_val = sum(samples) / len(samples)
                std_dev = (sum((x - mean_val) ** 2 for x in samples) / len(samples)) ** 0.5
                line = f"{elapsed_seconds:.1f},{mean_val:.6f},{std_dev:.6f}\n"
            else:
                line = f"{elapsed_seconds:.1f},NaN,NaN\n"

            if self.output_file:
                self.output_file.write(line)
                self.output_file.flush()

            print(line.strip())
        print("Measurement completed.")
        if self.output_file:
            self.output_file.close()
            plot_data(self.output_path)

    def closeEvent(self, event):
        """
        Handle application close event with proper cleanup.
        
        Saves GUI settings, stops GNU Radio flowgraph, and closes output files.
        """
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "gain_drift_measurement")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        if hasattr(self, "output_file") and self.output_file:
            self.output_file.close()
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(self.channel_freq, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.source.set_sample_rate(0, self.samp_rate)
        self.source.set_bandwidth(0, self.samp_rate)

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset
        self.blocks_add_const_vxx_0.set_k((-self.gain-self.offset))

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.blocks_add_const_vxx_0.set_k((-self.gain-self.offset))
        self.source.set_gain(0, min(max(self.gain, 0.0), 73.0))

    def get_freq_correction(self):
        return self.freq_correction

    def set_freq_correction(self, freq_correction):
        self.freq_correction = freq_correction

    def get_channel_freq(self):
        return self.channel_freq

    def set_channel_freq(self, channel_freq):
        self.channel_freq = channel_freq
        self.qtgui_freq_sink_x_0.set_frequency_range(self.channel_freq, self.samp_rate)
        self.source.set_frequency(0, self.channel_freq)

    def get_AVG(self):
        return self.AVG

    def set_AVG(self, AVG):
        self.AVG = AVG
        self.blocks_moving_average_xx_0.set_length_and_scale(self.AVG, (1/self.AVG))




def main(top_block_cls=GainDriftMeasurement, options=None):
    """
    Main application entry point for gain drift measurement.
    
    Initializes Qt application, creates and starts the GNU Radio flowgraph,
    and handles application lifecycle and signal processing.
    """

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()
    print(f"Measurement running for {MEASUREMENT_DURATION_MIN} minutes. Results: {tb.output_path}")

if __name__ == '__main__':
    main()
