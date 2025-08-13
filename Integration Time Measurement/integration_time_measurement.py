"""
SDR Integration Time Characterization Tool

This script performs automated power measurements and Allan Deviation analysis across
multiple gain settings using SDRs to find out the optimal integration time for processing the data. Supports RTLSDR,
HackRF, PlutoSDR, LimeSDR, and USRP.

For detailed measurement procedures, calibration requirements, and theoretical background,
refer to the integration_time_measurement_guide.md in the same folder.
"""

import time
import numpy as np
import allantools
from datetime import datetime
import os  # For directory and file path management
import threading
import matplotlib.pyplot as plt  # For plotting Allan Deviation results
import re  # For robust gain extraction from filename pattern matching

# GNU Radio framework and Qt GUI components
import sys
import signal
from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sip
from gnuradio import eng_notation
from gnuradio import soapy  # SoapySDR interface for multiple SDR platforms
from gnuradio import uhd    # USRP Hardware Driver for Ettus Research devices
from gnuradio import filter # DC blocking filter for signal conditioning


# ##################################################
# Parameters
# ##################################################

# --- SDR Device Selection ---
# Supported platforms: "RTLSDR", "PLUTOSDR", "LIMESDR", "USRPSDR", "HACKRF"
SDR_TYPE = "USRPSDR"  # Primary SDR platform for measurements

# --- RF and DSP Parameters ---
SAMP_RATE = 2.0e6         # ADC sampling rate in Hz
CHANNEL_FREQ = 1e9        # RF center frequency in Hz (1 GHz)
PPM_CORRECTION = 22       # Crystal oscillator frequency correction in ppm
CALIBRATION_OFFSET = 0    # System calibration offset in dB
AVG_LENGTH = 1000         # Moving average filter length for power smoothing

# Power measurement scaling: 10*log10 for power (dBm), 20*log10 for voltage (dBV)
LOG_MULTIPLY = 10  # Use 10 for power measurements in dBm units

# --- Allan Deviation Analysis Parameters ---
ALLAN_T_MIN = 0.1             # Minimum integration time for stability analysis (seconds)
ALLAN_T_MAX = 100             # Maximum integration time for stability analysis (seconds)
ALLAN_STEPS_PER_DECADE = 10   # Logarithmic sampling density for tau values

# Data collection duration to ensure sufficient statistics for maximum tau
# Allan Deviation requires data spanning at least 2*tau_max for reliable calculation
REQUIRED_DATA_COLLECTION_TIME_S = ALLAN_T_MAX * 2.1  # Added 10% safety margin

# --- Gain Sweep Configuration ---
MIN_SDR_GAIN = 40     # Start gain value in dB
MAX_SDR_GAIN = 70     # End gain value in dB
SDR_GAIN_STEP = 10    # Gain increment step size in dB


class integration_time_measurement(gr.top_block, Qt.QWidget):
    """
    GNU Radio flowgraph for SDR power measurements with real-time visualization.
    
    Implements a complete RF signal processing chain from SDR input to calibrated
    power measurements, including DC blocking, magnitude squaring, logarithmic
    conversion, and moving average filtering for Allan Deviation analysis.
    """

    def __init__(self, current_sdr_gain, base_output_dir):
        """
        Initialize the GNU Radio flowgraph with specified gain and output directory.
        
        Args:
            current_sdr_gain: RF gain setting in dB for the selected SDR
            base_output_dir: Base directory path for saving measurement results
        """
        gr.top_block.__init__(self, f"Combined Power Measurement (Gain: {current_sdr_gain} dB)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle(f"Combined Power Measurement (Gain: {current_sdr_gain} dB) with SDR Selection and Allan Deviation")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        
        # Configure Qt layout hierarchy for GUI components
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

        # Load Qt application settings for window geometry persistence
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "integration_time_measurement")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        
        # Threading synchronization for flowgraph startup coordination
        self.flowgraph_started = threading.Event()
        self.sdr_source_block = None  # Placeholder for dynamically selected SDR

        ##################################################
        # GNU Radio Variables
        ##################################################
        self.samp_rate = samp_rate = SAMP_RATE
        self.offset = offset = CALIBRATION_OFFSET
        self.gain = current_sdr_gain  # RF gain setting for current measurement
        self.freq_correction = freq_correction = PPM_CORRECTION
        self.channel_freq = channel_freq = CHANNEL_FREQ
        self.AVG = AVG = AVG_LENGTH
        self.log_multiply = LOG_MULTIPLY  # Power scaling factor for dBm conversion
        self.base_output_dir = base_output_dir  # Directory for measurement data storage

        ##################################################
        # GNU Radio Signal Processing Blocks
        ##################################################

        # Initialize SDR source block based on platform selection
        self._initialize_sdr_source()

        # DC blocking filter to remove DC offset from I/Q samples
        self.blocks_dc_blocker_xx_0 = filter.dc_blocker_cc(32, True)

        # Real-time power display with calibrated dBm readings
        self.qtgui_number_sink_0 = qtgui.number_sink(gr.sizeof_float,0,qtgui.NUM_GRAPH_HORIZ,1,None )
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title("Measured Power (dBm)")

        labels_number = ['Measured Power']
        units_number = ['dBm']
        colors_number = [("black", "black")]
        factor_number = [1]

        for i in range(1):
            self.qtgui_number_sink_0.set_min(i, -100) # Adjusted min/max for typical power range
            self.qtgui_number_sink_0.set_max(i, 10)
            self.qtgui_number_sink_0.set_color(i, colors_number[i][0], colors_number[i][1])
            if len(labels_number[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data %d" % (i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels_number[i])
            self.qtgui_number_sink_0.set_unit(i, units_number[i])
            self.qtgui_number_sink_0.set_factor(i, factor_number[i])

        self.qtgui_number_sink_0.enable_autoscale(False)
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)

        # Frequency domain visualization for signal analysis
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
             2048, #size
             window.WIN_BLACKMAN_hARRIS, #wintype
             channel_freq, #fc
             samp_rate, #bw
             "Frequency Domain", #name
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
        
        labels_freq = ['Signal']
        widths_freq = [1]
        colors_freq = ["blue"]
        alphas_freq = [1.0]
        styles_freq = [1] 
        for i in range(1):
            if len(labels_freq[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data %d" % (i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels_freq[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths_freq[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors_freq[i])
            self.qtgui_freq_sink_x_0.set_line_style(i, styles_freq[i]) 
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas_freq[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)

        # Signal processing chain for power measurement
        # Convert complex I/Q samples to magnitude squared (power)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        
        # Convert power to dB scale using logarithmic transformation
        # LOG_MULTIPLY=10 for power (dBm), LOG_MULTIPLY=20 for voltage (dBV)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(self.log_multiply, 1, 0)
        
        # Apply calibration correction: convert dBFS to dBm
        # Subtracts RF gain and system offset for absolute power calibration
        self.blocks_add_const_vxx_0 = blocks.add_const_ff((-self.gain - self.offset))
        
        # Moving average filter for power smoothing and noise reduction
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(self.AVG, (1.0 / self.AVG), 4000, 1)
        
        # Decimation filter to reduce data rate for Allan Deviation analysis
        # Keeps one sample every AVG_LENGTH samples to match desired time resolution
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_float, self.AVG)

        # In-memory data collection for offline Allan Deviation processing
        # Replaces file sink to prevent disk I/O during real-time operation
        self.vector_sink = blocks.vector_sink_f()

        ##################################################
        # Signal Flow Connections
        ##################################################
        
        # Primary signal path: SDR → DC Blocker → Power Processing
        self.connect((self.sdr_source_block, 0), (self.blocks_dc_blocker_xx_0, 0))
        self.connect((self.blocks_dc_blocker_xx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        
        # Parallel connection for frequency domain visualization
        self.connect((self.blocks_dc_blocker_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))

        # Power measurement processing chain
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_moving_average_xx_0, 0))
        
        # Decimation and data collection for Allan Deviation analysis
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_keep_one_in_n_0, 0))
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.vector_sink, 0))

        # Real-time power display
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.qtgui_number_sink_0, 0))

    def _initialize_sdr_source(self):
        """
        Initialize the appropriate SDR source block based on the global SDR_TYPE configuration.
        
        Configures platform-specific parameters including sample rate, center frequency,
        gain settings, and hardware-specific features for each supported SDR type.
        """
        samp_rate = self.samp_rate
        channel_freq = self.channel_freq
        gain = self.gain
        freq_correction = self.freq_correction

        if SDR_TYPE == "RTLSDR":
            print("Initializing RTL-SDR source...")
            self.sdr_source_block = soapy.source('driver=rtlsdr', "fc32", 1, '', 'bufflen=16384', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_frequency_correction(0, freq_correction)  # PPM correction for crystal accuracy
            self.sdr_source_block.set_gain_mode(0, False)  # Manual gain control
            self.sdr_source_block.set_gain(0, 'TUNER', gain)  # RF tuner gain stage
            
        elif SDR_TYPE == "HACKRF":
            print("Initializing HackRF source...")
            self.sdr_source_block = soapy.source('driver=hackrf', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, 'AMP', True)  # Enable RF amplifier
            self.sdr_source_block.set_gain(0, 'LNA', min(max(16, 0.0), 40.0))  # LNA gain: 0-40 dB
            self.sdr_source_block.set_gain(0, 'VGA', min(max(gain, 0.0), 62.0))  # VGA gain: 0-62 dB
            
        elif SDR_TYPE == "LIMESDR":
            print("Initializing LimeSDR source...")
            self.sdr_source_block = soapy.source('driver=lime', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_frequency_correction(0, freq_correction)
            self.sdr_source_block.set_gain(0, min(max(gain, -12.0), 61.0))  # Overall gain: -12 to 61 dB
            
        elif SDR_TYPE == "PLUTOSDR":
            print("Initializing PlutoSDR source...")
            self.sdr_source_block = soapy.source('driver=plutosdr', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_gain_mode(0, False)  # Manual gain control
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, min(max(gain, 0.0), 73.0))  # RX gain: 0-73 dB
            
        elif SDR_TYPE == "USRPSDR":
            print("Initializing USRP SDR source...")
            self.sdr_source_block = uhd.usrp_source(
                ",".join(("", '')),  # Auto-detect USRP device
                uhd.stream_args(
                    cpu_format="fc32",  # Complex float 32-bit samples
                    args='',
                    channels=list(range(0,1)),  # Single RX channel
                ),
            )
            self.sdr_source_block.set_samp_rate(samp_rate)
            self.sdr_source_block.set_time_unknown_pps(uhd.time_spec(0))  # Initialize time reference
            self.sdr_source_block.set_center_freq(channel_freq, 0)
            self.sdr_source_block.set_antenna("TX/RX", 0)  # Use TX/RX antenna port
            self.sdr_source_block.set_bandwidth(samp_rate, 0)
            self.sdr_source_block.set_rx_agc(False, 0)  # Disable automatic gain control
            self.sdr_source_block.set_gain(gain, 0)  # Manual RX gain setting
        else:
            raise ValueError(f"Unknown SDR_TYPE: '{SDR_TYPE}'. Please choose from: 'RTLSDR', 'PLUTOSDR', 'LIMESDR', 'USRPSDR', 'HACKRF'.")

    def start(self):
        """Start the GNU Radio flowgraph and set running state flag."""
        self.running = True
        super().start()

    def stop(self):
        """Stop the GNU Radio flowgraph and clear running state flag."""
        self.running = False
        super().stop()

    def closeEvent(self, event):
        """Handle Qt window close event with proper cleanup and settings persistence."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    def get_samp_rate(self):
        """Get current ADC sampling rate in Hz."""
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        """
        Set ADC sampling rate and update all dependent blocks.
        
        Args:
            samp_rate: New sampling rate in Hz
        """
        self.samp_rate = samp_rate
        self.qtgui_freq_sink_x_0.set_frequency_range(self.channel_freq, self.samp_rate)
        
        # Update sampling rate for each SDR platform
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block.set_sample_rate(0, self.samp_rate)
        elif SDR_TYPE == "HACKRF":
            self.sdr_source_block.set_sample_rate(0, self.samp_rate)
            self.sdr_source_block.set_bandwidth(0, self.samp_rate)
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block.set_sample_rate(0, self.samp_rate)
            self.sdr_source_block.set_bandwidth(0, self.samp_rate)
        elif SDR_TYPE == "PLUTOSDR":
            self.sdr_source_block.set_sample_rate(0, self.samp_rate)
            self.sdr_source_block.set_bandwidth(0, self.samp_rate)
        elif SDR_TYPE == "USRPSDR":
            self.sdr_source_block.set_samp_rate(self.samp_rate)
            self.sdr_source_block.set_bandwidth(self.samp_rate, 0)

    def get_offset(self):
        """Get current calibration offset in dB."""
        return self.offset

    def set_offset(self, offset):
        """
        Set calibration offset and update power correction block.
        
        Args:
            offset: Calibration offset in dB
        """
        self.offset = offset
        self.blocks_add_const_vxx_0.set_k((-self.gain-self.offset))

    def get_gain(self):
        """Get current RF gain setting in dB."""
        return self.gain

    def set_gain(self, gain):
        """
        Set RF gain for the SDR and update calibration correction.
        
        Args:
            gain: RF gain in dB (range depends on SDR platform)
        """
        self.gain = gain
        self.blocks_add_const_vxx_0.set_k((-self.gain-self.offset))
        
        # Apply gain setting to appropriate SDR platform
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block.set_gain(0, 'TUNER', self.gain)
        elif SDR_TYPE == "HACKRF":
            self.sdr_source_block.set_gain(0, 'VGA', min(max(self.gain, 0.0), 62.0))
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block.set_gain(0, min(max(self.gain, -12.0), 61.0))
        elif SDR_TYPE == "PLUTOSDR":
            self.sdr_source_block.set_gain(0, min(max(self.gain, 0.0), 73.0))
        elif SDR_TYPE == "USRPSDR":
            self.sdr_source_block.set_gain(self.gain, 0)

    def get_freq_correction(self):
        """Get current frequency correction in ppm."""
        return self.freq_correction

    def set_freq_correction(self, freq_correction):
        """
        Set frequency correction for crystal oscillator accuracy.
        
        Args:
            freq_correction: Frequency correction in parts per million (ppm)
        """
        self.freq_correction = freq_correction
        # Apply frequency correction for supported platforms
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block.set_frequency_correction(0, self.freq_correction)
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block.set_frequency_correction(0, self.freq_correction)
        # Note: HackRF, PlutoSDR, USRP do not support frequency correction via SoapySDR

    def get_channel_freq(self):
        """Get current RF center frequency in Hz."""
        return self.channel_freq

    def set_channel_freq(self, channel_freq):
        """
        Set RF center frequency and update GUI display range.
        
        Args:
            channel_freq: Center frequency in Hz
        """
        self.channel_freq = channel_freq
        self.qtgui_freq_sink_x_0.set_frequency_range(self.channel_freq, self.samp_rate)
        
        # Update center frequency for each SDR platform
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block.set_frequency(0, self.channel_freq)
        elif SDR_TYPE == "HACKRF":
            self.sdr_source_block.set_frequency(0, self.channel_freq)
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block.set_frequency(0, self.channel_freq)
        elif SDR_TYPE == "PLUTOSDR":
            self.sdr_source_block.set_frequency(0, self.channel_freq)
        elif SDR_TYPE == "USRPSDR":
            self.sdr_source_block.set_center_freq(self.channel_freq, 0)

    def get_AVG(self):
        """Get current moving average filter length."""
        return self.AVG

    def set_AVG(self, AVG):
        """
        Set moving average filter length and update decimation factor.
        
        Args:
            AVG: Number of samples for moving average calculation
        """
        self.AVG = AVG
        # Update moving average filter parameters
        self.blocks_moving_average_xx_0.set_length_and_scale(self.AVG, (1.0/self.AVG))
        # Update decimation factor to match averaging length
        self.blocks_keep_one_in_n_0.set_n(self.AVG)

def main(top_block_cls=integration_time_measurement):
    """
    Main measurement automation function for SDR Allan Deviation characterization.
    
    Performs systematic gain sweep measurements, collects power data in memory,
    calculates Allan Deviation offline, and generates comprehensive analysis plots.
    Creates organized directory structure for results with timestamp-based naming.
    
    Args:
        top_block_cls: GNU Radio top block class for power measurement flowgraph
    """
    # Create SDR-specific root directory for organized data storage
    sdr_type_root_dir = SDR_TYPE
    os.makedirs(sdr_type_root_dir, exist_ok=True)
    print(f"Using/Created SDR-specific root directory: {sdr_type_root_dir}")

    # Create timestamped subdirectory for current measurement session
    timestamp_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_run_dir = os.path.join(sdr_type_root_dir, timestamp_dir_name)
    os.makedirs(output_run_dir, exist_ok=True)
    print(f"Created timestamped subdirectory: {output_run_dir}")

    # Initialize Qt Application for GUI management across multiple flowgraphs
    qapp = Qt.QApplication(sys.argv)

    # Generate gain sweep values with proper floating-point handling
    gain_values = np.arange(MIN_SDR_GAIN, MAX_SDR_GAIN + SDR_GAIN_STEP / 2, SDR_GAIN_STEP)
    if len(gain_values) == 0:
        print("Warning: Gain range resulted in no steps. Using a single gain value (0 dB).")
        gain_values = [0]

    # Storage for Allan Deviation result file paths for final plotting
    allan_deviation_files = []

    # Calculate effective power measurement data rate after decimation
    if AVG_LENGTH > 0:
        power_data_rate = SAMP_RATE / AVG_LENGTH
    else:
        power_data_rate = SAMP_RATE
    print(f"Effective data rate of power measurement (collected in memory): {power_data_rate:.2f} Hz")


    # Main measurement loop: iterate through each gain setting
    for i, current_gain in enumerate(gain_values):
        print(f"\n--- Starting measurement for Gain: {current_gain} dB ({i+1}/{len(gain_values)}) ---")

        # Create new flowgraph instance for current gain setting
        tb = top_block_cls(current_sdr_gain=current_gain, base_output_dir=output_run_dir)

        # Define output file for Allan Deviation results
        allan_output_filename = os.path.join(output_run_dir, 
                                               f"allan_deviation_results_Gain_{current_gain}dB_{SDR_TYPE}.txt")
        allan_deviation_files.append(allan_output_filename)
        
        # Threading synchronization for data collection completion
        data_collection_complete_event = threading.Event()

        # Data collection worker thread for current gain measurement
        def data_collection_thread_func_current_gain(top_block_instance, collection_complete_event):
            """Worker thread for timed data collection with GUI responsiveness."""
            try:
                top_block_instance.flowgraph_started.wait()
                print(f">>> GNU Radio Flowgraph for Gain {current_gain}dB started. Beginning data collection for {REQUIRED_DATA_COLLECTION_TIME_S:.2f} s.")
                
                collection_start_time = time.time()
                while (time.time() - collection_start_time) < REQUIRED_DATA_COLLECTION_TIME_S:
                    # Check for premature flowgraph termination
                    if not top_block_instance.running:
                        print(f"Flowgraph terminated during data collection (Gain: {current_gain}dB).")
                        break
                    qapp.processEvents()  # Maintain GUI responsiveness
                    time.sleep(0.1)  # Prevent CPU-intensive busy waiting

                print(f">>> Data collection for Gain {current_gain}dB completed.")

            finally:
                collection_complete_event.set()

        # Start data collection in background thread
        data_collection_thread = threading.Thread(target=data_collection_thread_func_current_gain, 
                                                   args=(tb, data_collection_complete_event))
        data_collection_thread.daemon = True
        data_collection_thread.start()

        # Start GNU Radio flowgraph and display GUI
        tb.start()
        tb.show()
        tb.flowgraph_started.set()  # Signal data collection thread to begin

        # GUI event processing during data collection
        while data_collection_thread.is_alive():
            qapp.processEvents()
            time.sleep(0.01)

        # Clean flowgraph shutdown and resource cleanup
        print(f">>> Stopping GNU Radio Flowgraph for Gain {current_gain}dB.")
        tb.stop()
        tb.wait()
        
        # Close GUI window to prevent resource accumulation
        tb.close() 
        
        # Extract collected power data from vector sink
        all_power_data = np.array(tb.vector_sink.data(), dtype=np.float32)
        print(f"    Collected {len(all_power_data)} data points for Gain {current_gain}dB.")
        
        # Release flowgraph memory resources
        del tb
        print(f"--- Completed data collection and cleanup for Gain: {current_gain} dB. ---")

        # Offline Allan Deviation calculation from collected data
        print(f"--- Calculating Allan Deviation for Gain: {current_gain} dB (offline from memory) ---")
        
        if len(all_power_data) < 3:
            print(f"\n    WARNING: Not enough data points ({len(all_power_data)}) for Allan Deviation calculation. Skipping for this gain.")
            continue

        # Calculate overlapping Allan Deviation using allantools
        # Returns actual tau values and corresponding ADEV measurements
        (taus_out_actual, adev_actual, adeverr_actual, n_actual) = \
            allantools.oadev(all_power_data, rate=power_data_rate, data_type="freq")

        # Filter results to specified integration time range
        relevant_indices = (taus_out_actual >= ALLAN_T_MIN) & (taus_out_actual <= ALLAN_T_MAX)
        filtered_taus = taus_out_actual[relevant_indices]
        filtered_adev = adev_actual[relevant_indices]

        if len(filtered_taus) == 0:
            print(f"    No Allan Deviation points found within [{ALLAN_T_MIN}, {ALLAN_T_MAX}] for Gain {current_gain}dB.")
            continue

        # Save Allan Deviation results to text file with metadata header
        with open(allan_output_filename, 'w') as f:
            f.write("# Allan Deviation Measurement\n")
            f.write(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# SDR Type: {SDR_TYPE}, Gain: {current_gain} dB, Freq: {CHANNEL_FREQ/1e6} MHz, Samp Rate: {SAMP_RATE/1e6} Msps\n")
            f.write(f"# Collected Data Points: {len(all_power_data)}\n")
            f.write("# Integration Time (s), Allan Deviation\n")

            # Write tau and ADEV pairs, handling NaN values appropriately
            for tau_val, adev_val in zip(filtered_taus, filtered_adev):
                if not np.isnan(adev_val):
                    f.write(f"{tau_val}, {adev_val}\n")
                else:
                    f.write(f"{tau_val}, NaN\n")
            print(f">>> Allan Deviation calculations for Gain {current_gain}dB completed and saved.")
        

    print("\nAll gain measurements completed. Exiting application.")
    
    # Generate comprehensive Allan Deviation plot for all gain settings
    print("\n>>> Generating plot of Allan Deviation results...")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('viridis', len(gain_values))  # Distinct colors for each gain

    for idx, file_path in enumerate(allan_deviation_files):
        if os.path.exists(file_path):
            try:
                # Load Allan Deviation data, skipping comment header lines
                data = np.genfromtxt(file_path, comments='#', delimiter=',')
                
                # Handle data validation and NaN filtering
                if data.ndim > 1:
                    data = data[~np.isnan(data[:, 1])]  # Remove invalid ADEV values
                else:
                    if np.any(np.isnan(data)):
                         data = np.array([])
                
                if data.size == 0:
                    print(f"Warning: No valid Allan Deviation data points found in {file_path}. Skipping plotting for this file.")
                    continue

                # Extract gain value from filename using robust regex pattern matching
                gain_val_str = "N/A"
                match = re.search(r'Gain_(-?\d+\.?\d*)dB', os.path.basename(file_path))
                if match:
                    gain_val_str = match.group(1)
                else:
                    # Fallback pattern search
                    match = re.search(r'(-?\d+\.?\d*)dB', os.path.basename(file_path))
                    if match:
                        gain_val_str = match.group(1)
                
                # Plot Allan Deviation data with logarithmic axes
                plt.loglog(data[:, 0], data[:, 1], 'o', color=colors(idx), 
                           label=f'Gain: {gain_val_str} dB') 
            except Exception as e:
                print(f"Warning: Could not read or plot data from {file_path}: {e}")
        else:
            print(f"Warning: Data file not found for plotting: {file_path}")

    # Configure plot appearance and labels
    plt.title('Allan Deviation vs. Integration Time for Different Gains')
    plt.xlabel('Integration Time $\\tau$ (s)')
    plt.ylabel('Allan Deviation $\\sigma_y(\\tau)$')
    plt.grid(True, which="both", ls="-")
    plt.legend(title="Measurement Series")
    plt.tight_layout()

    # Save plot as PDF in timestamped output directory
    plot_filename = os.path.join(output_run_dir, f"Allan_Deviation_Plot_{timestamp_dir_name}.pdf")
    plt.savefig(plot_filename)
    print(f">>> Plot saved as: {plot_filename} <<<")

    plt.show()
    print(">>> Plot generated. <<<")

    qapp.quit()

if __name__ == '__main__':
    main()