"""
Dynamic Range Measurement Script for SDR Receiver Characterization

This script automates dynamic range measurements by controlling a signal generator
to sweep through power levels while using GNU Radio flowgraphs with various SDRs (RTL-SDR, HackRF, PlutoSDR, LimeSDR, USRP) to measure received power.
The script performs gain sweeps, calculates statistics, and generates comparison plots.
Keep in mind, that in the output data the gain is already subtracted from the measured power.

For detailed setup instructions and measurement procedures, refer to the
accompanying dynamic_range_guide.md in the same folder.
"""

import sys
import signal
import time
import datetime
import os
import threading
import numpy as np
import matplotlib.pyplot as plt

# --- GPIB/PyVISA import with fallback for optional dependency ---
try:
    import pyvisa
except ImportError:
    print("Warning: pyvisa not found. GPIB functionality will be disabled.")
    pass  # Continue without pyvisa if not installed
# ----------------------------

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sip
from gnuradio import eng_notation
from gnuradio import soapy
from gnuradio import uhd  # UHD library for USRP devices

# ##################################################
# --- PROGRAM CONFIGURATION ---
# ##################################################

# --- Test and GPIB Configuration ---
USE_GPIB = True  
GPIB_ADDRESS = 'GPIB0::13::INSTR'  # GPIB device address for signal generator

# --- Signal Generator Configuration ---
FIXED_FREQUENCY_DISPLAY = "1000 MH"  # Display format for frequency (e.g., "1000 MH" for 1000 MHz)
MIN_AMPLITUDE_DBM = -120.0
MAX_AMPLITUDE_DBM = 0.0
STEP_AMPLITUDE_DBM = 1.0

# --- Measurement Configuration ---
BUFFER_TIME_S = 0.1          # Time to wait after power change (buffer values are discarded)
MEASUREMENT_TIME_S = 1.0     # Measurement duration per power step


# --- SDR/GNU Radio Configuration ---
SAMP_RATE = 2.0e6
CHANNEL_FREQ = 1e9
SDR_GAIN = 40  # Default gain, will be overridden by sweep
PPM_CORRECTION = 22
CALIBRATION_OFFSET = 0  # Your calibration offset in dB
AVG_LENGTH = 1000        # Length of the Moving Average Filter

# Parameter for power calculation: 1*log10 (voltage) or 10*log10 (power)
# Use 10 for power measurements in dBm (as in GNU Radio's default power blocks)
LOG_MULTIPLY = 10

# --- SDR Selection ---
# Choose the SDR to use by setting one of the following options:
# "RTLSDR", "PLUTOSDR", "LIMESDR", "USRPSDR", "HACKRF"
SDR_TYPE = "USRPSDR"

# --- SDR Gain Sweep Configuration ---
MIN_SDR_GAIN_SWEEP = 0    # Minimum SDR Gain in dB for the sweep
MAX_SDR_GAIN_SWEEP = 70   # Maximum SDR Gain in dB for the sweep
SDR_GAIN_STEP_SWEEP = 10  # Step size for SDR Gain in dB for the sweep

# ##################################################
# --- END OF CONFIGURATION ---
# ##################################################


class PowerSweepExperiment(gr.top_block, Qt.QWidget):
    """
    GNU Radio flowgraph class for automated power sweep measurements.
    
    Combines signal generator control with SDR-based power measurements,
    supporting multiple SDR types and gain configurations with real-time visualization.
    """

    def __init__(self, current_sdr_gain, output_file_path_template):
        gr.top_block.__init__(self, f"Combined Power Sweep and Measurement (SDR Gain: {current_sdr_gain} dB)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle(f"Combined Power Sweep and Measurement (SDR Gain: {current_sdr_gain} dB)")
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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "power_sweep_experiment")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        self.flowgraph_started = threading.Event()
        self.control_thread = None
        self.my_instrument = None
        self.output_file = None
        self.sdr_source_block = None # Placeholder for the selected SDR source block

        # Store the current gain and output file path template for this instance
        self.current_sdr_gain = current_sdr_gain
        self.output_file_path_template = output_file_path_template

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = SAMP_RATE
        self.offset = offset = CALIBRATION_OFFSET
        self.gain = self.current_sdr_gain  # Use the current gain passed to the constructor
        self.freq_correction = freq_correction = PPM_CORRECTION
        self.channel_freq = channel_freq = CHANNEL_FREQ
        self.AVG = AVG = AVG_LENGTH
        self.log_multiply = LOG_MULTIPLY  # Global variable assigned to instance variable

        ##################################################
        # Blocks
        ##################################################

        # --- SDR Source Block Initialization (Dynamically chosen) ---
        # Initialize the correct SDR source block based on the SDR_TYPE global variable
        self._initialize_sdr_source()

        # GUI Sinks for visualization
        self.qtgui_number_sink_0 = qtgui.number_sink(gr.sizeof_float, 0, qtgui.NUM_GRAPH_HORIZ, 1)
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title("Measured Power (dBm)")
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)

        # Additional GUI Sinks for Time and Frequency domain visualization
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024,  # Size of the FFT for time sink
            samp_rate,  # Sample rate for the time sink
            "Time Domain",  # Title of the time sink
            1,  # Number of inputs
            None  # Parent widget
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
        # Set up lines for the time sink (only one complex input expected)
        labels_time = ['Signal']
        widths_time = [1]
        colors_time = ['blue']
        alphas_time = [1.0]
        styles_time = [1]
        markers_time = [-1]
        for i in range(1):
            if len(labels_time[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{Data %d}" % (i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{Data %d}" % (i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels_time[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths_time[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors_time[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles_time[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers_time[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas_time[i])
        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)


        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            2048,  # Size of the FFT for freq sink
            window.WIN_BLACKMAN_hARRIS,  # Window type for FFT
            channel_freq,  # Center frequency
            samp_rate,  # Bandwidth
            "Frequency Domain",  # Title of the frequency sink
            1,  # Number of inputs
            None  # Parent widget
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
        # Set up lines for the frequency sink (only one complex input expected)
        labels_freq = ['Signal']
        widths_freq = [1]
        colors_freq = ["blue"]
        alphas_freq = [1.0]
        styles_freq = [1]  # Line style definition for frequency sink
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


        # Processing blocks (same for all SDR types)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        # Using self.log_multiply to set the first parameter of nlog10_ff (10*log10 for power in dB)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(self.log_multiply, 1, 0)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff((-self.gain - self.offset))
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(self.AVG, (1.0 / self.AVG), 4000, 1)
        
        # Probe Signal Block: Allows Python to access values from the GNU Radio flowgraph
        self.probe = blocks.probe_signal_f()
        
        ##################################################
        # Connections
        ##################################################
        # Connect the dynamically chosen SDR source to the processing chain and GUI sinks
        self.connect((self.sdr_source_block, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_moving_average_xx_0, 0))
        
        # Connect the final measured power to the number sink and the probe block
        self.connect((self.blocks_moving_average_xx_0, 0), (self.qtgui_number_sink_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.probe, 0))
        
        # Connect the SDR source to the time and frequency GUI sinks for visualization
        self.connect((self.sdr_source_block, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.sdr_source_block, 0), (self.qtgui_time_sink_x_0, 0))


    def _initialize_sdr_source(self):
        """
        Initializes the appropriate SDR source block based on the global SDR_TYPE variable.
        
        Called once during __init__ to configure device-specific parameters and gain settings.
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
            self.sdr_source_block.set_frequency_correction(0, freq_correction)
            self.sdr_source_block.set_gain_mode(0, False)
            self.sdr_source_block.set_gain(0, gain)
        elif SDR_TYPE == "HACKRF":
            print("Initializing HackRF source...")
            self.sdr_source_block = soapy.source('driver=hackrf', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, 'AMP', True)  # Enable amplifier
            self.sdr_source_block.set_gain(0, 'LNA', min(max(16, 0.0), 40.0))  # Set LNA gain (fixed)
            self.sdr_source_block.set_gain(0, 'VGA', min(max(gain, 0.0), 62.0))  # Set VGA gain (variable)
        elif SDR_TYPE == "LIMESDR":
            print("Initializing LimeSDR source...")
            self.sdr_source_block = soapy.source('driver=lime', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_frequency_correction(0, freq_correction)
            self.sdr_source_block.set_gain(0, min(max(gain, -12.0), 61.0))
        elif SDR_TYPE == "PLUTOSDR":
            print("Initializing PlutoSDR source...")
            self.sdr_source_block = soapy.source('driver=plutosdr', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_gain_mode(0, False)  # Manual gain control
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, min(max(gain, 0.0), 73.0))
        elif SDR_TYPE == "USRPSDR":
            print("Initializing USRP SDR source...")
            # UHD source initialization
            self.sdr_source_block = uhd.usrp_source(
                ",".join(("", '')),  # Device arguments (empty for auto-detection)
                uhd.stream_args(
                    cpu_format="fc32",  # Complex float 32-bit
                    args='',
                    channels=list(range(0,1)),  # Single channel
                ),
            )
            self.sdr_source_block.set_samp_rate(samp_rate)
            self.sdr_source_block.set_time_unknown_pps(uhd.time_spec(0))
            self.sdr_source_block.set_center_freq(channel_freq, 0)
            self.sdr_source_block.set_antenna("TX/RX", 0)  # TX/RX antenna selection
            self.sdr_source_block.set_bandwidth(samp_rate, 0)
            self.sdr_source_block.set_rx_agc(False, 0)  # Disable AGC for manual gain
            self.sdr_source_block.set_gain(gain, 0)  # Set RX gain
        else:
            # Raise error for unsupported SDR types
            raise ValueError(f"Unknown SDR_TYPE: '{SDR_TYPE}'. Please choose from: 'RTLSDR', 'PLUTOSDR', 'LIMESDR', 'USRPSDR', 'HACKRF'.")

    def _setup_signal_generator(self):
        """Initializes the connection to the signal generator via GPIB/PyVISA."""
        if USE_GPIB:
            # --- GPIB Communication Setup ---
            try:
                print("Connecting to signal generator via PyVISA...")
                rm = pyvisa.ResourceManager()
                self.my_instrument = rm.open_resource(GPIB_ADDRESS)
                self.my_instrument.timeout = 5000  # Set timeout for communication
                self.my_instrument.read_termination = '\n'  # Read termination character
                self.my_instrument.write_termination = '\n'  # Write termination character
                self.my_instrument.encoding = 'utf-8'
                return True
            except pyvisa.errors.VisaIOError as e:
                print(f"ERROR: Could not connect to signal generator: {e}")
                self.my_instrument = None
                return False
        else:
            print("TEST MODE: Signal generator initialization is simulated.")
            return True

    def _set_generator_amplitude(self, amp_dbm):
        """Sets the amplitude on the signal generator using SCPI-like commands."""
        # SCPI command format depends on your specific signal generator model
        command = f"{FIXED_FREQUENCY_DISPLAY} {amp_dbm:.2f} DB"
        if USE_GPIB and self.my_instrument:
            # --- GPIB Command Transmission ---
            self.my_instrument.write(command)
        else:
            print(f"TEST COMMAND TO GENERATOR: '{command}'")

    def _run_measurement_sweep(self, measurement_complete_event):
        """
        Main control loop that runs in a separate thread for a single SDR gain configuration.
        
        Performs power sweep measurement, data collection, and result logging.
        """
        self.flowgraph_started.wait()  # Wait until the GNU Radio flowgraph is running
        print("Flowgraph started. Beginning measurement procedure.")

        if not self._setup_signal_generator():
            print("Aborting measurement as signal generator could not be initialized.")
            measurement_complete_event.set()  # Signal completion even on error
            return

        # Create amplitude list for the sweep
        amplitudes_to_sweep = np.arange(MIN_AMPLITUDE_DBM, MAX_AMPLITUDE_DBM + STEP_AMPLITUDE_DBM, STEP_AMPLITUDE_DBM)

        # Construct the output filename for this specific gain measurement
        actual_filename = os.path.join(self.output_file_path_template, 
                                       f"power_sweep_results_Gain_{self.current_sdr_gain}dB.txt")

        try:
            self.output_file = open(actual_filename, 'w')
            # Write header to the output file
            self.output_file.write("Set_Power_dBm,Mean_Measured_Power,Std_Dev_Measured_Power\n")
            print(f"Writing measurement results to: {actual_filename}")
        except Exception as e:
            print(f"ERROR: Could not open file {actual_filename}: {e}")
            measurement_complete_event.set()  # Signal completion even on error
            return

        # Main loop for the power sweep
        for amp in amplitudes_to_sweep:
            if not self.running:  # Check if the flowgraph is still running
                print("Flowgraph was terminated, aborting sweep.")
                break

            print("-" * 40)
            print(f"Setting generator power: {amp:.2f} dBm")
            self._set_generator_amplitude(amp)

            print(f"Waiting for buffer time ({BUFFER_TIME_S}s)...")
            time.sleep(BUFFER_TIME_S)
            
            # Discard old values from the probe block before starting new measurement
            _ = self.probe.level()  # Read and discard current value
            
            print(f"Starting measurement for {MEASUREMENT_TIME_S}s...")
            collected_samples = []
            measurement_start_time = time.time()
            while time.time() - measurement_start_time < MEASUREMENT_TIME_S:
                if not self.running: break  # Check again if flowgraph is still running
                time.sleep(0.05)  # Small delay to avoid busy-waiting and allow samples to accumulate
                
                new_data = self.probe.level()  # Get latest value from probe
                if new_data is not None:  # Ensure data is not None (0.0 is a valid value)
                    try:
                        # Attempt to extend if new_data is iterable (e.g., a list from probe)
                        collected_samples.extend(new_data)
                    except TypeError:
                        # If new_data is a single float, append it
                        collected_samples.append(new_data)

            print(f"Measurement for {amp:.2f} dBm completed. {len(collected_samples)} samples collected.")

            if collected_samples:
                mean_val = np.mean(collected_samples)
                std_dev = np.std(collected_samples)
                
                print(f"  -> Average: {mean_val:.4f}")
                print(f"  -> Standard Deviation: {std_dev:.4f}")

                self.output_file.write(f"{amp:.2f},{mean_val:.8f},{std_dev:.8f}\n")
                self.output_file.flush()  # Ensure data is written to disk immediately
            else:
                print("  -> WARNING: No samples collected during the measurement interval.")
                self.output_file.write(f"{amp:.2f},NaN,NaN\n")  # Write NaN if no data
                self.output_file.flush()

        # After the sweep is complete
        print("-" * 40)
        print("Sweep completed.")
        
        if USE_GPIB and self.my_instrument:
            print("Disabling RF output on signal generator.")
            # --- GPIB Command for RF Disable --- (adjust command to your device)
            self.my_instrument.write("RF 0")  # Command to turn off RF output
        else:
            print("TEST MODE: RF output simulated disabled.")
            
        if self.output_file:
            self.output_file.close()
            print(f"Result file '{actual_filename}' has been closed.")

        measurement_complete_event.set()  # Signal that this measurement iteration is complete

    # --- Standard Qt/GNU Radio Methods ---

    def start(self):
        """Starts the GNU Radio flowgraph and signals its readiness."""
        # Set running flag, start the GNU Radio flowgraph, and signal its start
        self.running = True
        super(PowerSweepExperiment, self).start()
        self.flowgraph_started.set()  # Signal that the flowgraph has started

    def stop(self):
        """Stops the GNU Radio flowgraph and clears the running flag."""
        # Clear running flag and stop the GNU Radio flowgraph
        self.running = False
        super(PowerSweepExperiment, self).stop()

    def closeEvent(self, event):
        """Handles window close event by saving settings and stopping the flowgraph."""
        # Save GUI geometry, stop the flowgraph, wait for its termination, and accept close event
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # --- Getter/Setter Methods for Dynamic SDR Parameter Adjustment ---

    def get_samp_rate(self): 
        return self.samp_rate
        
    def set_samp_rate(self, samp_rate):
        """Updates sample rate for both GUI elements and SDR-specific configurations."""
        self.samp_rate = samp_rate
        # Update GUI sinks with new sample rate
        self.qtgui_freq_sink_x_0.set_frequency_range(self.channel_freq, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        
        # Update SDR-specific sample rate and bandwidth settings
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
        return self.offset
        
    def set_offset(self, offset):
        """Updates calibration offset and adjusts the constant addition block."""
        self.offset = offset
        # Offset is applied in a separate GNU Radio block, common to all SDRs
        self.blocks_add_const_vxx_0.set_k((-self.gain-self.offset))

    def get_gain(self): 
        return self.gain
        
    def set_gain(self, gain):
        """Updates SDR gain settings for all supported device types."""
        self.gain = gain
        # Update offset compensation block
        self.blocks_add_const_vxx_0.set_k((-self.gain-self.offset))
        
        # Apply gain specific to each SDR type
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block.set_gain(0, self.gain)
        elif SDR_TYPE == "HACKRF":
            # For HackRF, 'VGA' gain is typically the variable gain parameter
            self.sdr_source_block.set_gain(0, 'VGA', min(max(self.gain, 0.0), 62.0))
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block.set_gain(0, min(max(self.gain, -12.0), 61.0))
        elif SDR_TYPE == "PLUTOSDR":
            self.sdr_source_block.set_gain(0, min(max(self.gain, 0.0), 73.0))
        elif SDR_TYPE == "USRPSDR":
            self.sdr_source_block.set_gain(self.gain, 0)

    def get_freq_correction(self): 
        return self.freq_correction
        
    def set_freq_correction(self, freq_correction):
        """Updates frequency correction for SDRs that support this feature."""
        self.freq_correction = freq_correction
        # Frequency correction is only applied to SDRs that explicitly support it
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block.set_frequency_correction(0, self.freq_correction)
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block.set_frequency_correction(0, self.freq_correction)
        # Note: HackRF, PlutoSDR, USRP do not have explicit frequency correction methods

    def get_channel_freq(self): 
        return self.channel_freq
        
    def set_channel_freq(self, channel_freq):
        """Updates center frequency for all supported SDR types."""
        self.channel_freq = channel_freq
        # Update GUI frequency sink range
        self.qtgui_freq_sink_x_0.set_frequency_range(self.channel_freq, self.samp_rate)
        
        # Apply channel frequency specific to each SDR type
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
        return self.AVG
        
    def set_AVG(self, AVG):
        """Updates the moving average filter length and scaling."""
        self.AVG = AVG
        # Update the moving average filter length and scaling factor
        self.blocks_moving_average_xx_0.set_length_and_scale(self.AVG, (1.0/self.AVG))


def main(top_block_cls=PowerSweepExperiment):
    """
    Main function that orchestrates the complete measurement procedure.
    
    Creates directory structure, initializes Qt application, performs gain sweeps,
    and generates comparison plots of all measurement results.
    """
    # 1. Create the SDR type specific root directory
    sdr_type_root_dir = f"{SDR_TYPE}_Measurements"
    os.makedirs(sdr_type_root_dir, exist_ok=True)
    print(f"Using/Created SDR-specific root directory: {sdr_type_root_dir}")

    # 2. Create a timestamped subdirectory inside the SDR-specific folder for this run
    timestamp_dir_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Dynamically construct the folder name based on configuration values
    power_config = f"Power_{MIN_AMPLITUDE_DBM}_{MAX_AMPLITUDE_DBM}_{STEP_AMPLITUDE_DBM}"
    gain_config = f"Gain_{MIN_SDR_GAIN_SWEEP}_{MAX_SDR_GAIN_SWEEP}_{SDR_GAIN_STEP_SWEEP}"
    output_run_dir_name = f"{timestamp_dir_name}_{power_config}_{gain_config}"
    
    output_run_dir = os.path.join(sdr_type_root_dir, output_run_dir_name)
    os.makedirs(output_run_dir, exist_ok=True)
    print(f"Created timestamped subdirectory for this run: {output_run_dir}")

    # 3. Initialize Qt Application once
    qapp = Qt.QApplication(sys.argv)

    # 4. Generate SDR gain values to iterate through
    gain_values = np.arange(MIN_SDR_GAIN_SWEEP, MAX_SDR_GAIN_SWEEP + SDR_GAIN_STEP_SWEEP / 2, SDR_GAIN_STEP_SWEEP)
    if len(gain_values) == 0:
        print("Warning: SDR Gain range resulted in no steps. Using a single gain value (0 dB).")
        gain_values = [0]

    # List to store paths of generated power sweep results text files
    all_power_sweep_files = []

    # Loop through each SDR gain value to perform a separate measurement
    for i, current_sdr_gain in enumerate(gain_values):
        print(f"\n--- Starting Power Sweep for SDR Gain: {current_sdr_gain} dB ({i+1}/{len(gain_values)}) ---")

        # Create a new instance of the top block for each SDR gain
        tb = top_block_cls(current_sdr_gain=current_sdr_gain, output_file_path_template=output_run_dir)
        
        # Event to signal when the current measurement thread is complete
        measurement_complete_event = threading.Event()

        # Define and start the measurement thread for the current iteration
        measurement_thread = threading.Thread(target=tb._run_measurement_sweep, args=(measurement_complete_event,))
        measurement_thread.daemon = True  # Make it a daemon thread so it exits with the main program
        measurement_thread.start()

        # Start the GNU Radio Flowgraph and show its GUI
        tb.start()
        tb.show()
        
        # Process Qt events while the measurement thread is running
        # This keeps the GUI responsive without blocking the main loop
        while measurement_thread.is_alive():
            qapp.processEvents()  # Process GUI events
            time.sleep(0.1)  # Small sleep to avoid busy-waiting

        # After the measurement thread signals completion, stop and clean up the flowgraph
        print(f">>> Stopping GNU Radio Flowgraph for SDR Gain {current_sdr_gain}dB.")
        tb.stop()
        tb.wait()
        
        # Explicitly hide and close the window for the previous flowgraph
        tb.close() 
        # Explicitly delete the top block instance to ensure resources are released
        del tb
        print(f"--- Completed measurement for SDR Gain: {current_sdr_gain} dB. ---")

        # Add the path of the generated file to the list for later plotting
        sweep_filename = os.path.join(
            output_run_dir,
            f"power_{MIN_AMPLITUDE_DBM}_{MAX_AMPLITUDE_DBM}_{STEP_AMPLITUDE_DBM}_Gain_{current_sdr_gain}dB.txt"
        )
        all_power_sweep_files.append(sweep_filename)
    
    print("\nAll SDR gain measurements completed. Exiting application.")
    
    # --- Generate Plots of Power Sweep Results ---
    print("\n>>> Generating plots of Power Sweep results...")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('viridis', len(gain_values))  # Get a colormap for distinct colors
    
    # First plot: Corrected gain
    for idx, file_path in enumerate(all_power_sweep_files):
        if os.path.exists(file_path):
            try:
                # Read data, skipping header lines (starting with '#')
                data = np.genfromtxt(file_path, comments='#', delimiter=',')
                if data.ndim == 1:  # Handle case where only one data point exists
                    data = data.reshape(1, -1)

                # Extract gain from filename for label
                filename = os.path.basename(file_path)
                gain_val_str = "N/A"  # Default in case parsing fails
                if "Gain_" in filename and "dB" in filename:
                    try:
                        gain_val_str = filename.split("Gain_")[1].split("dB")[0]
                    except (IndexError, ValueError):
                        pass  # Continue if split or conversion fails

                # Plot data: Set Power (column 0) vs. Mean Measured Power (column 1)
                plt.plot(data[:, 0], data[:, 1], 'o-', color=colors(idx),
                         label=f'SDR Gain: {gain_val_str} dB')
            except Exception as e:
                print(f"Warning: Could not read or plot data from {file_path}: {e}")
        else:
            print(f"Warning: Data file not found for plotting: {file_path}")
    
    plt.title('Measured Power vs. Input Power (Corrected Gain)')
    plt.xlabel('Set Input Power (dBm)')
    plt.ylabel('Measured Power (dBm)')
    plt.grid(True)
    plt.legend(title="Measurement Series")
    plt.tight_layout()
    
    # Save the first plot
    corrected_gain_plot_path = os.path.join(output_run_dir, f"{timestamp_dir_name}-corrected-gain.pdf")
    plt.savefig(corrected_gain_plot_path)
    print(f"Corrected gain plot saved to: {corrected_gain_plot_path}")
    
    # Show the first plot
    plt.show()
    
    # Second plot: Raw gain (add gain to measured power)
    plt.figure(figsize=(10, 6))
    for idx, file_path in enumerate(all_power_sweep_files):
        if os.path.exists(file_path):
            try:
                # Read data, skipping header lines (starting with '#')
                data = np.genfromtxt(file_path, comments='#', delimiter=',')
                if data.ndim == 1:  # Handle case where only one data point exists
                    data = data.reshape(1, -1)

                # Extract gain from filename for label
                filename = os.path.basename(file_path)
                gain_val = 0.0  # Default gain value
                if "Gain_" in filename and "dB" in filename:
                    try:
                        gain_val = float(filename.split("Gain_")[1].split("dB")[0])
                    except (IndexError, ValueError):
                        pass  # Continue if split or conversion fails

                # Add gain to measured power (column 1)
                adjusted_power = data[:, 1] + gain_val

                # Plot data: Set Power (column 0) vs. Adjusted Measured Power
                plt.plot(data[:, 0], adjusted_power, 'o-', color=colors(idx),
                         label=f'SDR Gain: {gain_val:.1f} dB')
            except Exception as e:
                print(f"Warning: Could not read or plot data from {file_path}: {e}")
        else:
            print(f"Warning: Data file not found for plotting: {file_path}")
    
    plt.title('Measured Power vs. Input Power (Raw Gain)')
    plt.xlabel('Set Input Power (dBm)')
    plt.ylabel('Measured Power + Gain (dBm)')
    plt.grid(True)
    plt.legend(title="Measurement Series")
    plt.tight_layout()
    
    # Save the second plot
    raw_gain_plot_path = os.path.join(output_run_dir, f"{timestamp_dir_name}-raw-gain.pdf")
    plt.savefig(raw_gain_plot_path)
    print(f"Raw gain plot saved to: {raw_gain_plot_path}")
    
    # Show the second plot
    plt.show()
    print(">>> Plots generated and saved. <<<")

    # Finally, quit the Qt application after all measurements are done
    qapp.quit()


if __name__ == '__main__':
    main()
