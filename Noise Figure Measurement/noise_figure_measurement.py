"""
Y-Factor Noise Figure Measurement Script

This script implements the Y-factor method to measure the noise figure of SDRs
including RTL-SDR, PlutoSDR, HackRF, LimeSDR, and USRP devices. The measurement involves 
switching a calibrated noise source on/off while recording received power levels across different 
SDR gain settings. The program will tell you when to switch on and off the power source.

For detailed setup instructions, measurement procedures, and equipment requirements, refer to the 
accompanying noise_figure_measurement_guide.md in the same folder.
"""


import sys
import signal
import time
import datetime
import os
import threading
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sip
from gnuradio import soapy
from gnuradio import uhd # Import UHD for USRP devices
from gnuradio import filter # Added for DC Blocker

# ##################################################
# --- MEASUREMENT CONFIGURATION ---
# ##################################################

# --- Y-Factor Measurement Parameters ---
ENR_DB = 14.54                # Excess Noise Ratio of the noise source in dB (from calibration data)
T_WAIT_S = 2.0               # Wait time in seconds before each measurement for stabilization
T_INT_S = 4.0               # Integration time (measurement duration) in seconds

# --- SDR Gain Sweep Configuration ---
MIN_SDR_GAIN_SWEEP = 50       # Minimum SDR gain in dB for the sweep
MAX_SDR_GAIN_SWEEP = 70      # Maximum SDR gain in dB for the sweep
SDR_GAIN_STEP_SWEEP = 10      # Step size for the SDR gain sweep in dB

# --- SDR/GNU Radio Configuration ---
# Select SDR type: "RTLSDR", "PLUTOSDR", "LIMESDR", "USRPSDR", "HACKRF"
SDR_TYPE = "PLUTOSDR"

SAMP_RATE = 2.0e6            # Sample rate in samples/second
CHANNEL_FREQ = 1e9           # Center frequency in Hz (e.g., 1 GHz)
PPM_CORRECTION = 0           # Frequency correction in parts-per-million (for RTL-SDR/LimeSDR)
CALIBRATION_OFFSET = 0       # Calibration offset in dB (subtracted from result)
AVG_LENGTH = 2000            # Length of the moving average filter for power smoothing

# Power calculation parameter: 10*log10 for power conversion (dB scale)
LOG_MULTIPLY = 10

# ##################################################
# --- END OF CONFIGURATION ---
# ##################################################


class NoiseFigureExperiment(gr.top_block, Qt.QWidget):
    """
    Main GNU Radio flowgraph class for Y-factor noise figure measurements.
    
    This class implements a complete measurement setup that interfaces with various SDR devices,
    processes received signals through DC blocking and power calculation chains, and provides
    real-time visualization through Qt GUI elements.
    """

    def __init__(self):
        gr.top_block.__init__(self, "Y-Factor Noise Figure Measurement", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Y-Factor Noise Figure Measurement")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except Exception as e:
            print(f"Qt GUI: Could not set icon: {str(e)}", file=sys.stderr)
        
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

        self.settings = Qt.QSettings("gnuradio", "noise_figure_experiment")
        try:
            geometry = self.settings.value("geometry")
            if geometry: self.restoreGeometry(geometry)
        except Exception as e:
            print(f"Qt GUI: Could not restore geometry: {str(e)}", file=sys.stderr)

        self.sdr_source_block = None # Placeholder for the SDR source block

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = SAMP_RATE
        self.offset = offset = CALIBRATION_OFFSET
        self.gain = gain = MIN_SDR_GAIN_SWEEP # Starting gain value
        self.freq_correction = freq_correction = PPM_CORRECTION
        self.channel_freq = channel_freq = CHANNEL_FREQ
        self.AVG = AVG = AVG_LENGTH
        self.log_multiply = LOG_MULTIPLY

        ##################################################
        # Blocks
        ##################################################
        self._initialize_sdr_source()

        # DC Blocker to remove DC offset from SDR input [Change 1]
        self.blocks_dc_blocker_xx_0 = filter.dc_blocker_cc(AVG_LENGTH, True)

        # GUI sinks for real-time visualization
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float, 0, qtgui.NUM_GRAPH_HORIZ, 1)
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title("Measured Power (dBm)")
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_number_sink_0_win)

        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            2048, window.WIN_BLACKMAN_hARRIS, channel_freq, samp_rate, "Frequency Spectrum", 1)
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)

        # Signal processing blocks
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(self.log_multiply, 1, 0)
        # Important: Gain compensation occurs here - output represents power *before* SDR amplifier
        self.blocks_add_const_vxx_0 = blocks.add_const_ff((-self.gain - self.offset))
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(self.AVG, (1.0 / self.AVG), 4000, 1)
        
        # Probe block to read values from the flowgraph into Python
        self.probe = blocks.probe_signal_f()
        
        ##################################################
        # Connections
        ##################################################
        # Connect SDR source to DC Blocker [Change 2]
        self.connect((self.sdr_source_block, 0), (self.blocks_dc_blocker_xx_0, 0))

        # Connect DC Blocker to subsequent blocks [Change 3]
        self.connect((self.blocks_dc_blocker_xx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_dc_blocker_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))
        
        # Original connections for power calculation chain (now fed by DC blocker output)
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_nlog10_ff_0, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.qtgui_number_sink_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.probe, 0)) # Probe also gets DC-blocked data


    def _initialize_sdr_source(self):
        """Initializes the appropriate SDR source block based on SDR_TYPE configuration."""
        samp_rate = self.samp_rate
        channel_freq = self.channel_freq
        gain = self.gain
        freq_correction = self.freq_correction

        print(f"Initializing {SDR_TYPE} source...")
        if SDR_TYPE == "RTLSDR":
            self.sdr_source_block = soapy.source('driver=rtlsdr', "fc32", 1, '', 'bufflen=16384', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_frequency_correction(0, freq_correction)
            self.sdr_source_block.set_gain_mode(0, False) # Manual gain control
            self.sdr_source_block.set_gain(0, gain)
        elif SDR_TYPE == "HACKRF":
            self.sdr_source_block = soapy.source('driver=hackrf', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, 'AMP', True)
            self.sdr_source_block.set_gain(0, 'LNA', 16) # Typical LNA value
            self.sdr_source_block.set_gain(0, 'VGA', gain)
        elif SDR_TYPE == "LIMESDR":
            self.sdr_source_block = soapy.source('driver=lime', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, gain)
        elif SDR_TYPE == "PLUTOSDR":
            self.sdr_source_block = soapy.source('driver=plutosdr', "fc32", 1, '', '', [''], [''])
            self.sdr_source_block.set_sample_rate(0, samp_rate)
            self.sdr_source_block.set_bandwidth(0, samp_rate)
            self.sdr_source_block.set_gain_mode(0, False) # Manual gain control [Change 4]
            self.sdr_source_block.set_frequency(0, channel_freq)
            self.sdr_source_block.set_gain(0, gain)
        elif SDR_TYPE == "USRPSDR":
            self.sdr_source_block = uhd.usrp_source(
                ",".join(("", "")),
                uhd.stream_args(cpu_format="fc32", channels=list(range(1))),
            )
            self.sdr_source_block.set_samp_rate(samp_rate)
            self.sdr_source_block.set_center_freq(channel_freq, 0)
            self.sdr_source_block.set_gain(gain, 0)
            self.sdr_source_block.set_antenna("RX2", 0) # Adjust if using different antenna port
            self.sdr_source_block.set_bandwidth(samp_rate, 0)
        else:
            raise ValueError(f"Unknown SDR_TYPE: '{SDR_TYPE}'.")

    def _collect_samples(self, duration_s):
        """Collects measurement values from the probe block for a given duration."""
        collected = []
        start_time = time.time()
        # Clear probe block buffer to discard old values
        _ = self.probe.level()
        time.sleep(0.1) # Short pause to allow new values to arrive
        _ = self.probe.level()

        print(f"Measuring for {duration_s} seconds...")
        while time.time() - start_time < duration_s:
            time.sleep(0.05)  # Reduce CPU load
            new_data = self.probe.level()
            if isinstance(new_data, list) or isinstance(new_data, tuple):
                 collected.extend(new_data)
            elif new_data is not None:
                 collected.append(new_data)
        return collected

    def run_single_y_factor_measurement(self):
        """Performs a single 'Hot' and 'Cold' measurement and returns the results."""
        
        # --- HOT Measurement ---
        print("\n" + "="*50)
        print(">>> PLEASE TURN ON THE NOISE SOURCE NOW <<<")
        print("="*50)
        print(f"Waiting {T_WAIT_S} seconds before 'HOT' measurement begins...")
        time.sleep(T_WAIT_S)
        
        hot_samples = self._collect_samples(T_INT_S)
        if not hot_samples:
            print("WARNING: No 'HOT' samples collected. Aborting measurement.")
            return None
        
        p_hot_dbm = np.mean(hot_samples)
        print(f"-> 'HOT' measurement completed. Average power: {p_hot_dbm:.3f} dBm")

        # --- COLD Measurement ---
        print("\n" + "="*50)
        print(">>> PLEASE TURN OFF THE NOISE SOURCE NOW <<<")
        print("="*50)
        print(f"Waiting {T_WAIT_S} seconds before 'COLD' measurement begins...")
        time.sleep(T_WAIT_S)

        cold_samples = self._collect_samples(T_INT_S)
        if not cold_samples:
            print("WARNING: No 'COLD' samples collected. Aborting measurement.")
            return None
            
        p_cold_dbm = np.mean(cold_samples)
        print(f"-> 'COLD' measurement completed. Average power: {p_cold_dbm:.3f} dBm")
        
        # --- Calculation ---
        y_factor_db = p_hot_dbm - p_cold_dbm
        
        if y_factor_db <= 0:
            print("WARNING: Y-factor is <= 0 dB. Noise figure cannot be calculated.")
            noise_figure_db = float('nan') # 'Not a Number' as placeholder
        else:
            y_factor_linear = 10**(y_factor_db / 10.0)
            noise_figure_db = ENR_DB - 10 * np.log10(y_factor_linear - 1)
        
        print("\n--- RESULT ---")
        print(f"Y-Factor:        {y_factor_db:.3f} dB")
        print(f"Noise Figure (NF): {noise_figure_db:.3f} dB")
        
        return p_hot_dbm, p_cold_dbm, y_factor_db, noise_figure_db

    # Standard Qt/GRC methods
    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # Getter/Setter for dynamic adjustments
    def get_gain(self): return self.gain
    def set_gain(self, gain):
        """
        Sets the SDR gain and updates the gain compensation block.
        
        This method configures the hardware gain for the specific SDR type and updates
        the software gain compensation to maintain calibrated power measurements.
        """
        self.gain = gain
        # Update gain compensation block
        self.blocks_add_const_vxx_0.set_k((-self.gain - self.offset))
        
        # Set gain for the respective SDR type
        print(f"Setting SDR gain to: {self.gain:.1f} dB")
        try:
            if SDR_TYPE == "RTLSDR":
                self.sdr_source_block.set_gain(0, self.gain)
            elif SDR_TYPE == "HACKRF":
                self.sdr_source_block.set_gain(0, 'VGA', min(max(self.gain, 0.0), 62.0))
            elif SDR_TYPE == "LIMESDR":
                self.sdr_source_block.set_gain(0, min(max(self.gain, -12.0), 61.0))
            elif SDR_TYPE == "PLUTOSDR":
                self.sdr_source_block.set_gain(0, min(max(self.gain, 0.0), 73.0))
            elif SDR_TYPE == "USRPSDR":
                self.sdr_source_block.set_gain(self.gain, 0)
        except Exception as e:
            print(f"Error setting gain for {SDR_TYPE}: {e}")

def main(top_block_cls=NoiseFigureExperiment):
    """
    Main function that orchestrates the complete Y-factor noise figure measurement.
    
    This function creates the directory structure, initializes the measurement system,
    performs gain sweep measurements, saves results to files, and generates plots.
    """
    # 1. Create directory structure
    sdr_type_root_dir = f"{SDR_TYPE}_Measurements"
    os.makedirs(sdr_type_root_dir, exist_ok=True)
    print(f"Storage location for results: '{sdr_type_root_dir}'")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gain_config = f"Gain_{MIN_SDR_GAIN_SWEEP}_{MAX_SDR_GAIN_SWEEP}_{SDR_GAIN_STEP_SWEEP}"
    run_dir_name = f"{timestamp}_{gain_config}"
    output_run_dir = os.path.join(sdr_type_root_dir, run_dir_name)
    os.makedirs(output_run_dir, exist_ok=True)
    print(f"Subdirectory for this run created: '{output_run_dir}'")

    # 2. Prepare results file
    results_filename = f"{timestamp}_noise_figure_results.txt"
    results_filepath = os.path.join(output_run_dir, results_filename)
    
    try:
        with open(results_filepath, 'w') as f:
            f.write("# Y-Factor Noise Figure Measurement Results\n")
            f.write(f"# SDR_TYPE: {SDR_TYPE}, ENR: {ENR_DB} dB, Freq: {CHANNEL_FREQ/1e6} MHz, SampRate: {SAMP_RATE/1e6} MSps\n")
            f.write("SDR_Gain_dB,P_hot_dBm,P_cold_dBm,Y_Factor_dB,Noise_Figure_dB\n")
    except IOError as e:
        print(f"FATAL: Could not create results file: {e}")
        return

    # 3. Initialize Qt application and flowgraph
    qapp = Qt.QApplication(sys.argv)
    tb = top_block_cls()
    tb.start()
    tb.show()
    
    print("\nFlowgraph started. Waiting 2 seconds for stabilization...")
    time.sleep(2)

    # 4. Perform gain sweep measurements
    gain_values = np.arange(MIN_SDR_GAIN_SWEEP, MAX_SDR_GAIN_SWEEP + 1, SDR_GAIN_STEP_SWEEP)
    all_results = []

    for i, current_gain in enumerate(gain_values):
        print(f"\n{'='*20} MEASUREMENT {i+1}/{len(gain_values)} {'='*20}")
        tb.set_gain(current_gain)
        
        # Process GUI events to prevent window freezing
        qapp.processEvents()
        time.sleep(0.5) # Short pause to allow gain setting to take effect
        
        # Perform the actual measurement
        result = tb.run_single_y_factor_measurement()
        
        if result:
            p_hot, p_cold, y_factor, nf = result
            # Write result to file
            with open(results_filepath, 'a') as f:
                f.write(f"{current_gain:.1f},{p_hot:.8f},{p_cold:.8f},{y_factor:.8f},{nf:.8f}\n")
            all_results.append((current_gain, nf))

    print("\n\nAll measurements completed.")

    # 5. Stop flowgraph and end application
    tb.stop()
    tb.wait()
    
    # 6. Plot results if data is available
    if all_results:
        print("Creating plot of noise figure vs. gain...")
        gains = [r[0] for r in all_results]
        nfs = [r[1] for r in all_results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gains, nfs, 'o-', label=f'{SDR_TYPE} Noise Figure')
        plt.title(f'Noise Figure vs. SDR Gain ({SDR_TYPE})')
        plt.xlabel('SDR Gain (dB)')
        plt.ylabel('Measured Noise Figure (dB)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(output_run_dir, f"{timestamp}_noise_figure_vs_gain.pdf")
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.show()

    qapp.quit()


if __name__ == '__main__':
    main()