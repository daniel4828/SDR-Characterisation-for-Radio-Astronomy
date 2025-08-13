"""
Automated calibration measurement script for signal generator characterization.

This script performs power sweep measurements using a Keysight signal generator and 
spectrum analyzer via GPIB communication. The measurement data is automatically saved 
to a timestamped file for calibration analysis.

For detailed setup instructions and usage guidelines, refer to the accompanying 
signal_generator_calibration_guide.md documentation in this folder.
"""

import pyvisa
import time
import numpy as np
from datetime import datetime


# ===========================================================
# Parameters
# ===========================================================

# GPIB device addresses - configure these for your specific setup
# Use Keysight Connection Expert to identify correct addresses
EXA_GPIB_ADDRESS = 'GPIB0::18::INSTR'  # Spectrum analyzer address
SIG_GEN_GPIB_ADDRESS = 'GPIB0::13::INSTR'  # Signal generator address

# Signal generator measurement parameters
FREQUENCY_HZ = 1_000_000_000  # Test frequency: 1 GHz
FIXED_FREQUENCY_DISPLAY = "1000 MH"  # SCPI command format for signal generator
STEP_AMPLITUDE_DBM = 1.0  # Power step size for sweep

# Spectrum analyzer configuration
CENTER_FREQUENCY_HZ = 1_000_000_000  # Center frequency: 1 GHz
SPAN_HZ = 10_000  # Frequency span: 10 kHz
AVERAGES_COUNT = 100  # Number of averages taken for one datapoint
INSTRUMENT_TIMEOUT_MS = 10000  # VISA communication timeout

# LOW RANGE measurement settings (configure for your test requirements)
MIN_AMPLITUDE_DBM = -120.0
MAX_AMPLITUDE_DBM = -95.0
RESOLUTION_BW_HZ = 3      
REFERENCE_LEVEL_DBM = -40.0
ATTENUATOR_DB = 10.0

# HIGH RANGE measurement settings (uncomment to use)
# MIN_AMPLITUDE_DBM = -95.0
# MAX_AMPLITUDE_DBM = 0.0
# RESOLUTION_BW_HZ = 220
# REFERENCE_LEVEL_DBM = 0.0
# ATTENUATOR_DB = 10.0



def generate_filename():
    """
    Generate timestamped filename for measurement data output.
    
    Returns filename with format: YYYYMMDD_range_<MIN>to<MAX>_RBW<Hz>_Ref<dBm>_Att<dB>.txt
    """
    current_date = datetime.now().strftime("%Y%m%d")
    min_amp = int(MIN_AMPLITUDE_DBM) if MIN_AMPLITUDE_DBM == int(MIN_AMPLITUDE_DBM) else MIN_AMPLITUDE_DBM
    max_amp = int(MAX_AMPLITUDE_DBM) if MAX_AMPLITUDE_DBM == int(MAX_AMPLITUDE_DBM) else MAX_AMPLITUDE_DBM
    rbw = int(RESOLUTION_BW_HZ) if RESOLUTION_BW_HZ == int(RESOLUTION_BW_HZ) else RESOLUTION_BW_HZ
    ref = int(REFERENCE_LEVEL_DBM) if REFERENCE_LEVEL_DBM == int(REFERENCE_LEVEL_DBM) else REFERENCE_LEVEL_DBM
    att = int(ATTENUATOR_DB) if ATTENUATOR_DB == int(ATTENUATOR_DB) else ATTENUATOR_DB
    
    filename = f"{current_date}_range_{min_amp}to{max_amp}_RBW{rbw}Hz_Ref{ref}dBm_Att{att}dB.txt"
    return filename

def run_calibration_sweep():
    """
    Execute complete calibration measurement sweep.
    
    Controls signal generator power output and measures corresponding values on spectrum
    analyzer. Automatically saves measurement data to timestamped output file.
    """
    print(">>> Starting calibration measurement...")

    # Generate power levels for measurement sweep
    amplitudes_to_sweep = np.arange(MIN_AMPLITUDE_DBM, MAX_AMPLITUDE_DBM + STEP_AMPLITUDE_DBM, STEP_AMPLITUDE_DBM)

    # Initialize VISA resource manager for instrument communication
    try:
        rm = pyvisa.ResourceManager()
        print("VISA Resource Manager successfully initialized.")
    except Exception as e:
        print(f"Error initializing VISA Resource Manager: {e}")
        return

    # Establish GPIB connections to both instruments
    try:
        print(f"Connecting to spectrum analyzer at address: {EXA_GPIB_ADDRESS}")
        exa = rm.open_resource(EXA_GPIB_ADDRESS)
        exa.timeout = INSTRUMENT_TIMEOUT_MS
        exa.read_termination = '\n'
        exa.write_termination = '\n'
        print("Spectrum analyzer connected.")

        print(f"Connecting to signal generator at address: {SIG_GEN_GPIB_ADDRESS}")
        sig_gen = rm.open_resource(SIG_GEN_GPIB_ADDRESS)
        sig_gen.timeout = INSTRUMENT_TIMEOUT_MS
        sig_gen.read_termination = '\n'
        sig_gen.write_termination = '\n'
        print("Signal generator connected.")
    except pyvisa.errors.VisaIOError as e:
        print(f"Error: Could not establish connection to one of the devices. Check addresses and connections.")
        print(f"VISA error details: {e}")
        return

    try:
        # Configure spectrum analyzer for measurement
        print("\n--- Configuring spectrum analyzer ---")

        # Reset device to known state and clear error queue
        exa.write('*RST')
        exa.write('*CLS')
        time.sleep(1)
        print("Device reset.")

        # Set frequency parameters
        exa.write(f'SENS:FREQ:CENT {CENTER_FREQUENCY_HZ} HZ')
        print(f"Center frequency set to {CENTER_FREQUENCY_HZ / 1e9} GHz.")
        exa.write(f'SENS:FREQ:SPAN {SPAN_HZ} HZ')
        print(f"Span set to {SPAN_HZ / 1e3} kHz.")

        # Configure resolution bandwidth for measurement precision
        exa.write(f'SENS:BAND:RES {RESOLUTION_BW_HZ} HZ')
        print(f"Resolution bandwidth set to {RESOLUTION_BW_HZ / 1e3} kHz.")

        # Enable marker for peak detection
        exa.write('CALC:MARK1:MODE POS')
        print("Marker 1 set to 'Peak Search'.")

        # Enable trace averaging
        exa.write('SENS:AVER ON')
        print(f"Trace set to 'Average' with {AVERAGES_COUNT} averages.")

        # Set display and input parameters
        exa.write(f'DISP:WIND:TRAC:Y:RLEV {REFERENCE_LEVEL_DBM} DBM')
        print(f"Reference level set to {REFERENCE_LEVEL_DBM} dBm.")
        exa.write(f'SENS:POW:ATT {ATTENUATOR_DB} DB')
        print(f"Attenuator set to {ATTENUATOR_DB} dB.")

        # Disable LNA
        exa.write('SENS:POW:GAIN:STAT OFF')
        print("LNA (Preamplifier) turned off.")

        print("--- Spectrum analyzer configuration completed. ---\n")
        time.sleep(2)  # Allow settings to stabilize

        # Execute measurement sweep across all power levels
        print("--- Starting measurement sweep ---")

        sweep_time_str = exa.query('SENS:SWE:TIME?')
        sweep_time_sec = float(sweep_time_str)
        print(f"Queried sweep duration: {sweep_time_sec:.4f} seconds.")
        
        # Create output file with descriptive filename
        output_filename = generate_filename()
        print(f"Generated output filename: {output_filename}")
        
        with open(output_filename, 'w') as f:
            f.write("Set_Power_dBm;Measured_Power_dBm\n")
            print(f"Output file '{output_filename}' has been opened.")

            for amp in amplitudes_to_sweep:
                # Set signal generator output power
                print("--------------------------------------------------")
                print(f"Setting power on signal generator: {amp:.2f} dBm")
                sig_gen.write(f"{FIXED_FREQUENCY_DISPLAY} {amp} DB")

                # Reset averaging to start fresh measurement
                exa.write('SENS:AVER:CLE')
                print("Average counter on spectrum analyzer reset.")

                # Wait for averaging process to complete
                wait_time = sweep_time_sec * AVERAGES_COUNT
                print(f"Waiting {wait_time:.2f} seconds for averaging to complete...")
                time.sleep(wait_time)
                
                # Ensure measurement completion before reading
                exa.write('*OPC')
                time.sleep(0.5)

                # Read power measurement from marker
                measured_power_str = exa.query('CALC:MARK1:Y?')
                measured_power_dbm = float(measured_power_str)
                print(f"Measured power on spectrum analyzer: {measured_power_dbm:.4f} dBm")

                # Save data point to file
                f.write(f"{amp:.4f};{measured_power_dbm:.4f}\n")
                print(f"Data point written: {amp:.4f} dBm -> {measured_power_dbm:.4f} dBm")

        print("--------------------------------------------------")
        print("\n>>> Measurement sweep successfully completed.")

    except Exception as e:
        print(f"\nAn error occurred during operation: {e}")
        print("Attempting to set devices to a safe state.")

    finally:
        # Safely disconnect instruments and restore default states
        print("\n--- Cleanup operations ---")
        try:
            # Disable signal generator RF output for safety
            if 'sig_gen' in locals() and sig_gen.session:
                sig_gen.write('OUTP:STAT OFF')
                print("Signal generator RF output deactivated.")
                sig_gen.close()
                print("Connection to signal generator closed.")
        except Exception as e:
            print(f"Error deactivating signal generator: {e}")

        try:
            # Close spectrum analyzer connection
            if 'exa' in locals() and exa.session:
                exa.close()
                print("Connection to spectrum analyzer closed.")
        except Exception as e:
            print(f"Error closing connection to spectrum analyzer: {e}")

        print(">>> Calibration program terminated.")


if __name__ == "__main__":
    run_calibration_sweep()
