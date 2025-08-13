# Calibration of the Rohde & Schwarz SMPC Signal Generator

This script performs an automated calibration of the Rohde & Schwarz SMPC signal generator using a Keysight N9010B spectrum analyzer via GPIB control.

## Objective
The signal generator is central to SDR characterization. It must be calibrated in advance to detect and correct any offsets or deviations, ensuring accurate measurements.

## Experimental Setup
The experiment uses the Rohde & Schwarz SMPC signal generator and the Keysight EXA Signal Analyzer N9010B.  Per Default the signal generator outputs at 1 GHz, while the spectrum analyzer is tuned to the same frequency with a 10 kHz span. The analyzer averages 100 samples per power step, with settings optimized separately for two ranges: Low (-120 dBm to -95 dBm) and High (-95 dBm to 0 dBm), each with distinct resolution bandwidths and reference levels to ensure accurate measurements across the spectrum. This is done to keep the Noise Floor of the spectrum analyzer low enough but also ensure the measurement doesn't take too long. 

## Required Libraries and Dependencies
- `pyvisa`
- `numpy`
- `time`
- `datetime`
- `os`
- `sys`
- `csv`

Ensure the National Instruments VISA runtime is installed to enable PyVISA communication with GPIB instruments.

## Procedure and Script Execution
Follow these steps to perform the signal generator calibration:

1. **Connect all instruments**:
   - Connect SMPC and N9010B via coaxial RF cable.
   - Synchronize clocks between both instruments.
   - Connect both to the PC via GPIB-USB adapter.

2. **Set instrument parameters**:
   - In the script, select measurement range: `"low"` or `"high"`.
   - Check and adjust:
     - Frequency (default: 1 GHz).
     - Power step (1 dB).
     - Number of averages (default: 100).
     - Spectrum analyzer settings (RBW, reference level, attenuation) according to chosen range.

3. **Run the script** `calibration_signal_generator.py`:
   - The script will sweep from -120 dBm to -95 dBm (low range) or -95 dBm to 0 dBm (high range).
   - After each power step, the analyzer collects and averages 100 measurements.
   - Each measurement result is printed and saved automatically.


## Script Output and Data Format
- **File format**: CSV
- **Filename**: `calibration_<range>_<timestamp>.csv`
- **Columns**:
  - Set Power (dBm)
  - Measured Power (dBm)
  - Timestamp (UTC)
  - Frequency (Hz)
  - RBW (Hz)
  - Reference Level (dBm)

## Troubleshooting and Known Issues
- **No GPIB connection**: Ensure VISA is installed and the instruments appear in `visa.ResourceManager().list_resources()`.
- **Invalid instrument address**: Confirm instrument identifiers match what's configured.
- **Sweep delay too short**: If measured values are inconsistent, increase wait time after setting power (`sleep_time` variable).
- **RBW mismatch**: Make sure analyzer RBW is set correctly for low (-120 to -95 dBm) vs. high (-95 to 0 dBm) range.
- **Noise floor too high in low range**: Verify analyzer is in low-noise mode and uses 3 Hz RBW.

## Data Visualization and Post-Processing
Use the script `analyze_calibration_signal_generator.py` to analyze the data and find slope and offset.
