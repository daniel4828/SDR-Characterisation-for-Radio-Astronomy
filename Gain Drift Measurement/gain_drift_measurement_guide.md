# Gain Drift Measurement with SDRs

This script measures gain stability (gain drift) of the SDRs over time.

## Objective
In radio astronomy, measurements are often conducted over extended periods of time. When using an SDR for such measurements, it is essential to know whether the gain remains stable or drifts over time. Any drift in gain can unknowingly distort the measurement results and lead to incorrect interpretations.

## Experimental Setup

To investigate the gain drift, the *Rohde & Schwarz SMPC* signal generator is used but you can use the one you prefer.

### Using a splitter 
This measurement can be performed using a splitter to allow parallel testing of all SDRs. 
Since multiple SDRs are measured in parallel, the input power must be chosen carefully to ensure that all devices operate within their respective dynamic ranges. Due to the use of a splitter, which reduces the signal level at each output, it may be necessary to set different gain values for each SDR to ensure that the input power is within the detectable range of each device.

## Required Libraries and Dependencies
The following Python libraries are required to run the script:

- `numpy`
- `matplotlib`
- `pyserial` (if any serial communication is needed)
- `os`, `datetime`, `csv` (standard libraries)
- SDR-specific libraries (e.g., `pyrtlsdr`, `SoapySDR`, or `pyadi` depending on used SDRs)

> Make sure to install any necessary driver or backend for your SDR hardware.

## Procedure and Script Execution

To perform the gain drift measurement:

1. Connect the signal generator to the splitter using a coaxial cable.
2. Terminate all unused ports of the splitter with 50 Ω loads.
3. Connect each RX output of the splitter to a separate SDR.
4. Connect all SDRs to the laptop via USB.
5. Adjust the input power at the signal generator such that each SDR operates within its dynamic range.
6. Set the gain values manually in the script for each SDR, based on the selected input power.
7. Open the script `gain_drift.py` and configure:
   - Measurement duration
   - Sampling rate
   - Frequency
   - SDR gain settings
8. Run the script for each SDR in a new shell

> Ensure that the gain values and sampling parameters are correctly set for your SDRs before running the measurement.

## Script Output and Data Format

- The script logs time series power values per SDR to a CSV file.
- "timestamp,measured_power_dBm,standard_deviation"
- The files are directly plotted with the `plot_util.py` file

## Troubleshooting and Known Issues

- Make sure the Signal Generator is connected to a clock.

