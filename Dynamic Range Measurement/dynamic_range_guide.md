# Dynamic Range Measurement of SDRs with Signal Generator

This Python script automates the measurement of the dynamic range of various Software Defined Radios (SDRs) using a calibrated signal generator.

## Objective
Characterizing the dynamic range of an SDR is essential to understand the amplitude range over which it can accurately measure signals, avoid ADC saturation, and ensure reliable radiometric measurements in later radio astronomy experiments.

## Experimental Setup
Each SDR is connected via coaxial cable to the previously calibrated signal generator **ROHDE & SCHWARZ SMPC**.  
A SMA right-angle adapter is used to avoid damaging the screw connection of the SDRs.  
The signal generator is synchronized with its own clock beforehand, otherwise, no power will be output.  
The SDR is connected to the laptop via USB-C, and the laptop is connected to the signal generator via a GPIB-to-USB-C adapter.  

## Required Libraries and Dependencies
- `numpy`
- `time`
- `math`
- `pathlib`
- `pyvisa` (for GPIB communication with the signal generator)
- `gnuradio` (GNU Radio runtime for executing flowgraphs)
- `pmt` (GNU Radio messaging)
- Installed GNU Radio Companion with required source/sink blocks
- Proper device drivers for the SDR in use (e.g., PlutoSDR, RTL-SDR)

## Procedure and Script Execution
1. Prepare Setup
2. Set the center frequency in the script to **1 GHz**.
3. Configure the power sweep parameters as you wish for example:  
   - Start power: −120 dBm  
   - Stop power: 0 dBm  
   - Step size: 1 dB  
   - Step duration: 1 s
4. Select the SDR you want to use
5. Type in the correct GPIB address of your signal generator
6. Run the Python script; it will control both the signal generator and the SDR, performing measurements automatically.
7. Wait until the sweep is completed; results will be stored in the output file.

## Script Output and Data Format
The script outputs a CSV file with three columns:
Set_Power_dBm, Mean_Measured_Power, Std_Dev_Measured_Power
-120.00, -162.57838186, 6.22517964
-119.00, -161.98318651, 5.74136887

Keep in mind, that in the output data the gain is already subtracted from the measured power.


## Troubleshooting and Known Issues
- **No output from signal generator**: Ensure the internal clock is synchronized before starting.
- **GPIB communication errors**: Verify the PyVISA installation and that the correct GPIB address is configured.


## Data Visualization and Post-Processing
The `plot_and_find_offsets_for_all_sdrs.py` script processes dynamic range measurement files for multiple SDR gain settings.  
It automatically:
- Reads result files and extracts gain values from filenames.
- Fits a linear regression of set power (dBm) vs. measured power (dBFS) for each gain.
- Calculates slope, intercept, and the **offset** needed to map measured to actual input power.
- Saves plots and prints a summary of offsets for later calibration which look like this:
```text
SDR: PLUTO

Gain: 0.0
  Input Power Range: -57.898790277876834 to -2.212922892476621
  Measured Power Range: -65.41964385 to -8.34240384
  R²: 0.999075
  Slope: 0.999482, Offset: -4.850065
  Offset - Gain: -4.850065

Gain: 10.0
...
```