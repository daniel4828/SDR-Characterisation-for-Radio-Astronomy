# Integration Time Measurement Script – Variable Gain

This script is used to measure and analyze the optimal integration time (τ) for minimizing statistical fluctuations in microwave noise measurements at various gain settings.

## Objective
Characterizing the integration time response is crucial to ensure that measurements are performed with the best possible accuracy by selecting a τ value that minimizes noise fluctuations while remaining within the dynamic range of the SDR.

## Experimental Setup
 An **Agilent 346B** serves as the noise source, powered by a **Keysight EX36243A** DC source at 28 V. The noise source generates a statistical noise signal with a known power of 14.54 dB ENR.  
The output of the noise source is connected to the SDR via a Low Noise Amplifier (LNA), which in turn is connected to the computer.  
The noise source is operated continuously at 28 V. Its average output signal of –96.3 dBm is amplified to –81 dBm by the LNA, ensuring the level lies within the dynamic range of the gain setting under investigation. Data is collected over a period of 120 s for each gain setting.  
Initially, this experiment was attempted with only a 50 Ω termination connected to the SDR input, but the resulting noise level was below the dynamic range of the SDR. The corrected approach uses an active noise source with known characteristics to ensure reliable and scientifically valid measurements.

## Required Libraries and Dependencies
- Python 3.x
- `numpy`
- `matplotlib`
- `scipy`
- SDR driver packages (depending on your SDR hardware, e.g., `pyrtlsdr`, `pyadi-iio`, or similar)
- GNU Radio (for live SDR data acquisition)
- System drivers for your SDR device (e.g., RTL-SDR, PlutoSDR)

## Procedure and Script Execution
1. **Prepare Hardware**  
   - Connect the Agilent 346B noise source to the LNA, and connect the LNA output to the SDR input.  
   - Power the noise source with 28 V from the Keysight EX36243A DC supply.  
   - Ensure the SDR is connected to the computer via USB and recognized by the operating system.
2. **Check Signal Levels**  
   - Verify that the amplified noise level is within the SDR’s usable dynamic range for the chosen gain setting.
3. **Configure Script Parameters**  
   - Open the Python script and set key parameters:
     - `sampling_rate` (e.g., 2 MS/s)  
     - `integration_times` array (τ values to test)  
     - SDR gain settings to investigate  
     - File paths for saving results
4. **Run the Script**  
   - Start the script to acquire data from the SDR.  
   - The script will integrate multiple data points over τ and compute variance for each integration time and gain setting.
5. **Data Storage**  
   - The script saves measurement results (e.g., mean power, variance) to a structured text or CSV file for later analysis.
6. **Repeat for All Gains**  
   - Repeat the acquisition for each desired SDR gain setting.

## Script Output and Data Format
```text
Allan Deviation Measurement  
Start time: 2025-07-31 14:56:18  
SDR Type: PLUTOSDR, Gain: 50.0 dB, Freq: 1000.0 MHz, Samp Rate: 2.0 Msps  
Collected Data Points: 420089  

Integration Time (s), Allan Deviation  
0.128, 0.011686  
0.256, 0.008714  
...
```

## Troubleshooting and Known Issues
- **Signal Out of Dynamic Range**: If the noise source output plus LNA gain exceeds the SDR’s maximum input, reduce gain to avoid saturation.

## Data Visualization and Post-Processing
- Use the provided plotting script. Nevertheless the main script itself also plots the values
- Look for the τ value where variance reaches a minimum — this is the optimal integration time for that gain setting.  
