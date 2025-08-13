# Y-Factor Noise Figure Measurement Guide

This script measures the noise figure of the SDRs using the Y-factor method by switching a calibrated noise source on and off while recording received power levels.

## **Objective**
For precise radiometric measurements, it is essential to know the noise figure (NF) of the receiver chain. The NF indicates how much additional noise is introduced by the receiving system itself and is therefore critical for the correct interpretation of the measured signal power.

## **Experimental Setup**
An Agilent 346B noise source is used, powered by a Keysight EX36243A DC supply at 28 V. The noise source generates a statistical noise signal with a known excess noise ratio (ENR) of 14.54 dB. Its output is connected to the SDR via a low-noise amplifier (LNA).

## **Required Libraries and Dependencies**
- Python 3.x
- numpy
- matplotlib
- PyQt5
- GNU Radio (>=3.10.x)
- gnuradio-soapy
- gnuradio-uhd
- sip

## **Procedure and Script Execution**
1. Connect the Agilent 346B noise source to the SDR input via the low-noise amplifier.
2. Power the noise source with the Keysight EX36243A DC supply set to 28 V.
3. Ensure the SDR is connected and recognized by the operating system.
4. Open the script `Y_Factor.py` in a text editor.
5. Set the correct **Parameters** in the script:
   - `ENR_DB` to match the noise source ENR.
   - `SDR_TYPE` to the connected SDR (e.g., "PLUTOSDR").
   - `SAMP_RATE`, `CHANNEL_FREQ`, and gain sweep settings. Make sure the gains can even detect the Noise Sources' output power.
6. Run the script from the terminal. Attention! The Power Source is not controlled by the program. It will tell you when to switch it on and off.
7. The noise source is first switched on (hot state). After a short settling time, the received power is measured over 4 seconds. The program will tell you when to switch on and off the Power Source.
8. Switch the noise source off (cold state) and measure the power again for 4 seconds.
9. The Y-factor is automatically calculated from the ratio of the hot and cold state measurements.

## **Script Output and Data Format**
```text
# Y-Factor Noise Figure Measurement Results
# SDR_TYPE: HACKRF, ENR: 14.54 dB, Freq: 1000.0 MHz, SampRate: 2.0 MSps
SDR_Gain_dB,P_hot_dBm,P_cold_dBm,Y_Factor_dB,Noise_Figure_dB
40.0,-80.86080056,-83.93643746,3.07563691,14.41029450
```

## **Troubleshooting and Known Issues**
- Make sure the noise source outputs power that can even be detected by the SDR and its respective gain. Otherwise you can only work with higher gains or must use a LNA. 

## Analyze Data
Use the script `analyze_noise_figure.py` to visualize the Noise Figure
- Converts measurement values between dB and absolute units.  
- Corrects the measured noise figure using the Friis formula, accounting for the LNA.  
- Checks for invalid measurement points (negative values after correction).  
- Calculates and visualizes measurement uncertainties.  
- Outputs a plot showing corrected noise figure values with uncertainty bands.  