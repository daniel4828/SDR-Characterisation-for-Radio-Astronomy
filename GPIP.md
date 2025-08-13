# Automation
Especially to measure the Dynamic Range for multiple SDRs, each at multiple gain settings, the time required is very high since for every measurement point (120 points per gain setting and SDR) the signal generator’s output power needs to be changed and the measured power needs to be noted. Further, an automation allows for repeatability, for instance with different SDRs or firmwares.

## GPIB
A widely used interface with lab equipment is serial communication via GPIB. To connect a computer, a GPIB to USB Adapter by National Instruments (GPIB-USB-HS Adapter) is used.


### GPIB Adapter Communication
To interface with the specific adapter, the backend NI-VISA is required to be installed [NI-VISA Download](https://www.ni.com/de/support/documentation/compatibility/21/ni-hardware-and-operating-system-compatibility.html). The interface with the GPIB-USB-HS Adapter however is operating system constrained and only worked successfully on Windows.

Later, to embed the control of the lab equipment into Python programs, the library `pyvisa` is used that allows for simple control of the devices. First the GPIB address of the device needs to be requested. This can be done by running the `rm.list_resources()` command, where `rm` is previously initialized as in the example below.

Furthermore, the commands for each device are different and can be looked up in the user and programmer guides online. The Keysight Spectrometers guide can be downloaded [here](https://www.keysight.com/).

```python
# GPIB: Setup and send a command
import pyvisa

rm = pyvisa.ResourceManager()
my_instrument = rm.open_resource('GPIB0::13::INSTR')  # Connect to the device with GPIB Address 'GPIB0::13::INSTR'

my_instrument.write("1000 MH -100 DB")  # Set the output to 1GHz with -100dBm output level
```

For the signal generator, the command above shows the only relevant one for our applications: setting amplitude and frequency of the output. The spectrum analyzer instead needs many more settings.

### Commands for Keysight EXA
The following important commands were used for the Spectrum Analyzer to make appropriate measurement settings:

```python
exa.write('*RST')  # Reset
exa.write('*CLS')  # Clear Status

exa.write(f'SENS:FREQ:CENT {CENTER_FREQUENCY_HZ} HZ')  # Set Center Frequency
exa.write(f'SENS:FREQ:SPAN {SPAN_HZ} HZ')  # Set Span
exa.write(f'SENS:BAND:RES {RESOLUTION_BW_HZ} HZ')  # Set Resolution Bandwidth
exa.write('CALC:MARK1:MODE POS')  # Set Marker with Peak Search
exa.write('SENS:AVER ON')  # Activate Averaging
exa.write(f'DISP:WIND:TRAC:Y:RLEV {REFERENCE_LEVEL_DBM} DBM')  # Set reference level
exa.write(f'SENS:POW:ATT {ATTENUATOR_DB} DB')  # Set attenuator level
exa.write('SENS:POW:GAIN:STAT OFF')  # Deactivate LNA

sweep_time_str = exa.query('SENS:SWE:TIME?')  # Query sweep time
exa.write('SENS:AVER:CLE')  # Clear Averages
measured_power_str = exa.query('CALC:MARK1:Y?')  # Query measured power
```
