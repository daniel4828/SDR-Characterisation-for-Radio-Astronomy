# SDR Measurement with GNU Radio and PyVISA

This repository contains Python programs for measuring signal power with various Software Defined Radios (SDRs) and controlling laboratory equipment for radiometric experiments. The programs are based on GNU Radio flowgraphs that were extended with PyVISA for instrument control via GPIB.


## 1. Introduction

[GNU Radio](https://www.gnuradio.org/) is an open-source toolkit for building signal processing applications and interacting with SDRs. It provides:

- **A graphical interface (GNU Radio Companion, GRC)** for creating flowgraphs.
- **Python export** functionality, allowing flowgraphs to be converted into standalone Python scripts.

The Python scripts in this repository were generated from GNU Radio flowgraphs and then extended to:

- **Control laboratory instruments** (e.g., signal generators, spectrum analyzers) via **PyVISA** and a GPIB adapter.
- **Synchronize** signal processing and instrument control using multi-threaded execution.

---

## 2. Installation and Environment Setup

To run the programs reliably, use **Radioconda**, which provides all required GNU Radio dependencies:

1. Install **Radioconda** from:  
   [https://github.com/ryanvolz/radioconda](https://github.com/ryanvolz/radioconda)
   Though the graphical interface of radioconda is nice, only the shell is needed to run the scripts. This can be helpful when using a Mac with a Silicon Chip, where we had problems installing the GUI. 

2. Depending on your device you might also need to install device support for your SDR. This is all explained in the above github repo. On Windows there were problems with the Lime SDR that could not be solved an on Mac M1 with the RTL SDR. To test whether the SDRs are found you can use the command
```bash
SoapySDRUtil --find
```
3. To interface with the Signal Generator or Spectrum Analyzer, the backend NI-VISA is required to be installed [NI-VISA Download](https://www.ni.com/de/support/documentation/compatibility/21/ni-hardware-and-operating-system-compatibility.html). The interface with the GPIB-USB-HS Adapter however is operating system constrained and only worked successfully on Windows. For more information on GPIB checkout file `GPIB.md`.

4. Open the **Radioconda shell** and install PyVISA to use the GPIB protocoll:

```bash
pip install pyvisa pyvisa-py
```
5. Now you are able to connect your PC to the SDRs and to GPIB controllable lab devices via the GPIB adapter.

