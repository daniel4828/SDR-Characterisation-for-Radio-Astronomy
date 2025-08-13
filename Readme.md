# The Project
This repository was created during participation in the module "SIERRA - Radio Astronomy" at TU Berlin. The aim of the project is to detect thermal radiation from celestial bodies, in particular the Sun, using a parabolic reflector and to derive its temperature. This concept builds on the work of Schiavolini et al. (2025 - Low-Cost Calibrated Microwave Radiometers for Sola), in which a low-cost microwave radiometer was implemented using a low-noise downconverter originally intended for satellite TV reception and a software-defined radio.This repository was created by WP100. The task of this work package was to characterize different SDRs. Although SDRs have been on the market for a long time, it is unclear to what extent they are suitable for radio astronomy.

# About this Repository
The scripts in this repository allow characterization of SDRs regarding their dynamic range, noise figure, gain drift, and optimal integration time. These scripts automate the measurements. The GPIB protocol from National Instruments and GNU Radio are used to control the SDRs. More information can be found in the Technical Report, which contains all the results obtained from the measurements.

The examined SDRs are the RTL-SDR, HackRF, PlutoSDR, LimeSDR, USRP B210 and USRP B200mini.

Each folder contains the relevant Python scripts to perform and evaluate the measurements, as well as a guide on how to conduct them. GNU Radio, NIVisa, and PyVisa are required for the measurements. Information on downloading and usage can be found in `GNU Radio and PyVisa.md` and `GPIB and NIVisa`.

Enjoy!

Artificial Intelligence (AI) tools were used for this work. These tools were employed for various tasks, including translations of text from German to English, improving formulations, correcting grammar and spelling in the final texts, as well as cleaning, commenting, generating, and debugging Python code for plotting and automation.

