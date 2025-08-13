"""
SDR Dynamic Range Analysis and Offset Correction Script

This script analyzes dynamic range measurements from Software Defined Radio (SDR) devices
including RTL-SDR, PlutoSDR, HackRF, and USRP B210/B200mini. It processes power sweep data
to determine optimal calibration offsets for accurate power measurements and generates
visualization plots of the dynamic range characteristics.

For detailed documentation and measurement procedures, refer to the accompanying 
dynamic_range_guide.md file in the same folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

def extract_gain_from_filename(filename: str) -> float:
    """
    Extracts the SDR gain value from a measurement filename.
    
    Parses filenames with the pattern "Gain_XXdB" to extract the gain setting
    used during the measurement.
    """
    if "Gain_" in filename and "dB" in filename:
        try:
            return float(filename.split("Gain_")[1].split("dB")[0])
        except ValueError:
            pass
    return 0.0


def load_power_sweep_data(folder: str, slope: float = 1.0, intercept: float = 0.0) -> list:
    """
    Loads all power sweep measurement files from a folder.
    
    Processes measurement files with naming pattern "*Gain_XXdB.txt" and applies
    optional linear correction to input power values.
    Returns a list of dictionaries containing gain, input power, and measured power data.
    """
    data_list = []
    for file in os.listdir(folder):
        if not file.endswith('dB.txt'):
            continue
        path = os.path.join(folder, file)
        gain = extract_gain_from_filename(file)
        try:
            data = np.genfromtxt(path, comments='#', delimiter=',')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            input_power = data[:, 0] * slope + intercept
            measured_power = data[:, 1]
            data_list.append({
                'gain': gain,
                'uncorrected_input_power': input_power,
                'measured_power': measured_power
            })
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    return sorted(data_list, key=lambda d: d['gain'])

def plot_power_sweep(dataset: list, x_data_key: str, y_data_key: str, ylabel: str, title: str, output_path: str, cutoff = False, raw = True):
    """
    Generates visualization plots of power sweep measurements.
    
    Creates scatter plots showing the relationship between input and measured power
    for different gain settings. Supports optional y-axis cutoff limits for better
    visualization of specific power ranges.
    """
    plt.figure(figsize=(10, 6))

    for idx, entry in enumerate(dataset):
        x = entry[x_data_key]
        y = entry[y_data_key]
        label = f"Gain: {entry['gain']} dB"
        plt.plot(x, y, 'o', markersize=5, label=label)

    if cutoff and raw: plt.ylim(bottom=-65, top=5)        # Limit raw data view to useful range
    if cutoff and not raw: plt.ylim(bottom=-120)          # Limit corrected data view
    plt.xlabel("Available Input Power (dBm)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    # plt.title(title)  # Title commented out for cleaner plots
    plt.savefig(output_path, dpi=600)
    plt.close()


def find_max_linear_region_with_fit(x, y, min_points=10, r2_threshold=0.999):
    """
    Identifies the longest linear region in measurement data with high correlation.
    
    Iteratively tests all possible sub-ranges of the data to find the longest
    continuous region that maintains linear behavior above the specified R² threshold.
    Returns the best linear fit parameters and data range.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)  # Remove NaN values from analysis
    x = x[mask]
    y = y[mask]

    n = len(x)
    best_start, best_end = None, None
    best_length = 0
    best_r2 = -np.inf
    best_model = None

    # Search all possible sub-ranges for best linear fit
    for start in range(n):
        for end in range(start + min_points, n + 1):
            x_sub = x[start:end].reshape(-1, 1)
            y_sub = y[start:end]
            model = LinearRegression().fit(x_sub, y_sub)
            y_pred = model.predict(x_sub)
            r2 = r2_score(y_sub, y_pred)

            # Update best fit if this region is longer and meets quality threshold
            if r2 >= r2_threshold and (end - start) > best_length:
                best_length = end - start
                best_start, best_end = start, end
                best_r2 = r2
                best_model = model

    if best_start is not None:
        x_best = x[best_start:best_end]
        y_best = y[best_start:best_end]
        slope = best_model.coef_[0]
        offset = best_model.intercept_
        return x_best, y_best, best_r2, slope, offset
    else:
        return np.array([]), np.array([]), None, None, None

def find_offset_and_linear_region(data, x_data_string, folder, slope_input_power, offset_input_power, sdr_name):
    """
    Analyzes measurement data to determine calibration offsets and linear regions.
    
    For each gain setting, finds the optimal linear operating region and calculates
    the calibration offset needed to align measured power with expected values.
    Results are saved as both CSV and human-readable text files.
    """
    # Analysis for all gain settings
    results = []

    for entry in data:
        gain = entry['gain']
        x = entry[x_data_string]
        y = entry['measured_plus_gain']
        x_best, y_best, r2_best, slope, offset = find_max_linear_region_with_fit(x, y)

        results.append({
            'gain': gain,
            'linear_input_power_start': x_best[0] if len(x_best) > 0 else None,
            'linear_input_power_end': x_best[-1] if len(x_best) > 0 else None,
            'linear_measured_power_start': y_best[0] if len(y_best) > 0 else None,
            'linear_measured_power_end': y_best[-1] if len(y_best) > 0 else None,
            'r_squared': r2_best,
            'slope': slope,
            'offset': offset,
            'final_offset': offset-gain if offset is not None else None
        })

    # Save results as DataFrame for further processing
    df = pd.DataFrame(results)
    output_path = os.path.join(folder, f'linear_fit_{slope_input_power:.2f}_{offset_input_power:.2f}_input_power.csv')
    df.to_csv(output_path, sep="\t", index=False)

    # Save detailed human-readable report
    output_path = os.path.join(folder, f'linear_fit_{slope_input_power:.2f}_{offset_input_power:.2f}_input_power.txt')
    with open(output_path, "w") as f:
        f.write(f"SDR: {sdr_name}\n\n")

        for _, row in df.iterrows():
            f.write(f"Gain: {row['gain']}\n")
            f.write(f"  Input Power Range: {row['linear_input_power_start']} to {row['linear_input_power_end']}\n")
            f.write(f"  Measured Power Range: {row['linear_measured_power_start']} to {row['linear_measured_power_end']}\n")
            f.write(f"  R²: {row['r_squared']:.6f}\n")
            f.write(f"  Slope: {row['slope']:.6f}, Offset: {row['offset']:.6f}\n")
            f.write(f"  Offset - Gain: {row['final_offset']:.6f}\n")
            f.write("\n")


if __name__ == "__main__":
    # Define measurement folders and corresponding SDR device names
    # Replace these paths with your actual measurement directories
    # Note: The script generates both full-range and cutoff-limited plots for better comparison.
    # Cutoff plots restrict the y-axis range to focus on the useful measurement range,
    # making it easier to compare performance between different SDR devices.
    folders = [
        (r"/path/to/RTLSDR_Measurements/2025-06-24_16-31-02_Power_-120.0_0.0_1.0_Gain_0_70_10", "RTL"),
        (r"/path/to/PLUTOSDR_Measurements/2025-06-24_15-25-32", "PLUTO"),
        (r"/path/to/HACKRF_Measurements/2025-06-24_15-55-26", "HACKRF"),
        (r"/path/to/USRPSDR_Measurements/2025-06-30_15-57-14_Power_-120.0_0.0_1.0_Gain_0_70_10", "USRP B210"),
        (r"/path/to/USRPSDR_Measurements/B200mini_2025-06-30_15-28-56_Power_-120.0_0.0_1.0_Gain_0_70_10", "USRP B200mini")
    ]

    for folder, sdr_name in folders:

        data = load_power_sweep_data(folder)

        # Generate raw plots which only show the measured power. Because in the measurement the gain is already subtracted from the measured output power, it is added again
        for entry in data:
            entry['measured_plus_gain'] = entry['measured_power'] + entry['gain']


        # The following is experimental for further processing the data with the signal generator calibration coefficients
        # Plot the data for different slope and intercept values, calculated by calibrating the signal generator
        # Load signal generator calibration coefficients
        coeffs = np.loadtxt('/path/to/signal_generator_calibration/linear_fit_params_signalgenerator.txt')
        slope, intercept = coeffs

        # Apply signal generator correction to input power values
        for entry in data:
            entry['corrected_input_power'] = entry['uncorrected_input_power'] * slope + intercept

        # Generate plots with signal generator calibration applied
        plot_power_sweep(
            dataset=data,
            x_data_key= 'corrected_input_power',
            y_data_key='measured_plus_gain',
            ylabel='Measured Output Power (dBm)',
            title=f'{sdr_name}: Measured Output Power vs Input Power',
            output_path=os.path.join(folder, f'{sdr_name}_plot_raw_gain_{slope:.2f}_{intercept:.2f}.pdf'),
            raw = True
        )

        # Generate limited range plots for better visualization
        plot_power_sweep(
            dataset=data,
            x_data_key= 'corrected_input_power',
            y_data_key='measured_plus_gain',
            ylabel='Measured Output Power (dBm)',
            title=fr'{sdr_name}: Measured Output Power (shown up to -65 dbm) vs Input Power',
            output_path=os.path.join(folder, f'{sdr_name}_plot_raw_gain_cutoff_{slope:.2f}_{intercept:.2f}.pdf'),
            cutoff=True,
            raw=True
        )

        # Generate corrected power plots with signal generator calibration
        plot_power_sweep(
            dataset=data,
            x_data_key= 'corrected_input_power',
            y_data_key='measured_power',
            ylabel='Retrieved Input Power (dBm)',
            title=f'{sdr_name}: Retrieved Input Power vs Input Power',
            output_path=os.path.join(folder, f'{sdr_name}_plot_corrected_gain_{slope:.2f}_{intercept:.2f}.pdf'),
            raw = False
        )
        
        # Generate limited range corrected plots
        plot_power_sweep(
            dataset=data,
            x_data_key= 'corrected_input_power',
            y_data_key='measured_power',
            ylabel='Retrieved Input Power (dBm)',
            title=fr'{sdr_name}: Retrieved Input Power (shown up to -120 dbm) vs Input Power',
            output_path=os.path.join(folder, f'{sdr_name}_plot_corrected_gain_cutoff_{slope:.2f}_{intercept:.2f}.pdf'),
            cutoff=True,
            raw = False
        )

        # Perform linear region analysis with corrected and uncorrected data
        find_offset_and_linear_region(data, 'corrected_input_power', folder, slope, intercept, sdr_name)