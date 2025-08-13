"""
Gain Drift Measurement Data Analysis and Visualization

This script processes gain drift measurement data from CSV files containing timestamp,
measured power, and optional standard deviation columns. It performs data interpolation,
smoothing, and generates plots showing both absolute power measurements and relative
drift over time for SDR device characterization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def plot_data(filepath):
    """
    Processes a single CSV measurement file and generates a gain drift analysis plot.
    
    Args:
        filepath (str): Path to the CSV file
    """
    if not os.path.isfile(filepath):
        print(f"[WARN] File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Could not read file {filepath}: {e}")
        return

    required_columns = ['timestamp', 'measured_power_dBm']
    if not all(col in df.columns for col in required_columns):
        print(f"[ERROR] Missing columns in {filepath}. Required: {', '.join(required_columns)}")
        return

    timestamps = df['timestamp'].values
    power_dBm = df['measured_power_dBm'].values

    use_error = 'standard_deviation' in df.columns
    std_dev = df['standard_deviation'].values if use_error else None

    # Linear interpolation for uniform time grid
    interpolator = interp1d(timestamps, power_dBm, kind='linear', fill_value='extrapolate')
    t_interp = np.linspace(timestamps[0], timestamps[-1], num=1000)
    p_interp = interpolator(t_interp)
    
    # Apply Savitzky-Golay filter for noise reduction
    p_smooth = savgol_filter(p_interp, window_length=51, polyorder=2)

    # Calculate relative drift as percentage from initial measurement
    relative_diff = (p_smooth - p_smooth[0]) / abs(p_smooth[0]) * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary y-axis: Power measurements in dBm
    color1 = 'tab:blue'
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Power [dBm]", color=color1)
    ax1.plot(t_interp, p_smooth, color=color1, label='Smoothed Power')
    if use_error:
        ax1.errorbar(timestamps, power_dBm, yerr=std_dev, fmt='o', alpha=0.3, label='Raw Data ± Std Dev')
    else:
        ax1.plot(timestamps, power_dBm, 'o', alpha=0.3, label='Raw Data')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper left')

    # Secondary y-axis: Relative drift percentage
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("Relative Change [%]", color=color2)
    ax2.plot(t_interp, relative_diff, color=color2, linestyle='--', label='Relative Drift')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color='gray', linestyle=':')  # Zero reference line

    plt.title("Gain Drift over Time with Interpolation and Relative Change")
    fig.tight_layout()

    # Save plot with .plot.pdf suffix in same directory
    plot_path = filepath.replace('.csv', '.plot.pdf')
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Plot saved: {plot_path}")

def main():
    """
    Main execution function that processes predefined measurement files.
    """
    # CSV measurement files to process
    files_to_plot = [
        ""
    ]

    for file in files_to_plot:
        plot_data(file)

if __name__ == "__main__":
    main()