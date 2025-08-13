"""
Allan Deviation Data Visualization Tool

This script generates plots the Allan Deviation measurement data. Supports batch
processing of multiple measurement sessions and selective gain filtering.

For more information, refer to the
integration_time_measurement_guide.md in the same folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

def extract_gain_from_filename(filename):
    """
    Extract RF gain value from measurement data filename using robust pattern matching.
    
    Args:
        filename: Input filename containing gain information
        
    Returns:
        str: Extracted gain value or "N/A" if pattern not found
    """
    # Primary pattern: "Gain_XXdB" format
    match = re.search(r'Gain_(-?\d+\.?\d*)dB', filename)
    if match:
        return match.group(1)
    
    # Fallback pattern: "XXdB" format
    match = re.search(r'(-?\d+\.?\d*)dB', filename)
    if match:
        return match.group(1)
    return "N/A"

def plot_allan_deviation_from_folder(folder_path, sdr_name, gains=[]):
    """
    Generate Allan Deviation plots from measurement data files in specified directory.
    
    Processes all Allan Deviation text files in the given folder, validates data format,
    and creates logarithmic plots with optional gain filtering for comparative analysis.
    
    Args:
        folder_path: Path to directory containing Allan Deviation measurement files
        sdr_name: SDR platform name for plot labels and output filenames
        gains: Optional list of specific gain values to plot (empty list plots all)
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found: {folder_path}")
        return

    print(f"Scanning directory for Allan Deviation files: {folder_path}")
    
    # Find all Allan Deviation text files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt") and "allan_deviation" in f]
    if not files:
        print("No Allan Deviation files found.")
        return

    files.sort()  # Sort filenames for consistent plotting order
    plt.figure(figsize=(10, 6))

    for idx, filename in enumerate(files):
        filepath = os.path.join(folder_path, filename)
        try:
            # Load data, skipping comment lines starting with '#'
            data = np.genfromtxt(filepath, comments="#", delimiter=",")
            
            # Handle single data point case by expanding dimensions
            if data.ndim == 1 and not np.isnan(data).any():
                data = np.expand_dims(data, axis=0)
            elif data.ndim != 2 or data.shape[1] != 2:
                print(f"  Warning: File {filename} has unexpected format.")
                continue

            # Remove NaN values from Allan Deviation column
            data = data[~np.isnan(data[:, 1])]
            if len(data) == 0:
                print(f"  Warning: No valid data points in file {filename}.")
                continue

            # Extract gain value for plot labeling
            gain = extract_gain_from_filename(filename)
            
            # Apply gain filtering if specified
            if not gains:
                # Plot all available gain settings
                plt.loglog(data[:, 0], data[:, 1], 'o--', label=f"Gain {gain} dB")
            elif float(gain) in gains:
                # Plot only selected gain settings
                plt.loglog(data[:, 0], data[:, 1], 'o--', label=f"Gain {gain} dB")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    # Configure plot appearance for publication quality
    plt.xlabel("Integration Time $\\tau$ (s)")
    plt.ylabel("Allan Deviation $\\sigma_y(\\tau)$")
    plt.grid(True, which="both", linestyle="-", color='gray', alpha=0.2)
    plt.legend()
    plt.tight_layout()

    # Save plot as PDF with descriptive filename
    pdf_path = os.path.join(folder_path, f"{sdr_name}_{gains}_Allan_Deviation_Plot.pdf")
    plt.savefig(pdf_path)
    print(f">>> Plot saved as: {pdf_path}")

if __name__ == "__main__":
    # Configuration for batch processing of Allan Deviation measurement datasets
    # Each tuple contains: (directory_path, gain_filter_list, sdr_platform_name)
    
    # Primary measurement dataset collection
    folders = [
        (r"../RTLSDR/final_2025-07-03_22-04-07", [], "RTL SDR"),
        (r"../PLUTOSDR/final_2025-07-03_20-03-54", [], "PLUTO SDR"),
    ]

    # Process each measurement folder and generate Allan Deviation plots
    for folder, gains, sdr_name in folders:
        plot_allan_deviation_from_folder(folder, sdr_name, gains)
