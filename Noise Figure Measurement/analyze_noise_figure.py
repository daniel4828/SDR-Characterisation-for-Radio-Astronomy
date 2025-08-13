"""
Noise Figure vs Gain Analysis Script for SDR Receiver Characterization

This script analyzes noise figure measurements from various SDRs (PLUTO, RTL, USRP, HACKRF)
with optional LNA correction using Friis formula. Generates comparative plots showing gain vs noise figure
characteristics across different SDR platforms. 

See noise_figure_measurement_guide.md in this folder for detailed measurement procedures and setup instructions.
"""

import matplotlib.pyplot as plt
import numpy as np

def db_to_abs(db_val):
    """Converts a dB value to absolute (linear) value."""
    return 10**(db_val / 10)

def abs_to_db(abs_val):
    """Converts an absolute (linear) value to dB value."""
    return 10 * np.log10(abs_val)

def friis_noise_figure_correction(measured_nf_db, measured_gain_db, lna_nf_db, lna_gain_db):
    """
    Corrects measured noise figure using Friis formula to remove LNA contribution.
    Calculates DUT noise figure from cascaded LNA+DUT measurement results.
    """
    F_measured_abs = db_to_abs(measured_nf_db)
    F_lna_abs = db_to_abs(lna_nf_db)
    G_lna_abs = db_to_abs(lna_gain_db)

    # Check for unphysical measurement points where LNA NF exceeds total measured NF
    problematic_points = (F_measured_abs - F_lna_abs) < 0
    if np.any(problematic_points):
        print(f"Warning: {np.sum(problematic_points)} measurement points have (F_measured_abs - F_lna_abs) < 0. "
              f"This may lead to unphysical results. "
              f"LNA NF: {lna_nf_db:.2f} dB")
        
    F_dut_abs = (F_measured_abs - F_lna_abs) * G_lna_abs + 1
    return abs_to_db(F_dut_abs)


# Plot configuration
plot_title = "Gain vs. Noise Figure Measurements"

# LNA parameters for noise figure correction (when LNA is used in measurement chain)
LNA_NOISE_FIGURE_DB = 0.5  # LNA noise figure from datasheet (dB)
LNA_GAIN_DB = 15.24        # Measured LNA gain (dB)

# Measurement file specifications
# Each tuple contains: (file_path, legend_name, apply_lna_correction, unreliable_gain_threshold)
measurement_files = [
    ("./measurements/sdr1_noise_figure_results.txt", "PLUTO SDR (with LNA)", True, 10.0),
    ("./measurements/sdr2_noise_figure_results.txt", "RTL SDR", False, 7.5),
    ("./measurements/sdr3_noise_figure_results.txt", "USRP B200 SDR (with LNA)", True, 22.5),
    ("./measurements/sdr4_noise_figure_results.txt", "USRP B210 SDR (with LNA)", True, 7.5),
    ("./measurements/sdr5_noise_figure_results.txt", "HACKRF SDR (with LNA)", True, 50.0),
]

# Color palette for plotting multiple SDR datasets
colors = plt.colormaps.get_cmap('tab10') 

plt.figure(figsize=(10, 6))
ax = plt.gca() 

for i, (file_path, legend_name, apply_correction, unreliable_gain_threshold) in enumerate(measurement_files):
    gains = []
    noise_figures = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Skip header lines and comments in measurement data files
                if line.startswith('#') or line.startswith('SDR_Gain_dB'):
                    continue
                parts = line.strip().split(',')
                try:
                    gain = float(parts[0])      # SDR gain setting (dB)
                    nf = float(parts[4])        # Measured noise figure (dB)
                    # Only include valid (non-NaN) noise figure measurements
                    if not np.isnan(nf): 
                        gains.append(gain)
                        noise_figures.append(nf)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Skipping this measurement series.")
        continue
    except Exception as e:
        print(f"Error reading '{file_path}': {e}. Skipping this measurement series.")
        continue

    gains = np.array(gains)
    noise_figures = np.array(noise_figures)

    # Apply Friis correction to remove LNA contribution from measurement results
    if apply_correction:
        print(f"Applying Friis correction to '{file_path}' with LNA NF={LNA_NOISE_FIGURE_DB} dB, LNA Gain={LNA_GAIN_DB} dB.")
        corrected_noise_figures = friis_noise_figure_correction(
            noise_figures, gains, LNA_NOISE_FIGURE_DB, LNA_GAIN_DB
        )
        plot_nf = corrected_noise_figures
    else:
        plot_nf = noise_figures

    current_color = colors(i / len(measurement_files)) 

    # Separate reliable and unreliable measurements based on gain threshold
    reliable_mask = gains >= unreliable_gain_threshold
    unreliable_mask = gains < unreliable_gain_threshold

    # Plot reliable measurements with solid markers
    plt.plot(gains[reliable_mask], plot_nf[reliable_mask], 'o', color=current_color, label=legend_name)

    # Plot unreliable measurements with different marker style (crosses)
    plt.plot(gains[unreliable_mask], plot_nf[unreliable_mask], 'x',
             color=current_color, alpha=0.7, markeredgecolor='black', markersize=7)

    # Connect all points with dashed line for trend visualization
    plt.plot(gains, plot_nf, '--', color=current_color, alpha=0.6, label='_nolegend_') 

plt.xlabel("Gain (dB)")
plt.ylabel("Noise Figure (dB)")
plt.title(plot_title)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()