"""
Signal Generator Calibration Data Analysis Script

This script analyzes the calibration measurement data. It loads data from both the high and low range measurements, 
performs linear fitting analysis, and generates calibration plots and parameters to find offsets in the actual 
signal generators output power.
"""

import numpy as np
import matplotlib.pyplot as plt

# Default file paths for calibration data
DEFAULT_HIGHRANGE_FILE = ""
DEFAULT_LOWRANGE_FILE = ""


def load_calibration_data(lowrange_file, highrange_file):
    """
    Load and combine calibration data from low and high range measurement files.
    
    Args:
        lowrange_file (str): Path to low range measurement data file
        highrange_file (str): Path to high range measurement data file
        
    Returns:
        numpy.ndarray: Combined calibration data array
    """
    data_low = np.genfromtxt(lowrange_file, delimiter=';', encoding='utf-8', names=True, dtype=None)
    data_high = np.genfromtxt(highrange_file, dtype=None, delimiter=';', names=True, encoding='utf-8')
    combined_data = np.concatenate((data_low, data_high))
    return combined_data

def perform_linear_fit(data):
    """
    Perform linear regression analysis on calibration data.
    
    Args:
        data (numpy.ndarray): Calibration data with set and measured power columns
        
    Returns:
        tuple: (fit_coefficients, x_values_for_plot, y_values_for_plot)
    """
    x_fit = np.linspace(data['Set_Power_dBm'].min(), data['Set_Power_dBm'].max(), 200)
    coeffs, residuals, _, _, _ = np.polyfit(data['Set_Power_dBm'], data['Measured_Power_dBm'], deg=1, full=True)
    fit_linear = np.poly1d(coeffs)
    y_fit_linear = fit_linear(x_fit)
    return coeffs, x_fit, y_fit_linear


def create_calibration_plot(data, x_fit, y_fit, coeffs):
    """
    Generate calibration plot showing measured data and linear fit.
    
    Args:
        data (numpy.ndarray): Calibration measurement data
        x_fit (numpy.ndarray): X values for fitted line
        y_fit (numpy.ndarray): Y values for fitted line  
        coeffs (numpy.ndarray): Linear fit coefficients [slope, intercept]
    """
    plt.figure()
    plt.plot(x_fit, y_fit, '-', color='red', label=f'Linear Fit f(x)={coeffs[0]:.4f}x{coeffs[1]:.4f}')
    plt.plot(data['Set_Power_dBm'], data['Measured_Power_dBm'], marker='o', linestyle='None', ms=2, label='Measured Data')
    plt.title('Set Power vs Measured Power')
    plt.legend()
    plt.grid()
    plt.xlabel('Set Power on Signal Generator (dBm)')
    plt.ylabel('Measured Power with Spectrum Analyzer (dBm)')
    plt.show()


def save_fit_parameters(coeffs, filename='linear_fit_params_signalgenerator.txt'):
    """
    Save linear fit parameters to text file for future reference.
    
    Args:
        coeffs (numpy.ndarray): Linear fit coefficients [slope, intercept]
        filename (str): Output filename for parameters
    """
    np.savetxt(filename, coeffs, header='slope intercept')
    print(f'Linear fit parameters saved to: {filename}')


def main():
    """
    Execute complete calibration data analysis workflow.
    """
    # Load calibration data from measurement files
    data = load_calibration_data(DEFAULT_LOWRANGE_FILE, DEFAULT_HIGHRANGE_FILE)
    
    # Perform linear regression analysis
    coeffs, x_fit, y_fit = perform_linear_fit(data)
    
    # Generate calibration plot
    create_calibration_plot(data, x_fit, y_fit, coeffs)
    
    # Display and save fit parameters
    print(f'Linear Fit Parameters: f(x)= {coeffs[0]}x {coeffs[1]}')
    save_fit_parameters(coeffs)


if __name__ == "__main__":
    main()