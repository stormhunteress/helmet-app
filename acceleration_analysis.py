"""
Backend module for acceleration data analysis.
Performs Welch power spectral density estimation on 3-axis acceleration data.
"""

from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def load_acceleration_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load acceleration data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (Accelx, Accely, Accelz, num_columns)
    """
    data = pd.read_csv(file_path)
    
    rownum = np.shape(data)[0]
    colnum = np.shape(data)[1]
    
    if colnum == 5:
        Accelx = data.iloc[:, 2].values
        Accely = data.iloc[:, 3].values
        Accelz = data.iloc[:, 4].values
    elif colnum == 4:
        Accelx = data.iloc[:, 1].values
        Accely = data.iloc[:, 2].values
        Accelz = data.iloc[:, 3].values
    else:
        raise ValueError(f"Expected 4 or 5 columns, got {colnum}")
    
    return Accelx, Accely, Accelz, colnum


def compute_welch_psd(Accelx: np.ndarray, Accely: np.ndarray, Accelz: np.ndarray,
                      fs: int = 5000, nperseg: int = 1024, noverlap: int = 50) -> Dict:
    """
    Compute Welch power spectral density estimation for 3-axis acceleration data.
    
    Args:
        Accelx, Accely, Accelz: Acceleration arrays
        fs: Sampling frequency in Hz
        nperseg: Length of each segment for Welch method
        noverlap: Number of points to overlap between segments
        
    Returns:
        Dictionary containing frequency array and PSD estimates
    """
    # Welch estimation power density
    fwx, Pxx_welx = signal.welch(Accelx, fs, nperseg=nperseg, noverlap=noverlap)
    fwy, Pxx_wely = signal.welch(Accely, fs, nperseg=nperseg, noverlap=noverlap)
    fwz, Pxx_welz = signal.welch(Accelz, fs, nperseg=nperseg, noverlap=noverlap)
    
    # Calculate average Welch estimation
    Pxx_total = Pxx_welx + Pxx_wely + Pxx_welz
    
    # Find dominant frequency
    argmax = np.argmax(abs(Pxx_total))
    Mode_1 = fwx[argmax]
    
    return {
        'frequency': fwx,
        'psd_x': Pxx_welx,
        'psd_y': Pxx_wely,
        'psd_z': Pxx_welz,
        'psd_total': Pxx_total,
        'mode_1_freq': Mode_1,
        'mode_1_index': argmax,
        'sampling_rate': fs
    }


def plot_psd(results: Dict, x_limit: float = 1000) -> plt.Figure:
    """
    Create a plot of the power spectral density.
    
    Args:
        results: Dictionary from compute_welch_psd
        x_limit: Upper limit for x-axis frequency display
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(results['frequency'], abs(results['psd_total']))
    ax.axvline(results['mode_1_freq'], color='r', linestyle='--', 
               label=f"Mode 1: {results['mode_1_freq']:.2f} Hz")
    ax.set_title("Power Spectral Density (Average)")
    ax.set_xlim(-0.5, x_limit)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [V²/Hz]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def plot_individual_axes(results: Dict, x_limit: float = 1000) -> plt.Figure:
    """
    Create individual plots for each axis.
    
    Args:
        results: Dictionary from compute_welch_psd
        x_limit: Upper limit for x-axis frequency display
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    freq = results['frequency']
    axes[0].semilogy(freq, abs(results['psd_x']))
    axes[0].set_title("X-Axis PSD")
    axes[0].set_xlim(-0.5, x_limit)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('PSD [V²/Hz]')
    
    axes[1].semilogy(freq, abs(results['psd_y']))
    axes[1].set_title("Y-Axis PSD")
    axes[1].set_xlim(-0.5, x_limit)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('PSD [V²/Hz]')
    
    axes[2].semilogy(freq, abs(results['psd_z']))
    axes[2].set_title("Z-Axis PSD")
    axes[2].set_xlim(-0.5, x_limit)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel('PSD [V²/Hz]')
    axes[2].set_xlabel('Frequency [Hz]')
    
    plt.tight_layout()
    return fig
