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
    Automatically skips metadata rows until it finds the x, y, z header.
    Supports multiple CSV formats.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (Accelx, Accely, Accelz, num_columns)
    """
    import csv
    
    try:
        # Manual CSV parsing to handle metadata and inconsistent row lengths
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if len(rows) < 1:
            raise ValueError("CSV file is empty")
        
        # Find the header row (contains x, y, z or similar)
        header_idx = None
        for idx, row in enumerate(rows):
            row_lower = [col.lower().strip() for col in row]
            # Check if this row contains x, y, z columns
            if any('x' in col for col in row_lower) and any('y' in col for col in row_lower) and any('z' in col for col in row_lower):
                header_idx = idx
                break
        
        if header_idx is None:
            raise ValueError("Could not find header row with 'x', 'y', 'z' columns. Check your CSV format.")
        
        # Use the found header
        header = [col.lower().strip() for col in rows[header_idx]]
        colnum = len(header)
        
        # Find x, y, z column indices
        x_col = None
        y_col = None
        z_col = None
        
        for i, col in enumerate(header):
            if 'x' in col and x_col is None:
                x_col = i
            elif 'y' in col and y_col is None:
                y_col = i
            elif 'z' in col and z_col is None:
                z_col = i
        
        if x_col is None or y_col is None or z_col is None:
            raise ValueError("Could not find x, y, z columns in header row")
        
        # Extract data rows (skip header and all rows before it)
        x_data = []
        y_data = []
        z_data = []
        
        for row_idx, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
            try:
                # Skip empty rows
                if not row or all(not col.strip() for col in row):
                    continue
                
                # Extract values, handling rows with different lengths
                x_val = None
                y_val = None
                z_val = None
                
                if x_col < len(row) and row[x_col].strip():
                    x_val = pd.to_numeric(row[x_col], errors='coerce')
                
                if y_col < len(row) and row[y_col].strip():
                    y_val = pd.to_numeric(row[y_col], errors='coerce')
                
                if z_col < len(row) and row[z_col].strip():
                    z_val = pd.to_numeric(row[z_col], errors='coerce')
                
                # Only add if all three values are valid
                if pd.notna(x_val) and pd.notna(y_val) and pd.notna(z_val):
                    x_data.append(x_val)
                    y_data.append(y_val)
                    z_data.append(z_val)
            except Exception as e:
                continue  # Skip problematic rows
        
        Accelx = np.array(x_data, dtype=float)
        Accely = np.array(y_data, dtype=float)
        Accelz = np.array(z_data, dtype=float)
        
        if len(Accelx) == 0:
            raise ValueError("No valid acceleration data found in CSV")
        
        return Accelx, Accely, Accelz, colnum
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


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
