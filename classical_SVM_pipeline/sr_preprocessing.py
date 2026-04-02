"""
SR-Specific Preprocessing for Maiti 2021 Replication
Step 1: Mean centering and normalization within spectral ranges
"""

import numpy as np
from typing import Dict, Tuple

# Spectral ranges from Table 1 of the paper
SR_CENTERS = {
    'SR_1005': 1005,  # Acetic anhydride
    'SR_1190': 1190,  # Propyl propionate
    'SR_1203': 1203,  # Ethyl vinyl ketone
    'SR_530': 530,    # Acetaldehyde
    'SR_1050': 1050,  # Carbon dioxide
    'SR_2170': 2170,  # Carbon monoxide
    'SR_1130': 1130,  # Ethyl pyruvate
    'SR_1170': 1170,  # Methyl butyrate
}


def extract_sr_window(spectrum: np.ndarray, 
                      wavenumbers: np.ndarray, 
                      center: float, 
                      window_width: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a spectral range window centered at 'center' with given width.
    
    Parameters:
    -----------
    spectrum : np.ndarray
        Full absorption spectrum (preprocessed: baseline corrected)
    wavenumbers : np.ndarray
        Wavenumber axis matching spectrum
    center : float
        Center of the spectral range in cm^-1
    window_width : float
        Width of window in cm^-1 (default 30, so ±15 cm^-1 from center)
    
    Returns:
    --------
    sr_spectrum : np.ndarray
        Extracted spectral window
    sr_wavenumbers : np.ndarray
        Corresponding wavenumber axis
    """
    half_width = window_width / 2.0
    mask = (wavenumbers >= center - half_width) & (wavenumbers <= center + half_width)
    
    sr_spectrum = spectrum[mask]
    sr_wavenumbers = wavenumbers[mask]
    
    return sr_spectrum, sr_wavenumbers


def mean_center_sr(sr_spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Mean center a spectral range.
    
    Parameters:
    -----------
    sr_spectrum : np.ndarray
        Spectral range data
    
    Returns:
    --------
    centered_spectrum : np.ndarray
        Mean-centered spectrum
    mean_value : float
        The mean that was subtracted (for reference)
    """
    mean_value = np.mean(sr_spectrum)
    centered_spectrum = sr_spectrum - mean_value
    
    return centered_spectrum, mean_value


def normalize_sr(centered_spectrum: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize a mean-centered spectral range by its standard deviation.
    
    Parameters:
    -----------
    centered_spectrum : np.ndarray
        Mean-centered spectrum
    
    Returns:
    --------
    normalized_spectrum : np.ndarray
        Normalized spectrum (zero mean, unit variance)
    std_value : float
        The standard deviation used (for reference)
    """
    std_value = np.std(centered_spectrum)
    
    # Avoid division by zero (though shouldn't happen with real spectral data)
    if std_value < 1e-10:
        print(f"Warning: Very small std ({std_value}), using 1.0 instead")
        std_value = 1.0
    
    normalized_spectrum = centered_spectrum / std_value
    
    return normalized_spectrum, std_value


def preprocess_sr(sr_spectrum: np.ndarray) -> Dict:
    """
    Complete preprocessing pipeline for a single spectral range.
    Order: mean centering → normalization
    
    Parameters:
    -----------
    sr_spectrum : np.ndarray
        Raw spectral range (after baseline correction)
    
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'preprocessed': final preprocessed spectrum
        - 'mean': mean value that was subtracted
        - 'std': std value used for normalization
    """
    # Step 1: Mean center
    centered, mean_val = mean_center_sr(sr_spectrum)
    
    # Step 2: Normalize
    normalized, std_val = normalize_sr(centered)
    
    return {
        'preprocessed': normalized,
        'mean': mean_val,
        'std': std_val
    }


def preprocess_all_srs(spectrum: np.ndarray, 
                       wavenumbers: np.ndarray,
                       window_width: float = 30.0) -> Dict:
    """
    Extract and preprocess all 8 spectral ranges from a full spectrum.
    
    Parameters:
    -----------
    spectrum : np.ndarray
        Full baseline-corrected spectrum
    wavenumbers : np.ndarray
        Wavenumber axis
    window_width : float
        Width of SR windows in cm^-1
    
    Returns:
    --------
    results : dict
        Dictionary with keys = SR names, values = preprocessing results
    """
    results = {} 

    for sr_name, center in SR_CENTERS.items():
        # Extract window
        sr_spec, sr_wn = extract_sr_window(spectrum, wavenumbers, center, window_width)
        
        # Preprocess
        preprocessed = preprocess_sr(sr_spec)
        
        # Store results
        results[sr_name] = {
            'spectrum': preprocessed['preprocessed'],
            'wavenumbers': sr_wn,
            'mean': preprocessed['mean'],
            'std': preprocessed['std'],
            'center': center
        }
    
    return results

