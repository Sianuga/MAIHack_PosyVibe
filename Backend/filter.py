"""
filter.py - EEG Filtering Functions
===================================

Provides filtering functions for EEG data processing.
Implements bandpass and notch filtering similar to MNE.
"""

import numpy as np
from scipy import signal
from typing import Optional


def filter_eeg_like_mne(
    data: np.ndarray,
    sfreq: float,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freq: Optional[float] = None,
    method: str = 'iir',
    iir_params: Optional[dict] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Filter EEG data with bandpass and notch filters.
    
    This is a simplified version that mimics MNE's filtering behavior
    but uses scipy for actual implementation.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data array of shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    l_freq : float, optional
        Low cutoff frequency for high-pass filter. None to skip high-pass.
    h_freq : float, optional  
        High cutoff frequency for low-pass filter. None to skip low-pass.
    notch_freq : float, optional
        Frequency for notch filter (e.g., 50 or 60 Hz). None to skip notch.
    method : str
        Filter method (only 'iir' is implemented)
    iir_params : dict, optional
        IIR filter parameters (not used in this simplified version)
    verbose : bool
        Whether to print verbose output
        
    Returns
    -------
    np.ndarray
        Filtered data of same shape as input
    """
    
    # Make a copy to avoid modifying the original data
    filtered_data = data.copy()
    
    # Get the Nyquist frequency
    nyq = sfreq / 2.0
    
    # Apply bandpass filter if requested
    if l_freq is not None or h_freq is not None:
        # Determine filter type and frequencies
        if l_freq is not None and h_freq is not None:
            # Bandpass
            if l_freq >= h_freq:
                raise ValueError(f"Low freq ({l_freq}) must be less than high freq ({h_freq})")
            if h_freq > nyq:
                raise ValueError(f"High freq ({h_freq}) exceeds Nyquist frequency ({nyq})")
                
            filter_type = 'bandpass'
            freqs = [l_freq, h_freq]
            if verbose:
                print(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
                
        elif l_freq is not None:
            # High-pass only
            if l_freq > nyq:
                raise ValueError(f"Low freq ({l_freq}) exceeds Nyquist frequency ({nyq})")
                
            filter_type = 'highpass'
            freqs = l_freq
            if verbose:
                print(f"Applying high-pass filter: {l_freq} Hz")
                
        else:
            # Low-pass only
            if h_freq > nyq:
                raise ValueError(f"High freq ({h_freq}) exceeds Nyquist frequency ({nyq})")
                
            filter_type = 'lowpass'
            freqs = h_freq
            if verbose:
                print(f"Applying low-pass filter: {h_freq} Hz")
        
        # Design the filter
        # Use 4th order Butterworth (good balance of steep rolloff and phase response)
        try:
            sos = signal.butter(N=4, Wn=freqs, btype=filter_type, fs=sfreq, output='sos')
            
            # Apply filter to each channel
            for ch_idx in range(filtered_data.shape[0]):
                # Use sosfiltfilt for zero-phase filtering (forward-backward)
                filtered_data[ch_idx, :] = signal.sosfiltfilt(sos, filtered_data[ch_idx, :])
                
        except Exception as e:
            if verbose:
                print(f"Warning: Bandpass filter failed with error: {e}")
                print("Data will be returned without bandpass filtering")
    
    # Apply notch filter if requested
    if notch_freq is not None:
        if notch_freq > nyq:
            if verbose:
                print(f"Warning: Notch freq ({notch_freq}) exceeds Nyquist ({nyq}), skipping notch filter")
        else:
            try:
                # Design notch filter
                # Q factor determines the width of the notch
                Q = 30  # Higher Q = narrower notch
                
                # Design the notch filter
                b_notch, a_notch = signal.iirnotch(notch_freq, Q, sfreq)
                
                if verbose:
                    print(f"Applying notch filter at {notch_freq} Hz")
                
                # Apply to each channel
                for ch_idx in range(filtered_data.shape[0]):
                    filtered_data[ch_idx, :] = signal.filtfilt(b_notch, a_notch, filtered_data[ch_idx, :])
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Notch filter failed with error: {e}")
                    print("Data will be returned without notch filtering")
    
    return filtered_data


def create_filter_params(sfreq: float, l_freq: float, h_freq: float) -> dict:
    """
    Helper function to create filter parameters.
    
    Parameters
    ----------
    sfreq : float
        Sampling frequency
    l_freq : float
        Low cutoff frequency
    h_freq : float
        High cutoff frequency
        
    Returns
    -------
    dict
        Filter parameters dictionary
    """
    nyq = sfreq / 2.0
    
    # Validate frequencies
    if l_freq >= h_freq:
        raise ValueError("Low frequency must be less than high frequency")
    if h_freq > nyq:
        raise ValueError(f"High frequency {h_freq} exceeds Nyquist frequency {nyq}")
    if l_freq < 0:
        raise ValueError("Low frequency must be positive")
    
    return {
        'sfreq': sfreq,
        'l_freq': l_freq,
        'h_freq': h_freq,
        'nyquist': nyq,
        'filter_length': int(3.3 * sfreq / l_freq)  # Approximate filter length
    }


def estimate_filter_length(sfreq: float, l_freq: float) -> int:
    """
    Estimate the filter length needed for a given frequency.
    
    This follows MNE's convention of using approximately 3.3 * sfreq / l_freq
    
    Parameters
    ----------
    sfreq : float
        Sampling frequency
    l_freq : float
        Low cutoff frequency
        
    Returns
    -------
    int
        Estimated filter length in samples
    """
    if l_freq <= 0:
        raise ValueError("Frequency must be positive")
        
    return int(3.3 * sfreq / l_freq)


# Test the filter if run directly
if __name__ == "__main__":
    # Create test data
    sfreq = 250  # 250 Hz sampling rate
    duration = 3  # 3 seconds
    n_samples = int(sfreq * duration)
    n_channels = 14
    
    # Generate test signal with multiple frequency components
    t = np.linspace(0, duration, n_samples)
    test_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Mix of different frequencies
        test_data[ch, :] = (
            0.5 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz (alpha)
            0.3 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz (beta)
            0.2 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz (power line)
            0.1 * np.random.randn(n_samples)     # noise
        )
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Sampling rate: {sfreq} Hz")
    print(f"Duration: {duration} seconds")
    
    # Apply filters
    filtered = filter_eeg_like_mne(
        data=test_data,
        sfreq=sfreq,
        l_freq=1.0,
        h_freq=40.0,
        notch_freq=60.0,
        verbose=True
    )
    
    print(f"Filtered data shape: {filtered.shape}")
    print("Filter test completed successfully!")