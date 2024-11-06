from datetime import datetime as dt

import heartpy as hp
import numpy as np
import scipy.signal as signal
from scipy import stats


def safe_compute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return 0


def add_suffix(original_dict, suffix):
    # Create a new dictionary with modified keys
    modified_dict = {key + suffix: value for key, value in original_dict.items()}

    return modified_dict


# Step 1: Apply low-pass filter to remove high-frequency noise
def low_pass_filter(eda_signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_eda = signal.filtfilt(b, a, eda_signal)
    return filtered_eda


# Step 2: Decompose into tonic (SCL) and phasic (SCR) components
def decompose_eda(eda_signal, fs):
    window_length = int(fs * 1)
    # Simple decomposition: Assume tonic is slow component, phasic is fast
    tonic = signal.savgol_filter(eda_signal, window_length=window_length, polyorder=3)
    phasic = eda_signal - tonic
    return tonic, phasic


# Hurst Exponent
def hurst_exponent(eda_signal):
    lags = range(2, 100)
    tau = [np.std(np.subtract(eda_signal[lag:], eda_signal[:-lag])) for lag in lags]
    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst


# Fractal Dimension using Higuchi's method
def higuchi_fractal_dimension(eda_signal, kmax=10):
    N = len(eda_signal)
    L = []
    x = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            Lmk = sum(
                abs(eda_signal[m + i * k] - eda_signal[m + (i - 1) * k]) for i in range(1, int(np.floor((N - m) / k)))) * (
                              N - 1) / (np.floor((N - m) / k) * k)
            Lk.append(Lmk)
        L.append(np.mean(Lk))
        x.append(np.log(1 / k))
    FD = np.polyfit(x, np.log(L), 1)[0]
    return FD


# Function to extract basic statistics and features from a signal segment
def extract_features(eda_signal, fs):
    features = {}

    # Time-domain features
    features['mean'] = safe_compute(np.mean, eda_signal)
    features['std'] = safe_compute(np.std, eda_signal)
    features['var'] = safe_compute(np.var, eda_signal)
    features['min'] = safe_compute(np.min, eda_signal)
    features['max'] = safe_compute(np.max, eda_signal)
    features['skewness'] = safe_compute(stats.skew, eda_signal)
    features['kurtosis'] = safe_compute(stats.kurtosis, eda_signal)

    # SCR specific: Peak Detection and Amplitude
    peaks, _ = safe_compute(signal.find_peaks, eda_signal, height=0.01)
    if isinstance(peaks, np.ndarray) and len(peaks) > 0:
        features['num_peaks'] = len(peaks)
        features['mean_peak_amplitude'] = safe_compute(np.mean, eda_signal[peaks])
        features['peak_variance'] = safe_compute(np.var, eda_signal[peaks])
    else:
        features['num_peaks'] = 0
        features['mean_peak_amplitude'] = 0
        features['peak_variance'] = 0

    # Frequency-domain features using Power Spectral Density (PSD)
    freqs, psd = safe_compute(signal.welch, eda_signal, fs)
    features['psd_mean'] = safe_compute(np.mean, psd) if isinstance(psd, np.ndarray) else 0
    features['psd_max'] = safe_compute(np.max, psd) if isinstance(psd, np.ndarray) else 0
    features['psd_sum'] = safe_compute(np.sum, psd) if isinstance(psd, np.ndarray) else 0

    # Entropy (Shannon Entropy)
    probability_distribution = safe_compute(np.histogram, eda_signal, bins=10, density=True)
    features['entropy'] = safe_compute(stats.entropy, probability_distribution[0]) if isinstance(probability_distribution, tuple) else 0

    return features


def compile_feature_eda(eda_signal, fs):
    filtered_eda = low_pass_filter(eda_signal, cutoff=0.5, fs=fs)

    tonic, phasic = decompose_eda(filtered_eda, fs)

    # Extract features for tonic and phasic components
    tonic_features = extract_features(tonic, fs)
    tonic_features = add_suffix(tonic_features, '_tonic')
    phasic_features = extract_features(phasic, fs)
    phasic_features = add_suffix(phasic_features, '_phasic')

    # Combine tonic and phasic features
    features = {**tonic_features, **phasic_features, 'hurst': hurst_exponent(eda_signal),
                'fractal_dimension': higuchi_fractal_dimension(eda_signal)}

    return features


def process_eda(signal_eda, tmsp_eda):
    eda = []
    for i in range(len(tmsp_eda)):
        eda.append(np.mean(signal_eda[i][:]))

    eda = np.array(eda)

    # Define filter parameters
    timer = []

    for i in range(len(tmsp_eda)):
        timer.append(dt.strftime(dt.utcfromtimestamp(tmsp_eda[i]), '%Y-%m-%d %H:%M:%S.%f'))

    sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')

    fs = sample_rate  # Sampling frequency in Hz (modify based on your data)

    features = compile_feature_eda(eda, fs)

    return features
