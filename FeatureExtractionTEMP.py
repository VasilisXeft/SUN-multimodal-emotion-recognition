import numpy as np
import scipy.signal as signal
from scipy import stats
from scipy.fftpack import fft
from datetime import datetime as dt

import heartpy as hp


def safe_compute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return 0


def add_suffix(original_dict, suffix):
    # Create a new dictionary with modified keys
    modified_dict = {key + suffix: value for key, value in original_dict.items()}

    return modified_dict


# Step 1: Remove outliers (optional)
def remove_outliers(temp, threshold=3):
    mean = np.mean(temp)
    std = np.std(temp)
    return np.where(abs(temp - mean) > threshold * std, mean, temp)


# Step 2: Apply a smoothing filter (optional)
def smooth_signal(temp, window_length=51, polyorder=3):
    return signal.savgol_filter(temp, window_length, polyorder)


# Function to extract basic statistics and time-domain features
def extract_temperature_features(temperature_data):
    features = {}

    # Basic statistical features
    features['mean'] = safe_compute(np.mean, temperature_data)
    features['std'] = safe_compute(np.std, temperature_data)
    features['min'] = safe_compute(np.min, temperature_data)
    features['max'] = safe_compute(np.max, temperature_data)
    features['range'] = safe_compute(np.ptp, temperature_data)
    features['median'] = safe_compute(np.median, temperature_data)
    features['skewness'] = safe_compute(stats.skew, temperature_data)
    features['kurtosis'] = safe_compute(stats.kurtosis, temperature_data)

    # Rate of change (temperature derivative)
    temperature_derivative = safe_compute(np.diff, temperature_data)
    features['mean_derivative'] = safe_compute(np.mean, temperature_derivative)
    features['std_derivative'] = safe_compute(np.std, temperature_derivative)

    # Peak detection (identify significant changes)
    peaks, _ = safe_compute(signal.find_peaks, temperature_data, height=np.mean(temperature_data) + 0.5)
    features['num_peaks'] = len(peaks) if isinstance(peaks, np.ndarray) else 0
    features['mean_peak_value'] = safe_compute(np.mean, temperature_data[peaks]) if len(peaks) > 0 else 0

    return features


# Function to extract frequency-domain features
def extract_frequency_features(temperature_data, sampling_rate):
    features = {}

    N = len(temperature_data)
    temperature_fft = safe_compute(fft, temperature_data)
    freqs = np.fft.fftfreq(N, d=1 / sampling_rate)

    positive_freqs = freqs[:N // 2]
    positive_magnitude = safe_compute(np.abs, temperature_fft[:N // 2])

    psd = safe_compute(np.square, positive_magnitude)
    features['mean_psd'] = safe_compute(np.mean, psd)
    features['max_psd'] = safe_compute(np.max, psd)
    features['dominant_frequency'] = positive_freqs[np.argmax(psd)] if len(psd) > 0 else 0

    return features


# Function to extract nonlinear features like entropy
def extract_nonlinear_features(temperature_data):
    features = {}

    # Shannon Entropy
    probability_distribution = np.histogram(temperature_data, bins=10, density=True)[0]
    features['entropy'] = safe_compute(stats.entropy, probability_distribution)

    # Hurst Exponent
    def hurst_exponent(temp):
        lags = range(2, 100)
        tau = [np.std(np.subtract(temp[lag:], temp[:-lag])) for lag in lags]
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
        return hurst

    features['hurst'] = safe_compute(hurst_exponent, temperature_data)

    return features


# Combine all features
def extract_all_temperature_features(temperature_data, sampling_rate):
    features = {}

    # Time-domain features
    time_domain_features = extract_temperature_features(temperature_data)

    # Frequency-domain features
    frequency_domain_features = extract_frequency_features(temperature_data, sampling_rate)

    # Nonlinear features
    nonlinear_features = extract_nonlinear_features(temperature_data)

    # Combine all features
    features.update(time_domain_features)
    features.update(frequency_domain_features)
    features.update(nonlinear_features)

    return features


def process_temp(signal_temp, tmsp_temp):
    temp = []
    for i in range(len(tmsp_temp)):
        temp.append(np.mean(signal_temp[i][:]))

    temp = np.array(temp)

    # Define filter parameters
    timer = []

    for i in range(len(tmsp_temp)):
        timer.append(dt.strftime(dt.utcfromtimestamp(tmsp_temp[i]), '%Y-%m-%d %H:%M:%S.%f'))

    sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')

    fs = sample_rate  # Sampling frequency in Hz (modify based on your data)

    features = extract_all_temperature_features(temp, fs)

    features = add_suffix(features, '_therm')

    return features
