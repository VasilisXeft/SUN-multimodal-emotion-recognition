from datetime import datetime as dt

import heartpy as hp
import numpy as np
from scipy.signal import butter, filtfilt, welch


def safe_compute(func, *args, fallback=None):
    try:
        return func(*args)
    except Exception:
        if fallback is not None:
            return fallback
        # If we expect a tuple, we should return a tuple of appropriate length with zero values
        return 0  # Assuming the function normally returns 3 values


def calculate_pulse_amplitude(ppg, peaks):
    amplitudes = ppg[peaks]
    mean_amplitude = np.mean(amplitudes)
    std_amplitude = np.std(amplitudes)
    return mean_amplitude, std_amplitude


def sample_entropy(time_series, m=2, r=None):
    """Compute Sample Entropy of a time series."""
    if r is None:
        r = 0.2 * np.std(time_series)
    N = len(time_series)

    def _count(m):
        count = 0
        for i in range(N - m):
            if np.all(np.abs(time_series[i:i + m] - time_series[i + 1:i + 1 + m]) <= r):
                count += 1
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return np.inf
    return -np.log(A / B) if A != 0 else np.inf


def frequency_domain_hrv(ibi, fs):
    # Resample IBIs to a uniform time series
    # This requires interpolation; for simplicity, use cubic spline
    from scipy.interpolate import interp1d

    time_ibis = np.cumsum(ibi)
    f = interp1d(time_ibis, ibi, kind='cubic', fill_value="extrapolate")
    uniform_time = np.arange(0, time_ibis[-1], 1 / fs)
    ibi_uniform = f(uniform_time)

    # Detrend
    ibi_uniform = ibi_uniform - np.mean(ibi_uniform)

    # Compute power spectral density
    freqs, power = welch(ibi_uniform, fs=fs, nperseg=256)

    # Define frequency bands
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.40)

    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])

    lf_power = np.trapezoid(power[lf_mask], freqs[lf_mask])
    hf_power = np.trapezoid(power[hf_mask], freqs[hf_mask])

    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    return lf_power, hf_power, lf_hf_ratio


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y


def detect_peaks(signal, fs):
    peaks = []
    threshold = np.mean(signal) + 0.5 * np.std(signal)
    min_distance = int(0.5 * fs)  # Minimum distance between peaks (0.5 sec)

    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if len(peaks) == 0 or (i - peaks[-1]) > min_distance:
                peaks.append(i)
    return np.array(peaks)


def calculate_hr(peaks, fs):
    # Convert peak indices to time
    peak_times = peaks / fs
    # Calculate intervals between peaks (in seconds)
    ibi = np.diff(peak_times)
    # Calculate BPM
    bpm = 60 / ibi
    return bpm, ibi


def calculate_hrv_metrics(ibi):
    sdnn = np.std(ibi) * 1000  # in ms
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000  # in ms
    nn50 = np.sum(np.abs(np.diff(ibi)) > 0.05)  # more than 50 ms
    pnn50 = (nn50 / len(ibi)) * 100
    return sdnn, rmssd, pnn50


def estimate_respiration_rate(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_spectrum = np.abs(np.fft.rfft(signal))

    # Typically, respiration rate is between 0.1 Hz to 0.5 Hz (6 to 30 BPM)
    resp_band = (freqs >= 0.1) & (freqs <= 0.5)
    freqs_resp = freqs[resp_band]
    spectrum_resp = fft_spectrum[resp_band]

    dominant_freq = freqs_resp[np.argmax(spectrum_resp)]
    respiration_rate_bpm = dominant_freq * 60
    return respiration_rate_bpm


def calculate_ac_dc(signal, peaks):
    ac = np.mean(signal[peaks] - np.mean(signal))
    dc = np.mean(signal)
    return ac, dc


def compile_features_ppg(ppg, lowcut, highcut, fs):
    # Filter signal
    filtered_ppg = safe_compute(bandpass_filter, ppg, lowcut, highcut, fs)

    # Detect peaks
    peaks = safe_compute(detect_peaks, filtered_ppg, fs)

    # Compute heart rate (BPM) and inter-beat intervals (IBI)
    bpm, ibi = safe_compute(calculate_hr, peaks, fs)
    average_hr = safe_compute(np.mean, bpm)

    # Compute heart rate variability (HRV) metrics
    sdnn, rmssd, pnn50 = safe_compute(calculate_hrv_metrics, ibi, fallback=(0, 0, 0))

    # Compute frequency-domain HRV metrics
    lf_power, hf_power, lf_hf_ratio = safe_compute(frequency_domain_hrv, ibi, fs, fallback=(0, 0, 0))

    # Compute sample entropy
    sampen = safe_compute(sample_entropy, ibi)

    # Compute pulse amplitude metrics
    mean_amp, std_amp = safe_compute(calculate_pulse_amplitude, filtered_ppg, peaks, fallback=(0, 0))

    # Estimate respiration rate
    respiration_rate = safe_compute(estimate_respiration_rate, filtered_ppg, fs)

    # Collect all features in a dictionary
    features = {
        'Average HR (BPM)': average_hr if average_hr != 0 else 0,
        'SDNN (ms)': sdnn if sdnn != 0 else 0,
        'RMSSD (ms)': rmssd if rmssd != 0 else 0,
        'pNN50 (%)': pnn50 if pnn50 != 0 else 0,
        'LF Power': lf_power if lf_power != 0 else 0,
        'HF Power': hf_power if hf_power != 0 else 0,
        'LF/HF Ratio': lf_hf_ratio if lf_hf_ratio != 0 else 0,
        'Sample Entropy': sampen if sampen != 0 else 0,
        'Mean Pulse Amplitude': mean_amp if mean_amp != 0 else 0,
        'Std Pulse Amplitude': std_amp if std_amp != 0 else 0,
        'Respiration Rate (BPM)': respiration_rate if respiration_rate != 0 else 0
    }

    return features


def process_ppg(signal_ppg, tmsp_ppg):
    ppg = []
    for i in range(len(tmsp_ppg)):
        ppg.append(np.mean(signal_ppg[i][:]))

    ppg = np.array(ppg)

    # Define filter parameters
    timer = []

    for i in range(len(tmsp_ppg)):
        timer.append(dt.strftime(dt.utcfromtimestamp(tmsp_ppg[i]), '%Y-%m-%d %H:%M:%S.%f'))

    sample_rate = hp.get_samplerate_datetime(timer, timeformat='%Y-%m-%d %H:%M:%S.%f')

    fs = sample_rate  # Sampling frequency in Hz (modify based on your data)
    lowcut = 0.5  # Hz
    highcut = 3.0  # Hz

    features = compile_features_ppg(ppg, lowcut, highcut, fs)

    return features
