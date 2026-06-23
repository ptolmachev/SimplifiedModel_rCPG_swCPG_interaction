import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, periodogram, hilbert, savgol_filter
from scipy.optimize import minimize

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=2):
    def butter_lowpass(lowcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='lowpass')
        return b, a

    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_psd(t, signal):
    fs = 1.0 / (t[2] - t[1])
    f, psd = periodogram(signal, fs)
    return f, psd

def get_cutoff_freqz(f, psd, width, prominence_thr):
    threshold = np.quantile(psd, prominence_thr)
    peaks, info = find_peaks(psd, threshold=threshold, prominence=1)
    if len(peaks) == 0:
        return f[0], f[-1]
    # sort peaks by prominence
    prominences = info["prominences"]
    tmp = list(zip(peaks, prominences))
    tmp.sort(key=lambda a: a[1])
    peaks, prominences = list(zip(*tmp))
    highest_peak = peaks[-1]

    f_low = np.maximum(0, f[highest_peak] - width)
    f_high = f[highest_peak] + width
    return f_low, f_high

def fit_by_fourier(x, y, order):

    def fourier_sum(c, x, order):
        res = c[0] * np.ones_like(x)
        for i in range(order):
            res += c[1 + i] * np.cos((i + 1) * x) + c[1 + i + order] * np.sin((i + 1) * x)
        return res

    def func_to_minimise(c, x, y, order):
        return np.sum((fourier_sum(c, x, order) - y) ** 2)

    res = minimize(func_to_minimise, x0=np.random.rand(2 * order + 1), args=(x, y, order))
    coeffs = res.x
    signal = fourier_sum(coeffs, x, order)
    return coeffs, signal