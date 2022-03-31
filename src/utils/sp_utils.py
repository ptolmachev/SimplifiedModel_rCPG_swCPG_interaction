from copy import deepcopy
import sympy
from scipy.optimize import minimize
from scipy.signal import savgol_filter as sg
from sklearn.ensemble import IsolationForest as IF
# from num_experiments.params_gen import *
from scipy import signal
from scipy.signal import butter, periodogram, find_peaks, hilbert
import ruptures as rpt
import peakutils
import numpy as np

def scale(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def butter_lowpass_filter(data, lowcut, fs, order=2):
    def butter_lowpass(lowcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='lowpass')
        return b, a

    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    def butter_bandpass(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def get_psd(t, signal):
    fs = 1.0 / (t[2] - t[1])
    f, psd = periodogram(signal, fs)
    return f, psd

def get_cutoff_freqz(f, psd, width):
    threshold = np.quantile(psd, 0.95)
    peaks, info = find_peaks(psd, threshold=threshold, prominence=1)
    # sort peaks by prominence
    prominences = info["prominences"]
    tmp = list(zip(peaks, prominences))
    tmp.sort(key=lambda a: a[1])
    peaks, prominences = list(zip(*tmp))
    highest_peak = peaks[-1]

    f_low = np.maximum(0, f[highest_peak] - width)
    f_high = f[highest_peak] + width
    return f_low, f_high

def extract_protophase(t, signal, stim_start_ind, filter=False):
    '''
    1) determines the frequency of oscillations
    2) applies bandpass filter around it
    3) applies Hilbert transform to filtered data to extract the protophase
    :return: protophase - array, whose values correspond to t
    '''
    if filter == True:
        f, psd = get_psd(t[:stim_start_ind], signal[:stim_start_ind])
        f_low, f_high = get_cutoff_freqz(f, psd, width=0.09)
        fs = 1.0 / (t[2] - t[1])
        if f_low != 0:
            signal_filtered = butter_bandpass_filter(signal, f_low, f_high, fs, order=2)
        else:
            signal_filtered = butter_lowpass_filter(signal, f_high, fs, order=2)
        an_signal = hilbert(signal_filtered)
    else:
        an_signal = hilbert(signal)

    offset = np.mean(an_signal[:stim_start_ind])
    protophase = np.unwrap(np.angle(an_signal - offset))
    return protophase

def fit_sigma(points, y, order):

    def fourier_sum(c, x, order):
        res = c[0] * np.ones_like(x)
        for i in range(order):
            res += c[1 + i] * np.cos((i + 1) * x) + c[1 + i + order] * np.sin((i + 1) * x)
        return res

    def func_to_minimise(c, x, y, order):
        return np.sum((fourier_sum(c, x, order) - y) ** 2)

    def constr_fun(c):
        return c[0] - 1 / (2 * np.pi)

    cons = {'type': 'eq', 'fun': constr_fun}
    res = minimize(func_to_minimise, x0=np.random.rand(2 * order + 1), args=(points, y, order), constraints=cons)
    if res.success == False:
        print("The fit wasn't successful")
        return None
    else:
        return res.x

def extract_phase(protophase, stim_start_ind, n_bins=200, order = 30):
    res = np.histogram(protophase[:stim_start_ind] % (2 * np.pi), bins=n_bins, range=[0, 2 * np.pi], density=True)
    points = (res[1] - 2 * np.pi / (n_bins * 2))[1:]
    y = res[0]
    coeff = fit_sigma(points, y, order)
    z = sympy.Symbol('z')
    expr = coeff[0] * 2 * np.pi
    for i in range(order):
        expr += (coeff[i + 1] * sympy.cos((i + 1) * z) + coeff[i + 1 + order] * sympy.sin(
            (i + 1) * z)) * 2 * np.pi
    integrated_sigma = sympy.lambdify(z, sympy.integrate(expr, (z, 0, z)), 'numpy')
    return integrated_sigma(protophase)

def get_period(t, signal):
    protophase = extract_protophase(t, signal, -1, filter=False)
    phase = extract_phase(protophase, -1, n_bins=200, order = 30)

    def line(x, t, y):
        omega, c = x
        return np.sum((omega * t + c - y) ** 2)

    omega, c = minimize(line, x0=np.random.rand(2), args=(t, phase)).x
    T = 2 * np.pi / omega
    return T

def binarise_signal(signal, threshold):
    res = np.zeros_like(signal)
    res[np.where(signal > threshold)] = 1.0
    res[np.where(signal <= threshold)] = 0
    return res

def find_relevant_peaks(signal, threshold, min_dist):
    # peaks = scipy.signal.find_peaks(signal)[0]
    # peaks_filtered = peaks_above_threshold[np.where(np.diff(np.append(-np.inf, peaks_above_threshold)) > min_dist)]
    peaks = peakutils.indexes(signal, thres_abs=threshold, min_dist=min_dist)
    peaks_above_threshold = np.array([peaks[i] for i in range(len(peaks)) if abs(signal[peaks[i]]) > threshold])
    return peaks_above_threshold

def get_insp_starts_and_ends(signal):
    signal_filtered = sg(signal, 121, 1)
    threshold = 0.4
    signal_binary = binarise_signal(signal_filtered, threshold)
    signal_change = np.diff(signal_binary)
    starts_inds = find_relevant_peaks(signal_change, 0.5, min_dist = 100)
    ends_inds = find_relevant_peaks(-signal_change, 0.5, min_dist = 100)
    return starts_inds, ends_inds