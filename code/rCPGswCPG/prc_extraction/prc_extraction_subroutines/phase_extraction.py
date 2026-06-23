import pickle
import numpy as np
import sympy
from matplotlib import pyplot as plt
from rCPGswCPG.utils.gen_utils import get_project_root
from rCPGswCPG.prc_extraction.prc_extraction_subroutines.filtering_utils import *
from scipy.optimize import minimize
'''
given the time series, extract the phase of oscillations
'''
def extract_protophase(t, signal, stim_start_ind, filter=True, psd_peak_width = 0.2, prominence_thr=0.92):
    '''
    1) determines the frequency of oscillations
    2) applies bandpass filter around it
    3) applies Hilbert transform to filtered data to extract the protophase
    :return: protophase - array, whose values correspond to t
    '''
    if filter == True:
        f, psd = get_psd(t[:stim_start_ind], signal[:stim_start_ind])
        f_low, f_high = get_cutoff_freqz(f, psd, width=psd_peak_width, prominence_thr=prominence_thr)
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


def get_period(Model, t_stop):
    Model.run(T=t_stop)
    states = np.array(Model.state_history)
    t = np.array(Model.t_range)
    protophase = extract_protophase(t, states[:, 0], -1, filter=True)
    phase = extract_phase(protophase, -1, n_bins=200, order=25)

    def line(x, t, y):
        omega, c = x
        return np.sum((omega * t + c - y) ** 2)

    omega, c = minimize(line, x0=np.random.rand(2), args=(t, phase)).x
    T = 2 * np.pi / omega
    return T

def get_period_(t, phase):
    def line(x, t, y):
        omega, c = x
        return np.sum((omega * t + c - y) ** 2)

    omega, c = minimize(line, x0=np.random.rand(2), args=(t, phase)).x
    T = 2 * np.pi / omega
    return T

if __name__ == '__main__':
    root_folder = get_project_root()
    data_file = f'{root_folder}/data/runs/7aa6c00f696045dcf9d045b5e94f07f0/vdp_data_1.pkl'
    data = pickle.load(open(data_file, 'rb+'))
    x = data["x"]
    t = data["t"]
    inp = data["inp"]
    signal = x[:, 0]
    stim_start_ind = int(np.where(np.diff(inp) > 0)[0][0])
    protophase = extract_protophase(t, signal, stim_start_ind, filter=False)
    phase = extract_phase(protophase, stim_start_ind, n_bins=200, order = 50)
    plt.plot(t, protophase)
    plt.plot(t, phase)
    plt.show()




