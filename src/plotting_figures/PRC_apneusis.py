from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from scipy.interpolate import interp1d
import re
from scipy.optimize import minimize
from src.utils.gen_utils import get_project_root

def get_folders(root_folder, pattern):
    folders_all = os.listdir(root_folder)
    folders = []
    for i, folder in enumerate(folders_all):
        m = re.search(pattern, str(folder))
        if m is not None:
            folders.append(folder)
    return folders

def scale(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s))

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

def sort_ascending(x, y):
    tmp = list(zip(x, y))
    tmp.sort(key=lambda a: a[0])
    x, y = zip(*tmp)
    x = np.array(x)
    y = np.array(y)
    return x, y

def shift_cyclically(phase, phase_shift, x, y):
    tmp_phase = (phase + phase_shift) % (2 * np.pi)
    tmp = list(zip(tmp_phase, x, y))
    tmp.sort(key=lambda a: a[0])
    new_phase, x, y = zip(*tmp)
    new_phase = np.array(new_phase)
    x = np.array(x)
    y = np.array(y)
    return new_phase, x, y

# Simlations PRC data
sim_data_folder = os.path.abspath(r"C:\Users\betad\Projects\PRC_estimation\data")
load_from_filename = "PRC_model_complex_KF_inh.pkl"
prc_data_sim = pickle.load(open(os.path.join(sim_data_folder, "processed_data", load_from_filename), "rb+"))
phi_sim = prc_data_sim["phi"]
d_phi_sim = prc_data_sim["delta_phi"]

phi_sim, d_phi_sim = sort_ascending(phi_sim, d_phi_sim)
Insp = interp1d(np.linspace(0, 2 * np.pi, len(prc_data_sim["Insp"])),prc_data_sim["Insp"])(phi_sim)

phase_shift_sim = 1.0
phi_sim, d_phi_sim, Insp = shift_cyclically(phi_sim, phase_shift_sim, d_phi_sim, Insp)

fig = plt.figure(figsize = (15, 10))
# plotting experimental data

# plotting simulation data
plt.scatter(phi_sim, d_phi_sim, color='b', s = 10, marker = 'o', alpha = 0.5, label = 'PRC sim data')
coeffs, d_phi_sim_fit = fit_by_fourier(phi_sim, d_phi_sim, order=7)
plt.plot(phi_sim, d_phi_sim_fit, linewidth = 2, linestyle = '-', color='b', label = 'PRC sim data fit')

plt.plot(phi_sim, Insp, color = 'k', alpha = 0.3)
plt.legend(fontsize = 16, loc = 1)
# plt.ylim([-2.0 * np.pi - 0.15, 2.0 * np.pi])
# plt.arrow(-0.25, -1.5 * np.pi-0.1, 0, 3*np.pi - 0.25, head_width = 0.05, head_length = 0.2, fc ='k')
# plt.arrow(-0.25, -1.5 * np.pi-0.1, 2*np.pi + 0.3, 0, head_width = 0.1, head_length = 0.1, fc ='k')
# plt.arrow(0, 2*np.pi)
# plt.box(False)
img_folder = os.path.join(get_project_root(), "img")
plt.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(img_folder, "PRC_apneusis.pdf"))
plt.savefig(os.path.join(img_folder, "PRC_apneusis.svg"))
plt.show()