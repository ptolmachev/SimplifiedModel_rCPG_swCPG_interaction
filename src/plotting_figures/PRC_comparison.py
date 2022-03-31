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

# Experimental PRC data
method = 'prc_linear_fit'
prc_project_root = os.path.abspath(r"C:\Users\betad\Projects\Exp_Data_Processing_rCPG")
data_path = os.path.join(prc_project_root, "data")
img_path = os.path.join(prc_project_root, "img")
data_folder = os.path.join(data_path, "sln_prc_chunked")
folders = get_folders(data_folder, "_prc")
ind_datasets = [0,1,2,3]

phi_exp = np.array([])
d_phi_exp = np.array([])
data_folder = os.path.join(data_path, "prc_data", method)
folders = get_folders(data_folder, "_prc")
phase_shifts = [-0.75, -1.05, -0.6, -0.85]
for i, folder in enumerate(folders):
    if i in ind_datasets:
        phase_shift = phase_shifts[i]
        load_from = os.path.join(data_path, "prc_data", method, folder)
        data = pickle.load(open(os.path.join(load_from, "data.pkl"), 'rb+'))
        data_phi = (np.array(data['phi']) - phase_shift) % (2 * np.pi)
        phi_exp = np.concatenate([phi_exp, (data_phi)])
        d_phi_exp = np.concatenate([d_phi_exp, (data['delta_phi'])])
        PNA = data['PNA']
        phase_pna = (np.array(data['phase']) - phase_shift) % (2*np.pi)
        phase_pna, PNA = sort_ascending(phase_pna, PNA)
        phi_exp, d_phi_exp = sort_ascending(phi_exp, d_phi_exp)

coeffs, d_phi_exp_fit = fit_by_fourier(phi_exp, d_phi_exp, order=4)
PNA = interp1d(phase_pna, PNA)(phi_exp[1:])
# adding signal to a plot to have a comparison
max_val = 1.5*np.pi#np.max(d_phi_exp)
min_val = -1.5*np.pi#np.min(d_phi_exp)
PNA_scaled = scale(PNA) * (max_val - min_val) + min_val

# Simlations PRC data
sim_data_folder = os.path.abspath(r"C:\Users\betad\Projects\PRC_estimation\data")
load_from_filename = "PRC_model_complex.pkl"
prc_data_sim = pickle.load(open(os.path.join(sim_data_folder, "processed_data", load_from_filename), "rb+"))
phi_sim = prc_data_sim["phi"]
d_phi_sim = prc_data_sim["delta_phi"]

phi_sim, d_phi_sim = sort_ascending(phi_sim, d_phi_sim)
PNA_sim = interp1d(np.linspace(0, 2 * np.pi, len(prc_data_sim["RampI"])),prc_data_sim["RampI"])(phi_sim)

# adjusting the phases
phase_shift_exp = 0#0.88
phi_exp, d_phi_exp, PNA_scaled = shift_cyclically(phi_exp, phase_shift_exp, d_phi_exp, PNA_scaled)
phase_shift_sim = 0.38
phi_sim, d_phi_sim, PNA_sim = shift_cyclically(phi_sim, phase_shift_sim, d_phi_sim, PNA_sim)

fig = plt.figure(figsize = (7.5, 5))
# # plotting experimental data
# plt.scatter(phi_exp, d_phi_exp, color='r', s = 15, marker = 'x', alpha = 0.5, label = 'PRC experimental')
# coeffs, d_phi_exp_fit = fit_by_fourier(phi_exp, d_phi_exp, order=4)
# plt.plot(phi_exp, d_phi_exp_fit, linewidth = 2, linestyle = '-', color = 'r',label = 'PRC experimental, fit')
# plt.plot(phi_exp, PNA_scaled, linewidth = 2, color = 'k', alpha = 0.5, label = "PNA (for scale)")
# plt.plot(phi_exp, np.zeros_like(phi_exp), linewidth=2, color = 'k', linestyle = '--', alpha = 0.5)

# plotting simulation data
plt.scatter(phi_sim, d_phi_sim, color='b', s = 15, marker = 'o', alpha = 0.25, label = 'PRC simulated')
coeffs, d_phi_sim_fit = fit_by_fourier(phi_sim, d_phi_sim, order=50)
plt.plot(phi_sim, d_phi_sim_fit, linewidth = 2, linestyle = '-', color='b', label = 'PRC simulated, fit')
plt.plot(phi_sim, PNA_sim, linewidth=2, color = 'k', alpha = 0.5, label = 'simulated PNA (for scale)')
plt.plot(phi_sim, np.zeros_like(phi_sim), linewidth=2, color = 'k', linestyle = '--', alpha = 0.5)


plt.legend(fontsize = 16, loc = 1)
plt.ylim([-1.5 * np.pi - 0.15, 1.5 * np.pi])
plt.arrow(-0.25, -1.5 * np.pi-0.1, 0, 3*np.pi - 0.25, head_width = 0.05, head_length = 0.2, fc ='k')
plt.arrow(-0.25, -1.5 * np.pi-0.1, 2*np.pi + 0.3, 0, head_width = 0.1, head_length = 0.1, fc ='k')
# plt.arrow(0, 2*np.pi)
plt.box(False)
img_folder = os.path.join(get_project_root(), "img")
plt.tick_params(axis='both', which='major', labelsize=20)
tcks = [0, np.pi / 2, np.pi, 3*np.pi / 2, 2*np.pi]
lbls = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
plt.xticks(ticks=tcks, labels = lbls, fontsize = 20)

tcks = [-3*np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi ]
lbls = [r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
plt.yticks(ticks=tcks, labels = lbls, fontsize = 20)

plt.savefig(os.path.join(img_folder, "comparison_PRC.pdf"))
plt.savefig(os.path.join(img_folder, "comparison_PRC.svg"))
plt.show()