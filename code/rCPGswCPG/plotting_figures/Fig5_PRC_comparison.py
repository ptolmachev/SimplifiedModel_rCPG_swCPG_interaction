from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from scipy.interpolate import interp1d
import re
from scipy.optimize import minimize
from rCPGswCPG.utils.gen_utils import get_project_root
import matplotlib as mpl
mpl.use('MacOSX')  # on macOS built-in backend
# mpl.use('QtAgg')  # if you have PyQt5/PySide6 installed
from matplotlib import pyplot as plt


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

def fit_by_fourier(x, y, order, lam=1e3):
    x = np.asarray(x)
    y = np.asarray(y)
    w = (np.arange(1, order + 1) ** 4)

    def fsum(c):
        r = c[0] + np.zeros_like(x)
        for k in range(1, order + 1):
            r += c[k] * np.cos(k * x) + c[order + k] * np.sin(k * x)
        return r

    def obj(c):
        r = fsum(c) - y
        pen = lam * np.sum(w * (c[1:order + 1] ** 2 + c[order + 1:] ** 2))
        return np.dot(r, r) + pen

    c0 = np.zeros(2 * order + 1)
    res = minimize(obj, c0, method='L-BFGS-B')
    return res.x, fsum(res.x)

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

def plot_prc(phi, d_phi, d_phi_fit, insp, mode, outfile=None):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    color = 'r' if mode == 'experiment' else 'b'
    plt.scatter(phi, d_phi, color=color, s=15, marker='x', alpha=0.3, label=f'PRC {mode}')
    plt.plot(phi, d_phi_fit, linewidth=2, linestyle='-', color='forestgreen', label=f'PRC {mode}, fit')
    plt.plot(phi, insp, linewidth=2, color='k', alpha=0.5, label=f"PNA {mode}")
    plt.plot(phi, np.zeros_like(phi), linewidth=2, color='k', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc=1, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    ax.set_ylim([-1.5 * np.pi - 0.15, 1.5 * np.pi])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    tcks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    lbls = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    ax.set_xticks(ticks=tcks, labels=lbls, fontsize=12)
    tcks = [-3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    lbls = [r"$-\frac{3\pi}{2}$", r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    ax.set_yticks(ticks=tcks, labels=lbls, fontsize=12)
    if outfile:
        plt.savefig(outfile, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    # Experimental PRC data
    method = 'prc_linear_fit'
    # replace prc_exp_folder with the correct path
    prc_project_root = os.path.abspath(r"/Users/tolmach/Documents/GitHub/Exp_Data_Processing_rCPG")
    sim_data_folder = os.path.abspath(r"/Users/tolmach/Documents/GitHub/rCPGswCPG/data")
    data_path = os.path.join(prc_project_root, "data")
    data_folder = os.path.join(data_path, "sln_prc_chunked")
    folders = get_folders(data_folder, "_prc")
    ind_datasets = [0, 1, 2, 3] # which data to take

    phi_exp = np.array([])
    d_phi_exp = np.array([])
    data_folder = os.path.join(data_path, "prc_data", method)
    folders = get_folders(data_folder, "_prc")
    for i, folder in enumerate(folders):
        if i in ind_datasets:
            # phase_shift = phase_shifts[i]
            load_from = os.path.join(data_path, "prc_data", method, folder)
            files = os.listdir(load_from)
            file = [file for file in files if file.endswith(".pkl") and file.startswith("data_")][0]
            data = pickle.load(open(os.path.join(load_from, file), 'rb+'))
            data_phi = np.array(data['phi']) % (2 * np.pi)
            phi_exp = np.concatenate([phi_exp, (data_phi)])
            d_phi_exp = np.concatenate([d_phi_exp, (data['delta_phi'])])
            PNA = data['PNA']
            phase_pna = np.array(data['phase']) % (2*np.pi)
            # phase_pna, PNA = sort_ascending(phase_pna, PNA)
            phi_exp, d_phi_exp = sort_ascending(phi_exp, d_phi_exp)

    coeffs, d_phi_exp_fit = fit_by_fourier(phi_exp, d_phi_exp, order=4)
    PNA = interp1d(phase_pna, PNA)(phi_exp[1:])
    # adding signal to a plot to have a comparison
    max_val = 1.5 * np.pi#np.max(d_phi_exp)
    min_val = -1.5 * np.pi#np.min(d_phi_exp)
    PNA_exp = scale(PNA) * (max_val - min_val) + min_val

    # Simlations PRC data
    load_from_filename = "PRC_model_complex.pkl"
    prc_data_sim = pickle.load(open(os.path.join(sim_data_folder, "processed_data", load_from_filename), "rb+"))
    phi_sim = prc_data_sim["phi"]
    d_phi_sim = prc_data_sim["delta_phi"]

    phi_sim, d_phi_sim = sort_ascending(phi_sim, d_phi_sim)
    PNA_sim = interp1d(np.linspace(0, 2 * np.pi, len(prc_data_sim["RampI"])), prc_data_sim["RampI"])(phi_sim)

    # adjusting the phases
    phase_shift_exp = 0.1
    phi_exp, d_phi_exp, PNA_exp = shift_cyclically(phi_exp, phase_shift_exp, d_phi_exp, PNA_exp)
    phase_shift_sim = 0.44
    phi_sim, d_phi_sim, PNA_sim = shift_cyclically(phi_sim, phase_shift_sim, d_phi_sim, PNA_sim)
    coeffs_exp, d_phi_exp_fit = fit_by_fourier(phi_exp, d_phi_exp, order=31, lam=0.1)
    coeffs_sim, d_phi_sim_fit = fit_by_fourier(phi_sim, d_phi_sim, order=31, lam=0.1)

    img_folder = os.path.join(get_project_root(), 'img')
    plot_prc(phi_exp, d_phi_exp, d_phi_exp_fit, PNA_exp, mode='experiment', outfile=os.path.join(img_folder, f"PRC_experiment.pdf"))
    plot_prc(phi_sim, d_phi_sim, d_phi_sim_fit, PNA_sim, mode='simulation', outfile=os.path.join(img_folder, f"PRC_simulation.pdf"))
