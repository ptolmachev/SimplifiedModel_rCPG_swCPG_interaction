import json
import pickle
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
from rCPGswCPG.prc_extraction.prc_extraction_subroutines.phase_extraction import extract_protophase, extract_phase, get_period_
from rCPGswCPG.utils.gen_utils import get_project_root, get_files
import os
from matplotlib import pyplot as plt

def line(x, t, y):
    omega, c = x
    return np.sum((omega * t + c - y) ** 2)

def get_phase_shift(t, phase, stim_start_ind, stim_duration_ind, transient_inds = 0):
    # transient inds is the number of data points to discard after the stimulus (cause hilbert transform is continues)
    # and the change in the phase is reflected after some initial transients
    dt = t[2] - t[1]
    phase_b = phase[:stim_start_ind-transient_inds]
    t_b = np.arange(len(phase_b)) * dt
    phase_a = phase[stim_start_ind + stim_duration_ind + transient_inds:]
    t_a = np.arange(len(phase))[stim_start_ind + stim_duration_ind + transient_inds:] * dt
    omega, c = minimize(line, x0=np.random.rand(2), args=(t_b, phase_b)).x
    def constr_fun(x):
        return x[0] - omega
    omega, b = minimize(line, x0=np.random.rand(2), args=(t_a, phase_a), constraints={'type': 'eq', 'fun': constr_fun}).x
    Delta_Phi = (b - c)
    # we cant't trust the real phase, so we use a linear approximation of it
    Phi = (omega * stim_start_ind * dt + c) % (2 * np.pi)
    return Phi, Delta_Phi

def run_prc_extraction_direct(data_folder, files, save_to, filter=True):
    data_phi = []
    data_delta_phi = []
    for i, file in tqdm(enumerate(files)):
        data = pickle.load(open(os.path.join(data_folder, file), "rb+"))
        signal = data["signal"]
        t = data["t"]
        inp = data["inp"]
        stim_start_ind = int(np.where(np.diff(inp) > 0)[0][0]) + 1
        stim_duration_ind = (int(np.where(np.diff(inp) < 0)[0][0]) - int(np.where(np.diff(inp) > 0)[0][0]))
        try: # if there is some error we just through out this point
            protophase = extract_protophase(t, signal, stim_start_ind, filter=filter, psd_peak_width=0.3, prominence_thr=0.92)
            phase = extract_phase(protophase, stim_start_ind, n_bins=100, order=30)
            phi, delta_phi = get_phase_shift(t, phase, stim_start_ind, stim_duration_ind, transient_inds=500)
            data_phi.append(deepcopy(phi))
            data_delta_phi.append(deepcopy(delta_phi))
        except:
            pass

    #get the signal starting at phi = 0, ending up at phi = 2 pi:
    inds_zero_phase = (np.where(np.abs(np.diff(phase % (2 * np.pi))) > np.pi)[0])
    ind1 = inds_zero_phase[0] + 1
    ind2 = inds_zero_phase[1]

    prc_data = dict()
    prc_data['signal'] = signal[ind1: ind2]
    prc_data['dt'] = data['dt']
    prc_data['phase'] = phase[ind1: ind2] % (2*np.pi)
    prc_data['phi'] = data_phi
    prc_data['delta_phi'] = data_delta_phi
    root_folder = get_project_root()

    pickle.dump(prc_data, open(os.path.join(root_folder, "data", "processed_data", save_to), "wb+"))
    return None
