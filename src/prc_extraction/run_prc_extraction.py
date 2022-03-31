# the extraction of the PRC from the recording obtained on a simplified model (simplified neurons) of interaction
# between the recpiratory and the swallowing CPG.
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from src.prc_extraction.prc_extraction_subroutines.prc_extraction_linear_fit import get_phase_shift
from src.prc_extraction.prc_extraction_subroutines.filtering_utils import fit_by_fourier
from src.prc_extraction.prc_extraction_subroutines.phase_extraction import extract_protophase, extract_phase
from src.utils.gen_utils import get_files, get_project_root, create_dir_if_not_exist
import pickle
import os

def scale(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def run_PRC_estimation(data_folder, save_to_filename):
    data_phi = []
    data_delta_phi = []
    # list all the files the files which contain some specific pattern "run_" and then go through all of them
    files = get_files(data_folder, pattern="run_")
    for i, file in tqdm(enumerate(files)):
        data = pickle.load(open(f'{data_folder}/{file}', "rb+"))
        signals = data["signals"]
        t = data["t"]
        dt = data["dt"]
        t_stim_start = data["t_stim_start"]
        stim_duration = data["stim_duration"]
        pnames = data["population_names"]
        # inp = signals[:, pnames.index("Sensory_relay")]
        # stim_start_ind = int(np.where((inp) > 0.5)[0][0]) + 1
        Insp = signals[:, pnames.index("Insp")]
        ind_stim_start = int(t_stim_start * 1000 / dt)
        stim_duration_ind = int(stim_duration * 1000 / dt)
        protophase = extract_protophase(t, Insp, ind_stim_start, filter=False)
        phase = extract_phase(protophase, ind_stim_start, n_bins=150, order=50)

        #transient inds is the number of data points to discard after the stimulus (cause hilbert transform is continues)
        # and the change in the phase is reflected after some initial transients
        phi, delta_phi = get_phase_shift(t, phase, ind_stim_start, stim_duration_ind, transient_inds=2000)
        # if the delta_phi is within the sane range:
        if np.abs(delta_phi) < 2 * np.pi:
            data_phi.append(deepcopy(phi))
            data_delta_phi.append(deepcopy(delta_phi))

        #_ - all the signals before the stim start
        Insp_ = Insp[:ind_stim_start]
        phase_ = np.array(phase[:ind_stim_start]) % (2 * np.pi)
        # inds of the start of a new cycle
        inds = np.where(np.diff(phase_) < -np.pi)[0]
        # for the plotting purposes (to have an actual signal on the background)
        Insp_ = Insp_[inds[0]:inds[1]]
        max_val = 1.5 * np.pi #np.max(prc_data['delta_phi'])
        min_val = -1.5 * np.pi#np.min(prc_data['delta_phi'])
        Insp_scaled = scale(Insp_) * (max_val - min_val) + min_val

        prc_data = dict()
        prc_data['phi'] = np.array(data_phi).squeeze()
        prc_data['delta_phi'] = np.array(data_delta_phi).squeeze()
        prc_data['Insp'] = Insp_scaled
        root_folder = get_project_root()
        create_dir_if_not_exist(os.path.join(root_folder, "data", "processed_data"))
        pickle.dump(prc_data, open(os.path.join(root_folder, "data", "processed_data", save_to_filename), "wb+"))
    return None

def plot_prc(prc_data, phase_shift):
    # phase shift to rotate the cycle so that PNA was at the beginning
    phi = prc_data['phi']
    delta_phi = prc_data['delta_phi']

    phi = (np.array(phi) - phase_shift) % (2 * np.pi)
    tmp = list(zip(phi, delta_phi))
    tmp.sort(key = lambda a: a[0])
    phi, delta_phi = zip(*tmp)
    phi = np.array(phi)

    delta_phi = np.array(delta_phi)
    Insp = prc_data['Insp']
    phase = (np.linspace(0, 2 * np.pi, len(Insp)) - phase_shift) % (2 * np.pi)
    tmp = list(zip(phase, Insp))
    tmp.sort(key = lambda a: a[0])
    phase, Insp = zip(*tmp)
    Insp = np.array(Insp)
    phase = np.array(phase)

    coeffs, Delta_Phi_fit = fit_by_fourier(phi, delta_phi, order=7)
    fig = plt.figure(figsize=(20,10))
    plt.scatter(phi, delta_phi, color = 'b', s = 10, marker = 'x', label = 'PRC')
    plt.plot(phi, Delta_Phi_fit, color='r', label = 'PRC Fourier fit')
    plt.plot(phase, Insp, color = 'k', linestyle = '--', label = 'Insp', linewidth = 3)
    plt.legend(fontsize=24)
    plt.xlabel("Phase",fontsize=24)
    plt.ylabel("Phase change",fontsize=24)
    # plt.ylim([-1.5 * np.pi, 1.5 * np.pi])
    plt.grid(True)
    plt.show(block=True)
    plt.close()

if __name__ == '__main__':
    # #RUNNING THE PRC EXTRACTION
    data_folder = os.path.join(get_project_root(), "data",
                             "experiments",
                             "PRC_experiments",
                             "model_complex",
                             "PRC_short_stim", # change the name here
                             "num_run_short_stim_0.45_0.25") # change the name here
    save_to_filename = "PRC_model_complex.pkl"
    run_PRC_estimation(data_folder, save_to_filename)

    # PLOTTING THE DATA
    root_folder = get_project_root()
    load_from_filename = "PRC_model_complex.pkl"
    prc_data = pickle.load(open(os.path.join(root_folder, "data", "processed_data", load_from_filename), "rb+"))
    phase_shift = -0.4
    plot_prc(prc_data, phase_shift)
