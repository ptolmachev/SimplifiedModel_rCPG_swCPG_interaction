# the extraction of the PRC from the recording obtained on a simplified model (simplified neurons) of interaction
# between the recpiratory and the swallowing CPG.
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from rCPGswCPG.prc_extraction.prc_extraction_subroutines.prc_extraction_linear_fit import get_phase_shift
from rCPGswCPG.prc_extraction.prc_extraction_subroutines.filtering_utils import fit_by_fourier
from rCPGswCPG.prc_extraction.prc_extraction_subroutines.phase_extraction import extract_protophase, extract_phase
from rCPGswCPG.utils.gen_utils import get_files, get_project_root
import pickle
import os
import matplotlib as mpl
mpl.use('MacOSX')  # on macOS built-in backend

def scale(s):
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def run_PRC_estimation(data_folder, save_to_filename):
    data_phi = []
    data_delta_phi = []
    # list all the files which contain some specific pattern "run_" and then go through all of them
    files = get_files(data_folder, pattern="run_")
    for i, file in tqdm(enumerate(files), total=len(files), desc="Processing recordings for PRC estimation"):
        data = pickle.load(open(os.path.join(data_folder, file), "rb+"))
        signals = data["signals"]
        t = data["t"]
        dt = data["dt"]
        t_stim_start = data["t_stim_start"]
        stim_duration = data["stim_duration"]
        pnames = [p.name for p in data["population_names"]]
        print(pnames)
        # inp = signals[:, pnames.index("Sensory_relay")]
        # stim_start_ind = int(np.where((inp) > 0.5)[0][0]) + 1
        Insp = signals[:, pnames.index("Insp")]
        RampI = signals[:, pnames.index("RampI")]
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
        RampI_ = RampI[:ind_stim_start]
        phase_ = np.array(phase[:ind_stim_start]) % (2 * np.pi)
        # inds of the start of a new cycle
        inds = np.where(np.diff(phase_) < -np.pi)[0]
        # for the plotting purposes (to have an actual signal on the background)
        Insp_ = Insp_[inds[0]:inds[1]]
        RampI_ = RampI_[inds[0]:inds[1]]
        max_val = 1.5 * np.pi # np.max(prc_data['delta_phi'])
        min_val = -1.5 * np.pi # np.min(prc_data['delta_phi'])
        Insp_scaled = scale(Insp_) * (max_val - min_val) + min_val
        RampI_scaled = scale(RampI_) * (max_val - min_val) + min_val

        prc_data = dict()
        prc_data['phi'] = np.array(data_phi).squeeze()
        prc_data['delta_phi'] = np.array(data_delta_phi).squeeze()
        prc_data['Insp'] = Insp_scaled
        prc_data['RampI'] = RampI_scaled
        root_folder = get_project_root()
        os.makedirs(os.path.join(root_folder, "data", "processed_data"), exist_ok=True)
        # Saves the (phi - phase of the stimulation, delta_phi - phase shift) points as data as well as the scaled Insp and RampI signals for visualization
        pickle.dump(prc_data, open(os.path.join(root_folder, "data", "processed_data", save_to_filename), "wb+"))
    return None

if __name__ == '__main__':
    # first one needs to run the experiments in experiments/prc_running_simulations.py
    #RUNNING THE PRC EXTRACTION
    data_folder = os.path.join(get_project_root(), "data",
                             "experiments",
                             "PRC_experiments",
                             "model_complex",
                             "PRC_short_stim", # change the name here
                             "num_run_short_stim_0.45_0.25") # change the name here
    save_to_filename = "PRC_model_complex.pkl"
    run_PRC_estimation(data_folder, save_to_filename)
