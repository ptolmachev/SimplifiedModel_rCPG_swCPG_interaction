from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

def get_index(T, dt):
    return int(T*1000 / dt)

def plot_VNA_responses(VNA, long_stim_start_ind, long_stim_stop_ind, short_stim_start_ind, short_stim_stop_ind):
    fig, ax = plt.subplots(1, 1, figsize = (8, 3))
    ax.plot(VNA, color='k', label='VNA')
    ax.axvline(long_stim_start_ind, color='r')
    ax.axvline(long_stim_stop_ind, color='r')
    ax.axvspan(long_stim_start_ind, long_stim_stop_ind, color='red', alpha=0.2)
    ax.axvline(short_stim_start_ind, color='r')
    ax.axvline(short_stim_stop_ind, color='r')
    ax.axvspan(short_stim_start_ind, short_stim_stop_ind, color='red', alpha=0.2)
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    outpath = os.path.join(f"..", "..", "img", f"{prefix}_{p}_{param}.pdf")
    plt.savefig(outpath, bbox_inches='tight', transparent=True)
    plt.show()

params2vals = {"W_Sw1_to_Sw2": [-1.18, -0.4],
               "W_Sw2_to_Sw1": [-0.87, -0.5],
               "Sw1_tau" : [535, 1821],
               "arousal": [0.09, 0.5]}
VNA_coeffs = {"KF_phasic" : 0.75, "Sw1" : 0.6, "RampI" : 0.9}
params = []
for p in params2vals.keys():
    print(p)
    prefix = 'Experiment' if p in ["arousal", "wakefulness", "KF_lesioning"]  else "Varying"
    data_folder = os.path.join(f"..", "..", "data", "experiments", f"{prefix}_{p}", "runs", "recordings")
    files = os.listdir(data_folder)
    files_filtered = []
    vals2plot = params2vals[p]
    vals = np.array([float(files.split("_")[2].split(".pkl")[0]) for files in files])
    for v2p in vals2plot:
        files_filtered.append(files[np.argmin(np.abs(vals - v2p))])
    print(files_filtered)
    for file in files_filtered:
        data = pickle.load(open(os.path.join(data_folder, file), "rb"))
        param = data['param_point']
        data = data['protocol_runs']['Protocol_LongShortSI']
        pnames = data['population_names']
        VNA = np.sum(np.stack([VNA_coeffs[pname] * data["fr_history"][:, pnames.index(pname)] for pname in VNA_coeffs.keys()], axis = 1), axis=1)

        conf_file_path = os.path.join(f"..", "..", "data", "experiments", f"{prefix}_{p}", "config_file.pkl")
        config = pickle.load(open(conf_file_path, 'rb'))
        dt = config["model_params"]["dt"]
        noSI_T = config["protocol_dict"]["Protocol_LongShortSI"]["noSI_T"]
        longSI_T = config["protocol_dict"]["Protocol_LongShortSI"]["longSI_T"]
        interim_T = config["protocol_dict"]["Protocol_LongShortSI"]["interim_T"]
        stim_duration = config["protocol_dict"]["Protocol_LongShortSI"]["stim_duration"]

        t_start_ind = get_index(3 * noSI_T / 4, dt)
        t_stop_ind = -get_index(3 * noSI_T / 4, dt)
        long_stim_start_ind = get_index(noSI_T, dt) - t_start_ind
        long_stim_stop_ind = get_index(noSI_T + longSI_T, dt) - t_start_ind
        short_stim_start_ind = get_index(noSI_T + longSI_T + interim_T, dt) - t_start_ind
        short_stim_stop_ind = get_index(noSI_T + longSI_T + interim_T + stim_duration, dt) - t_start_ind
        VNA = VNA[t_start_ind:t_stop_ind]

        plot_VNA_responses(VNA, long_stim_start_ind, long_stim_stop_ind, short_stim_start_ind, short_stim_stop_ind)

