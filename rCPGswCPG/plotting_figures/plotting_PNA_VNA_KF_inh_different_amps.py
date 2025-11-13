from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import pickle
from rCPGswCPG.utils.gen_utils import get_project_root, create_dir_if_not_exist

W_Insp_Sw1 = 0.005
data_path = os.path.join(get_project_root(), "data", "experiments", f"KF_lesioning_different_amplitude_stim_{W_Insp_Sw1}", "runs")
img_path = os.path.join(get_project_root(), "data", "experiments", f"KF_lesioning_different_amplitude_stim_{W_Insp_Sw1}", "imgs", f"VNA_PNA_W_Insp_Sw1={W_Insp_Sw1}")
create_dir_if_not_exist(img_path)
files = os.listdir(data_path)
#sort files
nums = [int(file.split("_")[0]) for file in files]
nums, files = zip(*sorted(zip(nums, files)))
for file in files[::-1]:
    data = pickle.load(open(os.path.join(data_path, file), "rb+"))
    num = file.split("_")[0]
    amp = file.split("_")[-1].split(".pkl")[0]
    dt = data["protocol_runs"]["full_protocol"]["dt"]
    pnames = data["protocol_runs"]["full_protocol"]["population_names"]
    fr_all = data["protocol_runs"]["full_protocol"]["fr_history"]
    offset = int((2000)/dt)
    T_start = int(30*1000/dt) - offset
    T_stop = int(40*1000/dt) + offset
    T_dur = int(10 * 1000 / dt)
    coeffs_VNA = {"KF_phasic" : 0.75,  "Sw1" : 0.6, "RampI" : 0.9}
    PNA = fr_all[T_start:T_stop, pnames.index("Insp")]
    VNA = np.sum(np.array([fr_all[T_start:T_stop, pnames.index(pop)] * coeffs_VNA[pop] for pop in list(coeffs_VNA.keys())]), axis = 0)
    time_series = [PNA, VNA]
    labels = ["PNA", "VNA"]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,3))

    plt.suptitle(f"Stim amplitude = {amp}")
    for i, ax in enumerate(axes):
        ax.set_axis_off()
        ax.plot(np.arange(len(time_series[i]))*dt, time_series[i], color='k', linewidth=2, label=labels[i])
        ax.axvline(offset*dt, color = 'k', linestyle ="--")
        ax.axvline(offset*dt+T_dur*dt, color = 'k', linestyle ="--")
        legend = ax.legend(fontsize=16, loc=1)
        legend.get_frame().set_linewidth(0)
    plt.subplots_adjust(wspace=0, hspace = 0)
    plt.savefig(os.path.join(img_path, f"{file}.png"), bbox_inches='tight', dpi=300, transparent=True)
    plt.savefig(os.path.join(img_path, f"{file}.pdf"), bbox_inches='tight', dpi=300, transparent=True)
    plt.show(block = True)
    plt.close(fig)




