#plotting the comparison of experimental and simulated recordings of Vagal and Phrenic nerves
# during normal swallowing and eupneic breathing
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle

from matplotlib.gridspec import GridSpec

from src.Network import firing_rate
from src.construct_model import construct_model
from src.exp_protocols.protocols import run_full_protocol, run_KF_inhibited_protocol
from src.utils.gen_utils import get_project_root, modify_array
from scipy.signal import savgol_filter as sg
from scipy.interpolate import interp1d

def run_sim():
    model_name = "model_complex"
    param_folder = os.path.join(f'{get_project_root()}', 'data', 'model_params')
    model_params = pickle.load(open(os.path.join(f'{param_folder}', f'params_{model_name}.pkl'), 'rb+'))
    model = construct_model(model_params)
    external_inputs = np.zeros(model_params["N"])
    pnames = model_params["pnames"]

    T_transient = 15 # give it some time to approach the limit cycle
    model.run(T_transient, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
    model.clear_history()
    model.run(15, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
    model.run(10, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.45))
    model.run(15, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))

    # collecting data
    v_history = model.get_raw_history()
    fr_history = firing_rate(v_history)
    t_sim = (model_params['dt'] * np.arange(fr_history.shape[0]) / 1000)  # in sec
    sim_PNA = fr_history[:, pnames.index("RampI")]
    sim_VNA = + 0.75 * fr_history[:, pnames.index("KF_phasic")] \
              + 0.9 * fr_history[:, pnames.index("RampI")] \
              + 0.6 * fr_history[:, pnames.index("Sw1")]
    recordings_dict = dict()
    recordings_dict["PNA"] = sim_PNA
    recordings_dict["VNA"] = sim_VNA
    recordings_dict["traces"] = fr_history
    recordings_dict["pnames"] = model.pnames
    recordings_dict["t"] = t_sim
    return recordings_dict

# recordings_dict = run_sim()
# save_to = os.path.join(get_project_root(), "data", "recordings_swallowing_and_breathing.pkl")
# pickle.dump(recordings_dict, open(save_to, "wb+"))

# Simulation data
load_from = os.path.join(get_project_root(), "data", "recordings_swallowing_and_breathing.pkl")
recordings = pickle.load(open(load_from, "rb+"))
sim_PNA = recordings["PNA"]
sim_VNA = recordings["VNA"]
t_sim = recordings["t"]
neural_traces = recordings["traces"]
pnames = recordings["pnames"]

#make all of the arrays to have the same length.
interp_length = 10000
t_new = np.linspace(0,interp_length, interp_length)
PNA_s = interp1d(np.linspace(0,interp_length, len(sim_PNA)),sim_PNA)(t_new)
VNA_s = interp1d(np.linspace(0,interp_length, len(sim_VNA)),sim_VNA)(t_new)

fig1 = plt.figure(figsize=(15, 0.8 * 3))
gs = GridSpec(3,1, figure=fig1)
# create sub plots as grid
axes = []
axes.append(fig1.add_subplot(gs[0:1]))
axes.append(fig1.add_subplot(gs[1:3]))
axes[0].plot(PNA_s, color='r', linewidth=2, label="simulated PNA")
axes[1].plot(VNA_s, color='b', linewidth=2, label="simulated VNA")
stim_start_ind = 3750
stim_end_ind = 6250
for i in range(2):
    axes[i].legend(loc = 1, fontsize = 15)
    axes[i].axis('off')
    axes[i].axvline(stim_start_ind, color='k', linestyle='--')
    axes[i].axvline(stim_end_ind, color='k', linestyle='--')
    axes[i].axvspan(stim_start_ind, stim_end_ind, color='k', alpha=0.05)
    axes[i].plot(np.zeros_like(t_new), color='k', alpha=0.1)

plt.subplots_adjust(wspace=0, hspace=0)
img_folder = os.path.join(get_project_root(), "img")
plt.savefig(os.path.join(img_folder, "eupneic_nerve_activity.pdf"),bbox_inches='tight')
plt.savefig(os.path.join(img_folder, "eupneic_nerve_activity.svg"),bbox_inches='tight')
plt.show(block=True)
plt.close()

# other files which closely match
# file = "2019-08-22_16-06-37_t4"
# ind_start = 2300
# ind_end = 14200
# file = "2019-08-22_16-10-57_t6"
# ind_start = 4600
# ind_end = 16500
# load the data