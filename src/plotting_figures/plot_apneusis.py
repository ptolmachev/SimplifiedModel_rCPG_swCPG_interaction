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

    KF_populations = ["KF_gate", "KF_phasic"]
    for KF_pop in KF_populations:
        model.populations[pnames.index(KF_pop)].drive = 0
        for name in pnames:
            model.W[pnames.index(name), pnames.index(KF_pop)] = 0.0
            model.W[pnames.index(KF_pop), pnames.index(name)] = 0.0
    model.populations[pnames.index("Exp")].drive = 0.0
    model.populations[pnames.index("Insp")].drive = 0.3
    model.populations[pnames.index("RampI")].drive = 0.3

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
    recordings_dict["pnames"] = pnames
    recordings_dict["traces"] = fr_history
    recordings_dict["t"] = t_sim
    return recordings_dict

# recordings_dict = run_sim()
# save_to = os.path.join(get_project_root(), "data", "recordings_apneustic_breathing.pkl")
# pickle.dump(recordings_dict, open(save_to, "wb+"))

# # Simulation data
load_from = os.path.join(get_project_root(), "data", "recordings_apneustic_breathing.pkl")
recordings = pickle.load(open(load_from, "rb+"))
sim_PNA = recordings["PNA"]
sim_VNA = recordings["VNA"]
neural_traces = recordings["traces"]
t_sim = recordings["t"]

interp_length = 10000
t_new = np.linspace(0,interp_length, interp_length)
lb, ub = 0,10000
t1, t2, t3 = 687, 1007, 1886
t_stim_start, t_stim_end = 3750, 6250
pnames = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
pnames_to_plot = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay"]
lbls = ["I (early-I)", "ramp-I", "E (post-I)", "late-E", r"$Sw_{1}$", r"$Sw_{2}$", "Sensory relay"]
N = len(pnames_to_plot)
fig1 = plt.figure(figsize=(15, 0.8 * 3))
gs = GridSpec(3,1, figure=fig1)
# create sub plots as grid
axes1 = []
axes1.append(fig1.add_subplot(gs[0:1]))
axes1.append(fig1.add_subplot(gs[1:3]))
for i in range(2):
    axes1[i].axis('off')
    axes1[i].plot(np.zeros_like(t_new[lb:ub]), color='k', alpha=0.1)
    axes1[i].axvline(t1, color='k', linestyle='--')
    axes1[i].axvline(t2, color='k', linestyle='--')
    axes1[i].axvline(t3, color='k', linestyle='--')
    axes1[i].axvline(t_stim_start, color='k', linestyle='--')
    axes1[i].axvline(t_stim_end, color='k', linestyle='--')
    axes1[i].axvline(t3, color='k', linestyle='--')
    axes1[i].axvspan(t1, t2, color='r', alpha=0.05)
    axes1[i].axvspan(t2, t3, color='b', alpha=0.05)
    axes1[i].axvspan(t_stim_start, t_stim_end, color='k', alpha=0.05)
    if i == 0:
        PNA_s = interp1d(np.linspace(0, interp_length, len(sim_PNA)),sim_PNA)(t_new)
        axes1[i].plot(PNA_s[lb:ub], color='r', linewidth=2, label="PNA, simulated")
    elif i == 1:
        VNA_s = interp1d(np.linspace(0, interp_length, len(sim_VNA)),sim_VNA)(t_new)
        axes1[i].plot(VNA_s[lb:ub], color='b', linewidth=2, label="VNA, simulated")
    axes1[i].legend(loc = 1, fontsize = 15)
    axes1[i].set_ylim([0, 1.0])

plt.subplots_adjust(wspace=0, hspace=0)
img_folder = os.path.join(get_project_root(), "img")
plt.savefig(os.path.join(img_folder, "comparison_apneusis_nerves.pdf"),bbox_inches='tight')
# plt.savefig(os.path.join(img_folder, "comparison_apneusis_nerves.svg"),bbox_inches='tight')
plt.show(block=True)
plt.close()


fig2, axes2 = plt.subplots(N, 1, figsize=(15, (N)*0.8))
for i in range(len(lbls)):
    axes2[i].axis('off')
    axes2[i].plot(np.zeros_like(t_new[lb:ub]), color='k', alpha=0.1)
    axes2[i].axvline(t1, color='k', linestyle='--')
    axes2[i].axvline(t2, color='k', linestyle='--')
    axes2[i].axvline(t3, color='k', linestyle='--')
    axes2[i].axvline(t_stim_start, color='k', linestyle='--')
    axes2[i].axvline(t_stim_end, color='k', linestyle='--')
    axes2[i].axvline(t3, color='k', linestyle='--')
    axes2[i].axvspan(t1, t2, color='r', alpha=0.05)
    axes2[i].axvspan(t2, t3, color='b', alpha=0.05)
    axes2[i].axvspan(t_stim_start, t_stim_end, color='k', alpha=0.05)
    raw_trace = neural_traces[:, i]
    trace = interp1d(np.linspace(0, interp_length, len(raw_trace)),raw_trace)(t_new)
    axes2[i].plot(trace[lb:ub], color='k', linewidth=2, label=lbls[i])
    axes2[i].legend(loc = 1, fontsize = 15)
    axes2[i].set_ylim([0, 1.0])
plt.subplots_adjust(wspace=0, hspace=0)
img_folder = os.path.join(get_project_root(), "img")
plt.savefig(os.path.join(img_folder, "comparison_apneusis_traces.pdf"),bbox_inches='tight')
# plt.savefig(os.path.join(img_folder, "comparison_apneusis_nerves.svg"),bbox_inches='tight')
plt.show(block=True)
plt.close()


