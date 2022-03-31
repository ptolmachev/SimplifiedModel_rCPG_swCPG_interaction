#plotting the comparison of experimental and simulated recordings of Vagal and Phrenic nerves
# during normal swallowing and eupneic breathing
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
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
    model.run(50, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))

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

# # DO NOT UNCOMMENT AND RUN
# recordings_dict = run_sim()
# save_to = os.path.join(get_project_root(), "data", "recordings_eupneic_breathing.pkl")
# pickle.dump(recordings_dict, open(save_to, "wb+"))

# Simulation data
load_from = os.path.join(get_project_root(), "data", "recordings_eupneic_breathing_swallowing.pkl")
recordings = pickle.load(open(load_from, "rb+"))
sim_PNA = recordings["PNA"]
sim_VNA = recordings["VNA"]
neural_traces = recordings["traces"]
t_sim = recordings["t"]

# Experimental data
experimental_recordings_data = os.path.abspath(r"C:\Users\betad\Projects\Exp_Data_Processing_rCPG\data\sln_prc_filtered")
file = "2019-08-22_15-59-55_t1"
ind_start = 11000+174
ind_end = 28000+174
exp_PNA_data = pickle.load(open(os.path.join(experimental_recordings_data, file, "100_CH10_processed.pkl"), "rb+"))
exp_PNA = exp_PNA_data["signal"]
exp_VNA_data = pickle.load(open(os.path.join(experimental_recordings_data, file, "100_CH15_processed.pkl"), "rb+"))
exp_VNA = exp_VNA_data["signal"]
fr = exp_PNA_data['fr']
dt = 1.0 / fr
t_exp = np.arange(len(exp_PNA)) * dt

#make all of the arrays to have the same length.
interp_length = 10000
t_new = np.linspace(0,interp_length, interp_length)
PNA_s = interp1d(np.linspace(0,interp_length, len(sim_PNA)),sim_PNA)(t_new)
PNA_e = interp1d(np.linspace(0,interp_length, len(exp_PNA[ind_start:ind_end])),sg(exp_PNA[ind_start:ind_end],51,3))(t_new)
VNA_s = interp1d(np.linspace(0,interp_length, len(sim_VNA)),sim_VNA)(t_new)
VNA_e = interp1d(np.linspace(0,interp_length, len(exp_VNA[ind_start:ind_end])),sg(exp_VNA[ind_start:ind_end],51,3))(t_new)
t1, t2, t3, t4 = 1250, 1335, 1642, 1752
lb, ub = 5600, 18500

for j, key in enumerate(["experiemtns", "simulations"]):
    fig1, axes1 = plt.subplots(2, 1, figsize=(15,0.8*2))
    plt.subplots_adjust(wspace= 0.25, hspace=0.25)
    if key == "experiemtns":
        axes1[0].plot(PNA_e[lb:ub], color='r', linewidth=2, label=f"PNA, {key}")
        axes1[1].plot(VNA_e[lb:ub], color='b', linewidth=2, label=f"VNA, {key}")
    elif key == "simulations":
        axes1[0].plot(PNA_s[lb:ub], color='r', linewidth=2, label=f"PNA, {key}")
        axes1[1].plot(VNA_s[lb:ub], color='b', linewidth=2, label=f"VNA, {key}")

    for i in range(2):
        axes1[i].legend(loc = 1, fontsize = 15)
        axes1[i].axis('off')
        axes1[i].plot(np.zeros_like(t_new[lb:ub]), color='k', alpha=0.1)
        axes1[i].axvline(t1, color='k', linestyle='--')
        axes1[i].axvline(t2, color='k', linestyle='--')
        axes1[i].axvline(t3, color='k', linestyle='--')
        axes1[i].axvline(t4, color='k', linestyle='--')
        axes1[i].axvspan(t1, t2, color='r', alpha=0.05)
        axes1[i].axvspan(t2, t3, color='b', alpha=0.05)
        axes1[i].axvspan(t3, t4, color='g', alpha=0.05)

    plt.subplots_adjust(wspace=0, hspace=0)
    img_folder = os.path.join(get_project_root(), "img")
    plt.savefig(os.path.join(img_folder, f"comparison_eupnea_nerves_and_neurons_nerves_{key}.pdf"))
    plt.savefig(os.path.join(img_folder, f"comparison_eupnea_nerves_and_neurons_nerves_{key}.svg"))
    plt.show(block=True)
    plt.close()

pnames = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
pnames_to_plot = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
labels = ["I (early-I)", "ramp-I", "E (post-I)", "late-E", r"Sw_1", r"Sw_2", "Sensory Relay", "pontine post-I", "sw.-gate control"]

fig2, axes2 = plt.subplots(len(labels), 1, figsize=(15,0.8*len(labels)))
plt.subplots_adjust(wspace= 0.25, hspace= 0.25)
for i in range(len(labels)):
    raw_trace = neural_traces[:, pnames.index(pnames_to_plot[i])]
    trace = interp1d(np.linspace(0, interp_length, len(raw_trace)), raw_trace)(t_new)
    axes2[i].plot(trace[lb:ub], color='k', linewidth=2, label=labels[i])
    axes2[i].legend(loc=1, fontsize=15)
    axes2[i].set_ylim([0, 0.9])
    axes2[i].legend(loc=1, fontsize=15)
    axes2[i].axis('off')
    axes2[i].plot(np.zeros_like(t_new[lb:ub]), color='k', alpha=0.1)
    axes2[i].axvline(t1, color='k', linestyle='--')
    axes2[i].axvline(t2, color='k', linestyle='--')
    axes2[i].axvline(t3, color='k', linestyle='--')
    axes2[i].axvline(t4, color='k', linestyle='--')
    axes2[i].axvspan(t1, t2, color='r', alpha=0.05)
    axes2[i].axvspan(t2, t3, color='b', alpha=0.05)
    axes2[i].axvspan(t3, t4, color='g', alpha=0.05)

plt.subplots_adjust(wspace=0, hspace=0)
img_folder = os.path.join(get_project_root(), "img")
plt.savefig(os.path.join(img_folder, f"comparison_eupnea_nerves_and_neurons_traces.pdf"))
plt.savefig(os.path.join(img_folder, f"comparison_eupnea_nerves_and_neurons_traces.svg"))
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