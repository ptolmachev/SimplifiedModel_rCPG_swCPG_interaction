from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from src.Network import firing_rate
from src.construct_model import construct_model
from src.utils.gen_utils import get_project_root, create_dir_if_not_exist, modify_array
from src.utils.utils import plot_data
# run model with the relevant paramters and save the data

model_name = "model_complex"
param_folder = os.path.join(f'{get_project_root()}', 'data', 'model_params')
model_params = pickle.load(open(os.path.join(f'{param_folder}', f'params_{model_name}.pkl'), 'rb+'))
model = construct_model(model_params)
external_inputs = np.zeros(model_params["N"])
pnames = model_params["pnames"]

#modify parameters if needed
# model.populations[model.pnames.index("KF_phasic")].drive = 0
# model.populations[model.pnames.index("Exp")].drive = 0
# model.populations[model.pnames.index("Insp")].drive = 0.3
# model.populations[model.pnames.index("RampI")].drive = 0.3
# model.W[model.pnames.index("Sensory_relay"), model.pnames.index("Insp")] = -0.3
# model.W[model.pnames.index("Sensory_relay"), model.pnames.index("Sw1")] = 0.052
# model.W[model.pnames.index("Sensory_relay"), model.pnames.index("Insp")] = -0.5
# model.W[model.pnames.index("Sensory_relay"), model.pnames.index("RampI")] = -0.5
# T_transient = 10
# T = 7.5
# T_long_stim = 10
# amp = 0.4
# external_inputs = np.zeros(len(pnames))
# model.run(T_transient, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
# model.clear_history()
# model.run(T, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0))
# model.run(T_long_stim, input=modify_array(external_inputs, pnames.index("Sensory_relay"), amp))
# model.run(T, input=modify_array(external_inputs, pnames.index("Sensory_relay"), 0.0))
# # collecting data
# v_history = model.get_raw_history()
# fr_history = firing_rate(v_history)
# t = (model_params['dt'] * np.arange(fr_history.shape[0]) / 1000)  # in sec
# data = dict()
# data["fr_history"] = fr_history
# data["v_history"] = v_history
# data["pnames"] = model.pnames
# data["t"] = t
# data["dt"] = model.dt

# save_to = os.path.join(get_project_root(), "data", "recordings_eupneic_breathing.pkl")
# pickle.dump(data, open(save_to, "wb+"))



load_from = os.path.join(get_project_root(), "data", "recordings_delayed_swallow.pkl")
data = pickle.load(open(load_from, "rb+"))

pnames = data["pnames"]
fr_history = data["fr_history"]
v_history = data["v_history"]
t = data["t"]

VNA_components = {"KF_phasic" : 0.75, "Sw1" : 0.6, "RampI" : 0.9}
PNA = fr_history[:, pnames.index("RampI")]
VNA = np.zeros_like(PNA)
for pop in VNA_components.keys():
    VNA += VNA_components[pop] * fr_history[:, pnames.index(pop)]

stim_start = 30000
stim_end = 70000
fig, axes = plt.subplots(2, 1, figsize = (10, 2.5))
axes[0].plot(PNA, linewidth = 2, c = 'r', label = "PNA")
axes[1].plot(VNA, linewidth = 2, c = 'b', label = "pathophysiological VNA")
for i in range(2):
    axes[i].legend(fontsize= 15, loc = 1)
    axes[i].axvline(stim_start, color = 'k', linestyle = '--')
    axes[i].axvline(stim_end, color = 'k', linestyle = '--')
    axes[i].axvspan(stim_start, stim_end, color = 'grey', alpha = 0.05)
    axes[i].axis('off')
# plt.suptitle("Normal swallowing", fontsize=15)
plt.suptitle("Weak and delayed swallowing response", fontsize=15)
plt.subplots_adjust(wspace=0, hspace=0)

img_folder = os.path.join(get_project_root(), "img")
plt.savefig(os.path.join(img_folder, "delayed_swallow.pdf"),bbox_inches='tight')
plt.savefig(os.path.join(img_folder, "delayed_swallow.svg"),bbox_inches='tight')
plt.show()
# plt.close()