from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from rCPGswCPG.Network_ import firing_rate
from rCPGswCPG.construct_model import construct_model
from rCPGswCPG.plotting_figures.plotting_utils import plot_recordings
from rCPGswCPG.utils.gen_utils import get_project_root, put


rerun = True
T = 7.5
T_long_stim = 10
amp = 0.4
# mod = 'insp_breakthroughs'
mod = 'weak_swallow'

model_name = "model_complex"
param_folder = os.path.join(f'{get_project_root()}', 'data', 'model_params')
model_params = pickle.load(open(os.path.join(f'{param_folder}', f'params_{model_name}.pkl'), 'rb+'))
model = construct_model(model_params)
external_inputs = np.zeros(model_params["N"])
pnames = model_params["pnames"]
# Modifications
if mod == 'insp_breakthroughs':
    model.W[pnames.index("Insp"), pnames.index("Sensory_relay")] = -0.22
    model.W[pnames.index("RampI"), pnames.index("Sensory_relay")] = -0.22
elif mod == 'weak_swallow':
    model.W[pnames.index("Sw1"), pnames.index("Sensory_relay")] = 0.05

recordings_file_name = os.path.join(get_project_root(), "data", f"recordings_{mod}.pkl")
if not os.path.exists(recordings_file_name) or rerun:
    external_inputs = np.zeros(len(pnames))
    model.run(T, input=put(external_inputs, pnames.index("Sensory_relay"), 0))
    model.run(T_long_stim, input=put(external_inputs, pnames.index("Sensory_relay"), amp))
    model.run(T, input=put(external_inputs, pnames.index("Sensory_relay"), 0.0))
    # collecting data
    v_history, m_history = model.get_raw_history()
    fr_history = firing_rate(v_history)
    t = (model_params['dt'] * np.arange(fr_history.shape[0]) / 1000)  # in sec
    data = dict()
    data["fr_history"] = fr_history
    data["v_history"] = v_history
    data["pnames"] = pnames
    data["t"] = t
    data["dt"] = model.dt
    pickle.dump(data, open(recordings_file_name, "wb+"))
load_from = os.path.join(get_project_root(), "data", f"recordings_{mod}.pkl")
data = pickle.load(open(load_from, "rb+"))

fr_history = data["fr_history"]
v_history = data["v_history"]
t = data["t"]

# PLOT
img_folder = os.path.join(get_project_root(), "img")
VNA_components = {"KF_phasic" : 0.75, "Sw1" : 0.6, "RampI" : 0.9}
PNA = fr_history[:, pnames.index("RampI")]
VNA = np.zeros_like(PNA)
for pop in VNA_components.keys():
    VNA += VNA_components[pop] * fr_history[:, pnames.index(pop)]
data_sim = {'PNA': PNA, 'VNA': VNA}
stim_start = T / model_params['dt'] * 1000
stim_end = (T + T_long_stim)/model_params['dt'] * 1000
plot_recordings(t, data_sim, ['PNA', 'VNA'],
                color_map= {'PNA': 'red', "VNA": "blue"},
                aspect_ratios=[1, 2],
                label_map={'PNA': 'PNA', 'VNA': 'VNA'},
                stim_start=stim_start, stim_end=stim_end,
                ylims=None, outpath=os.path.join(img_folder, f"pathological_behavior_{mod}.pdf"))