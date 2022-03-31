import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from src.Network import firing_rate
from src.construct_model import construct_model
from src.exp_protocols.protocols import run_full_protocol, run_KF_inhibited_protocol
from src.utils.gen_utils import get_project_root
from src.utils.utils import plot_data

#test
model_name = "model_complex"
param_folder = os.path.join(f'{get_project_root()}', 'data', 'model_params')
model_params = pickle.load(open(os.path.join(f'{param_folder}', f'params_{model_name}.pkl'), 'rb+'))
model = construct_model(model_params)
external_inputs = np.zeros(model_params["N"])
pnames = model_params["pnames"]

#modify parameters if needed
# model.W[pnames.index("Sensory_relay"), pnames.index("Exp")] = 0.3
# model.W[pnames.index("Sensory_relay"), pnames.index("Exp")] = 0.9
# model.W[pnames.index("Sensory_relay"), pnames.index("KF_phasic")] = 0.9
# model.W[pnames.index("Sensory_relay"), pnames.index("Insp")] = -0.0
# model.W[pnames.index("Sw2"), pnames.index("Sw1")] = -1.0
# model.W[pnames.index("Sensory_relay"), pnames.index("Insp")] = -0.1
# model.W[pnames.index("Sw1"), pnames.index("Insp")] = -1.0
# model.populations[pnames.index("KF_phasic")].drive = 0.6
# model.populations[pnames.index("KF_gate")].drive = 0.6

run_full_protocol(model)
# run_KF_inhibited_protocol(model)

# collecting data
v_history = model.get_raw_history()
fr_history = firing_rate(v_history)
t = (model_params['dt'] * np.arange(fr_history.shape[0]) / 1000)  # in sec

# plot
# VNA_components is the list of coefficients corresponding to the strengths of populations contributing to
# the Vagus nerve activity
VNA_components = {"KF_phasic" : 0.55, "Exp" : 0.2, "Sw1" : 0.6, "RampI" : 0.9}
# VNA_components = {"KF_phasic" : 0.75, "Sw1" : 0.6, "Insp" : 0.55}
# VNA_components = {"Exp" : 0.75, "Sw1" : 0.6, "Insp" : 0.55}

fig, axes = plot_data(t, fr_history, model_params['pnames'], model_params['pnames'], VNA_components=VNA_components)
plt.show(block=True)
plt.close(fig)