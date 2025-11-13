import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from rCPGswCPG.Network_ import firing_rate
from rCPGswCPG.construct_model import construct_model
from rCPGswCPG.utils.gen_utils import get_project_root

#test
model_name = "swHCO"
param_folder = os.path.join(f'{get_project_root()}', 'data', 'model_params')
model_params = pickle.load(open(os.path.join(f'{param_folder}', f'params_{model_name}.pkl'), 'rb+'))
model = construct_model(model_params)
external_inputs = np.zeros(model_params["N"])
pnames = model_params["pnames"]

T_no_stim = 15
T_stim = 0.15
T_afterstim = 0.5
stim_input = np.array([0.12, 0.07])
# stim_input = np.array([0.17, 0.07]) # <-- No PIR
model.run(T_no_stim, input=np.zeros(2))
model.run(T_stim, input=stim_input)
model.run(T_afterstim, input=np.zeros(2))

# collecting data
v_history, m_history = model.get_raw_history()
fr_history = firing_rate(v_history)
t = (model.dt * np.arange(fr_history.shape[0]) / 1000)  # in sec
t_start = int((T_no_stim * (1000 / model.dt) - 1500))

fig, ax = plt.subplots(1, 1, figsize = (7, 3))
ax.plot(t[t_start:], v_history[t_start:, 0], color = 'r', label = 'Sw1')
ax.plot(t[t_start:], v_history[t_start:, 1], color = 'b', label = 'Sw2')
ax.axvline(T_no_stim, alpha = 0.1)
ax.axvline(T_no_stim + T_stim, alpha = 0.1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend()
plt.show()