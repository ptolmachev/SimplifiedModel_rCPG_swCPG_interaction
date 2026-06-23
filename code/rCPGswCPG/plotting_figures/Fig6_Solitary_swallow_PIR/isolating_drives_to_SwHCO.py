import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from rCPGswCPG.Network import construct_model
from rCPGswCPG.Network import firing_rate
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
from rCPGswCPG.utils.gen_utils import get_project_root

model_name = "model_complex"
model_params = model_params_from_cfg(load_model_cfg_file(model_name.replace("model_", "")))
model = construct_model(model_params)
external_inputs = np.zeros(model_params["N"])
pnames = model_params["pnames"]
print(pnames)
pnames2plot = ["Sw1", "Sw2"]

T_no_stim = 15
T_stim = 0.15
T_afterstim = 0.5
stim_input = np.zeros(model_params["N"])
stim_input[pnames.index("Sensory_relay")] = 0.4

model.run(T_no_stim, input=np.zeros(model.N))
model.run(T_stim, input=stim_input)
model.run(T_afterstim, input=np.zeros(model.N))

# collecting data
v_history, m_history = model.get_raw_history()
v_history = v_history.T
m_history = m_history.T
fr_history = firing_rate(v_history, model.beta)
net_inputs = (model.W @ fr_history) + np.array([model.populations[i].drive for i in range(model.N)]).reshape(-1, 1)
net_inputs[pnames.index("Sw1")] -= model.W[pnames.index("Sw1"), pnames.index("Sw2")] * fr_history[pnames.index("Sw2")]
net_inputs[pnames.index("Sw2")] -= model.W[pnames.index("Sw2"), pnames.index("Sw1")] * fr_history[pnames.index("Sw1")]
SR2Sw1 = model.W[pnames.index("Sw1"), pnames.index("Sensory_relay")] * fr_history[pnames.index("Sensory_relay")]
SR2Sw2 = model.W[pnames.index("Sw2"), pnames.index("Sensory_relay")] * fr_history[pnames.index("Sensory_relay")]

t = (model.dt * np.arange(fr_history.shape[1]) / 1000)  # in sec
t_start = int((T_no_stim * (1000 / model.dt) - 1500))

fig, ax = plt.subplots(1, 1, figsize = (7, 3))
ax.plot(t[t_start:], SR2Sw1[t_start:], label = "SR2Sw1")
ax.plot(t[t_start:], SR2Sw2[t_start:], label = "SR2Sw2")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(frameon=False)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (7, 3))
for i, pname in enumerate(pnames2plot):
    ax.plot(t[t_start:], fr_history[pnames.index(pname), t_start:], label = pname)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(frameon=False)
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (7, 3))
for i, pname in enumerate(pnames2plot):
    ax.plot(t[t_start:], net_inputs[pnames.index(pname), t_start:], label = pname)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(frameon=False)
plt.show()
print(np.mean(net_inputs[pnames.index("Sw1")]))
print(np.mean(net_inputs[pnames.index("Sw2")]))