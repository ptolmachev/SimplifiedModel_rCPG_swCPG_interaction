import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from rCPGswCPG.Network import firing_rate
from rCPGswCPG.construct_model import construct_model
from rCPGswCPG.protocols.protocols import run_standalone_protocol
from rCPGswCPG.utils.gen_utils import get_project_root
from rCPGswCPG.utils.utils import plot_data
from rCPGswCPG.Experiment import Experiment

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

exp_name = f"Intact_full_model_test"
param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")
model_params = pickle.load(open(os.path.join(f"{param_folder}", f"params_{model_name}.pkl"), 'rb+'))
description = f"Intact full model test"

config_dict = dict()
config_dict["model_params"] = model_params
config_dict["description"] = description
config_dict["varied_params"] = None
config_dict["param_points"] = None
config_dict["protocol_dict"] = {"Protocol_noSI" : {"T": 30},
                                "Protocol_longSI" : {"T": 10, "amp": 0.45},
                                "Protocol_shortSI": {"interim_T": 10, "amp" : 0.45, "stim_duration": 0.1},
                                "Protocol_LongShortSI": {"noSI_T": 10, "longSI_T": 10, 
                                                         "interim_T": 10, "amp": 0.45, "stim_duration": 0.1, "n_stim": 1},
                                "Protocol_noSI" : {"T": 30}}
exp = Experiment(config_dict, base_folder)
exp.run(rerun=True)
exp.plot_traces(param_names_tuple=None, show=True)


# # alternatively one can run the model directly without using the Experiment class

# run_standalone_protocol(model)
# # collecting data
# v_history, m_history = model.get_raw_history()
# fr_history = firing_rate(v_history)
# t = (model_params['dt'] * np.arange(fr_history.shape[0]) / 1000)  # in sec
# # PLOT
# # VNA_components is the list of coefficients corresponding to the strengths of populations contributing to
# # the Vagus nerve activity
# VNA_components = {"KF_phasic" : 0.55, "Exp" : 0.2, "Sw1" : 0.6, "RampI" : 0.9}
# fig, axes = plot_data(t, fr_history, model_params['pnames'], model_params['pnames'], VNA_components=VNA_components)
# plt.show(block=True)
# plt.close(fig)