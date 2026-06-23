import os
import sys

import hydra
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

if __package__ is None or __package__ == "":
    # Support direct script execution: python /path/to/run_model.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rCPGswCPG.Network import firing_rate
from rCPGswCPG.construct_model import construct_model
from rCPGswCPG.model_params.config_loader import model_params_from_cfg
from rCPGswCPG.protocols.protocols import run_standalone_protocol
from rCPGswCPG.utils.gen_utils import get_project_root
from rCPGswCPG.utils.utils import plot_data
from rCPGswCPG.Experiment import Experiment

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_params = model_params_from_cfg(cfg.model_params)
    model = construct_model(model_params)
    external_inputs = np.zeros(model_params["N"])
    pnames = model_params["pnames"]

    # modify parameters if needed
    # model.W[pnames.index("Exp"), pnames.index("Sensory_relay")] = 0.3
    # model.W[pnames.index("Exp"), pnames.index("Sensory_relay")] = 0.9
    # model.W[pnames.index("KF_phasic"), pnames.index("Sensory_relay")] = 0.9
    # model.W[pnames.index("Insp"), pnames.index("Sensory_relay")] = -0.0
    # model.W[pnames.index("Sw1"), pnames.index("Sw2")] = -1.0
    # model.W[pnames.index("Insp"), pnames.index("Sensory_relay")] = -0.1
    # model.W[pnames.index("Insp"), pnames.index("Sw1")] = -1.0
    # model.populations[pnames.index("KF_phasic")].drive = 0.6
    # model.populations[pnames.index("KF_gate")].drive = 0.6

    exp_name = cfg.experiment.name
    base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")
    description = cfg.experiment.description

    config_dict = dict()
    config_dict["model_params"] = model_params
    config_dict["description"] = description
    config_dict["varied_params"] = None
    config_dict["param_points"] = None
    config_dict["protocol_dict"] = OmegaConf.to_container(cfg.experiment.protocol_dict, resolve=True)
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


if __name__ == "__main__":
    main()