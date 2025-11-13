import pickle
from rCPGswCPG.utils.gen_utils import get_project_root, create_dir_if_not_exist
from rCPGswCPG.Experiment import Experiment
import numpy as np
from itertools import product
import os

if __name__ == '__main__':
    model_name = "model_complex"
    exp_name = "Experiment_varying_connectivity_to_Insp"
    param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
    base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")

    model_params = pickle.load(open(os.path.join(f"{param_folder}", f"params_{model_name}.pkl"), 'rb+'))
    description = "Varying the strength of connections Sw1->Insp and Sensory_relay->Insp"

    config_dict = dict()
    config_dict["model_params"] = model_params
    config_dict["description"] = description
    config_dict["varied_params"] = ["self.model.W[self.model.pnames.index(\"Sensory_relay\"),self.model.pnames.index(\"Insp\")]",
                                    "self.model.W[self.model.pnames.index(\"Sw1\"),self.model.pnames.index(\"Insp\")]"]
    config_dict["param_points"] = list(product(np.linspace(-1, 0, 11), np.linspace(-1, 0, 11)))
    config_dict["protocol_dict"] = {"Protocol_noSI": {"T": 30},
                                    "Protocol_longSI": {"T": 10, "amp": 0.45},
                                    "Protocol_shortSI": {"interim_T": 10, "amp": 0.45, "stim_duration": 0.1},
                                    "Protocol_LongShortSI": {"noSI_T": 3, "longSI_T": 10,
                                                             "interim_T": 3, "amp": 0.45, "stim_duration": 0.1,
                                                             "n_stim": 1}}
    exp = Experiment(config_dict, base_folder)
    exp.run(rerun=True)
    exp.plot_traces(param_names_tuple="(W[Sw1, Insp], W[SR, Insp]): ")
    exp.analyse_data()