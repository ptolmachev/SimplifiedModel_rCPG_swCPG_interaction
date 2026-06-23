import pickle
from rCPGswCPG.utils.gen_utils import get_project_root
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
from rCPGswCPG.Experiment import Experiment
import numpy as np
import os

# see class Experiment for more details
# Runs the experiment only if the corresponding folder is not empty

if __name__ == '__main__':
    model_name = "model_complex"
    exp_name = "Experiment_KF_lesioning"
    base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")

    model_params = model_params_from_cfg(load_model_cfg_file(model_name.replace("model_", "")))
    description = "Reducing the drive to the neural populations of the Kolliker-Fuse from the full value down to zero."

    config_dict = dict()
    config_dict["model_params"] = model_params
    config_dict["description"] = description
    config_dict["varied_params"] = ["self.model.populations[self.model.pnames.index(\"KF_phasic\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"KF_gate\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Exp\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Insp\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"RampI\")].drive",
                                    "self.model.W[self.model.pnames.index(\"KF_phasic\"),self.model.pnames.index(\"Sensory_relay\")]"]
    # runs the model 50 times, so it takes quite a while!
    config_dict["param_points"] = list(zip(np.linspace(0, 0.35, 50),
                                           np.linspace(0, 0.25, 50),
                                           np.linspace(0, 0.35, 50),
                                           np.linspace(0.30, 0.35, 50),
                                           np.linspace(0.30, 0.35, 50),
                                           np.linspace(0.0, 0.03, 50)))[::-1]
    config_dict["protocol_dict"] = {"Protocol_noSI": {"T": 30},
                                    "Protocol_longSI": {"T": 10, "amp": 0.45},
                                    "Protocol_shortSI": {"interim_T": 10, "amp": 0.45, "stim_duration": 0.1},
                                    "Protocol_LongShortSI": {"noSI_T": 3, "longSI_T": 10,
                                                             "interim_T": 3, "amp": 0.45, "stim_duration": 0.1,
                                                             "n_stim": 1}}
    exp = Experiment(config_dict, base_folder)
    exp.run(rerun=True)
    exp.plot_traces(param_names_tuple="Drives to KF populations: ")
    exp.analyse_data()