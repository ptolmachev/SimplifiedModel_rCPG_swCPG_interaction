import pickle
from src.utils.gen_utils import get_project_root
from src.Experiment import Experiment
import numpy as np
import os

# see class Experiment for more details
# Runs the experiment only if the corresponding folder is not empty

if __name__ == '__main__':
    model_name = "model_complex"
    exp_name = "Experiment_KF_lesioning"
    param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
    base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")

    model_params = pickle.load(open(os.path.join(f"{param_folder}", f"params_{model_name}.pkl"), 'rb+'))
    description = "Reducing the drive to the neural populations of the Kolliker-Fuse from the full value down to zero."

    config_dict = dict()
    config_dict["model_params"] = model_params
    config_dict["description"] = description
    config_dict["varied_params"] = ["self.model.populations[self.model.pnames.index(\"KF_phasic\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Exp\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Insp\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"RampI\")].drive",
                                    "self.model.W[self.model.pnames.index(\"Sensory_relay\"),self.model.pnames.index(\"KF_phasic\")]"]
    # runs the model 50 times, so it takes quite a while!
    config_dict["param_points"] = list(zip(np.linspace(0, 0.35, 50),
                                           np.linspace(0, 0.35, 50),
                                           np.linspace(0.30, 0.35, 50),
                                           np.linspace(0.30, 0.35, 50),
                                           np.linspace(0.0, 0.03, 50)))[::-1]
    config_dict["protocol_dict"] = {"Protocol_noSI" : {"T": 30 },
                                    "Protocol_longSI": {"T": 10, "amp" : 0.4},
                                    "Protocol_shortSI": {"interim_T": 10, "amp" : 0.4, "stim_duration" : 0.1}}
    exp = Experiment(config_dict, base_folder)
    exp.run()
    exp.plot_traces(param_names_tuple="Drives to KF populations: ")
    exp.analyse_data()