import pickle
from src.utils.gen_utils import get_project_root
from src.Experiment import Experiment
import numpy as np
import os

# see class Experiment for more details
# Runs the experiment only if the corresponding folder is not empty

if __name__ == '__main__':
    model_name = "model_complex"
    exp_name = "Experiment_arousal"
    param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
    base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")

    model_params = pickle.load(open(os.path.join(f"{param_folder}", f"params_{model_name}.pkl"), 'rb+'))
    description = "Reducing the drive to the neural populations of swCPG and sw. gate control (KF_gate) from the full value down to zero."

    config_dict = dict()
    config_dict["model_params"] = model_params
    config_dict["description"] = description
    config_dict["varied_params"] = ["self.model.populations[self.model.pnames.index(\"Sw1\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Sw2\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"KF_gate\")].drive"]
    # runs the model 51 times, so it takes quite a while!
    N = 51
    config_dict["param_points"] = list(zip(np.linspace(0.0, 0.19*2, N),
                                           np.linspace(0.0, 0.35*2, N),
                                           np.linspace(0.0, 0.25*2, N)))[::-1]
    config_dict["protocol_dict"] = {"Protocol_noSI" : {"T": 30 },
                                    "Protocol_longSI": {"T": 10, "amp" : 0.4},
                                    "Protocol_shortSI": {"interim_T": 10, "amp" : 0.4, "stim_duration" : 0.1}}
    exp = Experiment(config_dict, base_folder)
    exp.run()
    exp.plot_traces(param_names_tuple="Drives to sw. gate control and swCPG populations: ")
    exp.analyse_data()