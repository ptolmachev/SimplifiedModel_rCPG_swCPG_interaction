import pickle
from src.utils.gen_utils import get_project_root, create_dir_if_not_exist
from src.Experiment import Experiment
import numpy as np
import os

if __name__ == '__main__':
    model_name = "full_model"
    exp_name = "Experiment_wakefulness"
    param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
    base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")

    model_params = pickle.load(open(os.path.join(f"{param_folder}", f"params_{model_name}.pkl"), 'rb+'))
    description = "Reducing the \"wakefulness\" drive to the neural populations."

    config_dict = dict()
    config_dict["model_params"] = model_params
    config_dict["description"] = description
    config_dict["varied_params"] = ["self.model.populations[self.model.pnames.index(\"KF_gate\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Sw1\")].drive",
                                    "self.model.populations[self.model.pnames.index(\"Sw2\")].drive"]
    N = 51 # 51 runs!
    config_dict["param_points"] = list(zip(np.linspace(0, 0.25, N), np.linspace(0, 0.25, N), np.linspace(0, 0.25, N)))
    config_dict["protocol_dict"] = {"Protocol_noSI" : {"T": 30},
                                    "Protocol_longSI": {"T": 10, "amp" : 0.45},
                                    "Protocol_shortSI": {"interim_T": 10, "amp" : 0.45, "stim_duration" : 0.25}}
    exp = Experiment(config_dict, base_folder)
    exp.run()
    exp.plot_traces(param_names_tuple="wakefulness drive to populations")
    exp.analyse_data()