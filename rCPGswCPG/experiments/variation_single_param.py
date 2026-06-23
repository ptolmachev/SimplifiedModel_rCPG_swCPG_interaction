import pickle
from rCPGswCPG.utils.gen_utils import get_project_root
from rCPGswCPG.utils.utils import *
from rCPGswCPG.Experiment import Experiment
import numpy as np
import os
import pickle


if __name__ == '__main__':
    model_name = "model_complex"
    #weights

    parameters_to_vary = [
        "self.model.W[self.model.pnames.index(\"Sw2\"),self.model.pnames.index(\"Sw1\")]",
        "self.model.W[self.model.pnames.index(\"Sw1\"),self.model.pnames.index(\"Sw2\")]",
        "self.model.populations[self.model.pnames.index(\"Sw1\")].drive",
        "self.model.populations[self.model.pnames.index(\"Sw2\")].drive",
        "self.model.populations[self.model.pnames.index(\"Sw1\")].tau_m",
        "self.model.populations[self.model.pnames.index(\"Sw2\")].tau_m",
        "self.model.W[self.model.pnames.index(\"Sw1\"),self.model.pnames.index(\"Sensory_relay\")]",
        "self.model.W[self.model.pnames.index(\"Sw2\"),self.model.pnames.index(\"Sensory_relay\")]",
        "self.model.W[self.model.pnames.index(\"Insp\"),self.model.pnames.index(\"Sensory_relay\")]",
        "self.model.W[self.model.pnames.index(\"LateExp\"),self.model.pnames.index(\"Sensory_relay\")]",
        "self.model.W[self.model.pnames.index(\"KF_phasic\"),self.model.pnames.index(\"Sensory_relay\")]",
        "self.model.W[self.model.pnames.index(\"Insp\"),self.model.pnames.index(\"Sw1\")]",
        "self.model.W[self.model.pnames.index(\"LateExp\"),self.model.pnames.index(\"Sw1\")]",
        "self.model.W[self.model.pnames.index(\"Exp\"),self.model.pnames.index(\"Insp\")]",
        "self.model.W[self.model.pnames.index(\"LateExp\"),self.model.pnames.index(\"Insp\")]",
        "self.model.W[self.model.pnames.index(\"KF_phasic\"),self.model.pnames.index(\"Insp\")]",
        "self.model.W[self.model.pnames.index(\"Insp\"),self.model.pnames.index(\"Exp\")]",
        "self.model.W[self.model.pnames.index(\"LateExp\"),self.model.pnames.index(\"Exp\")]",
        "self.model.W[self.model.pnames.index(\"KF_phasic\"),self.model.pnames.index(\"Exp\")]",
        "self.model.W[self.model.pnames.index(\"Exp\"),self.model.pnames.index(\"LateExp\")]",
        "self.model.W[self.model.pnames.index(\"Insp\"),self.model.pnames.index(\"LateExp\")]",
        "self.model.W[self.model.pnames.index(\"KF_phasic\"),self.model.pnames.index(\"LateExp\")]",
        "self.model.W[self.model.pnames.index(\"Exp\"),self.model.pnames.index(\"KF_phasic\")]",
        "self.model.W[self.model.pnames.index(\"Sw1\"),self.model.pnames.index(\"KF_gate\")]",
        "self.model.populations[self.model.pnames.index(\"Insp\")].drive",
        "self.model.populations[self.model.pnames.index(\"Exp\")].drive",
        "self.model.populations[self.model.pnames.index(\"LateExp\")].drive",
        "self.model.populations[self.model.pnames.index(\"KF_gate\")].drive",
        "self.model.populations[self.model.pnames.index(\"KF_phasic\")].drive",
        "self.model.populations[self.model.pnames.index(\"Insp\")].tau_m",
        "self.model.populations[self.model.pnames.index(\"Exp\")].tau_m",
        "self.model.populations[self.model.pnames.index(\"LateExp\")].tau_m",
        "self.model.populations[self.model.pnames.index(\"KF_gate\")].tau_m",
        "self.model.populations[self.model.pnames.index(\"KF_phasic\")].tau_m",
        "self.model.populations[self.model.pnames.index(\"Sensory_relay\")].tau_m"
    ] #35 parameters

    print(len(parameters_to_vary))
    for k, parameter_to_vary in enumerate(parameters_to_vary):
        hr_param_name = get_short_name(parameter_to_vary)
        exp_name = f"Varying_{hr_param_name}"
        param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
        base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")
        model_params = pickle.load(open(os.path.join(f"{param_folder}", f"params_{model_name}.pkl"), 'rb+'))
        description = f"Varying the {hr_param_name} in the range (0.1 - 1.5) * param_val; param_val={get_val_of_param(parameter_to_vary, model_params)}"

        config_dict = dict()
        config_dict["model_params"] = model_params
        config_dict["description"] = description
        config_dict["varied_params"] = [parameter_to_vary]
        param_val = get_val_of_param(parameter_to_vary, model_params)
        config_dict["param_points"] = np.linspace(0.1 * param_val, 1.5 * param_val, 25).reshape(-1, 1)
        config_dict["protocol_dict"] = {"Protocol_noSI" : {"T": 30},
                                        "Protocol_longSI" : {"T": 10, "amp": 0.45},
                                        "Protocol_shortSI": {"interim_T": 10, "amp" : 0.45, "stim_duration": 0.1},
                                        "Protocol_LongShortSI": {"noSI_T": 10, "longSI_T": 10,
                                                                 "interim_T": 10, "amp": 0.45, "stim_duration": 0.1, "n_stim":1}}
        exp = Experiment(config_dict, base_folder)
        exp.run(rerun=False)
        exp.plot_traces(param_names_tuple=hr_param_name)
        exp.analyse_data()
        exp.plot_analytics_1d()





