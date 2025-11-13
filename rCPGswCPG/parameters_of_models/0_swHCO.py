# a script which generates parameters for the "model 1"
import numpy as np
import pickle
from rCPGswCPG.utils.gen_utils import get_project_root, create_dir_if_not_exist
import os

def generate_paramconfig_full_model():
    params_folder = os.path.join(get_project_root(), "data", "model_params")
    create_dir_if_not_exist(params_folder)
    dt = 0.2

    pnames = ["Sw1", "Sw2"]
    N = len(pnames)
    W = np.zeros((N, N))

    W[pnames.index("Sw1"), pnames.index("Sw2")] = -0.9
    W[pnames.index("Sw2"), pnames.index("Sw1")] = -0.65

    tau = np.zeros(N)
    tau[pnames.index("Sw1")] = 1500
    tau[pnames.index("Sw2")] = 1500

    drives_misc = np.zeros((1, N))
    # wakefullness drive
    drives_misc[0, pnames.index("Sw1")] = 0.175
    drives_misc[0, pnames.index("Sw2")] = 0.35

    param_dict = dict()
    param_dict["pnames"] = pnames
    param_dict["N"] = len(pnames)
    param_dict["dt"] = dt
    param_dict["tau"] = tau
    param_dict["W"] = W
    param_dict["drives_misc"] = drives_misc
    pickle.dump(param_dict, open(os.path.join(params_folder, "params_swHCO.pkl"), 'wb+'))
    return None

if __name__ == '__main__':
    generate_paramconfig_full_model()








