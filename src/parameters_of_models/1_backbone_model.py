# a script which generates parameters for the "model 1"
import numpy as np
import pickle
from src.utils.gen_utils import get_project_root, create_dir_if_not_exist
import os

def generate_paramconfig_full_model():

    params_folder = os.path.join(get_project_root(), "data", "model_params")
    create_dir_if_not_exist(params_folder)
    dt = 0.2

    pnames = ["Insp", "Exp", "Sw1", "Sw2", "Sensory_relay"]
    N = len(pnames)

    W = np.zeros((N, N))
    W[pnames.index("Sensory_relay"), pnames.index("Sw1")] = 0.12
    W[pnames.index("Sensory_relay"), pnames.index("Sw2")] = 0.07
    W[pnames.index("Sensory_relay"), pnames.index("Insp")] = -0.40
    W[pnames.index("Sw1"), pnames.index("Sw2")] = -0.9
    W[pnames.index("Sw1"), pnames.index("Insp")] = -0.8
    W[pnames.index("Sw2"), pnames.index("Sw1")] = -0.65
    W[pnames.index("Insp"), pnames.index("Exp")] = -0.3
    W[pnames.index("Insp"), pnames.index("Sw1")] = -0.01
    W[pnames.index("Exp"), pnames.index("Insp")] = -0.7

    tau = np.zeros(N)
    tau[pnames.index("Insp")] = 5000
    tau[pnames.index("Exp")] = 5000
    tau[pnames.index("Sw1")] = 1500
    tau[pnames.index("Sw2")] = 1500
    tau[pnames.index("Sensory_relay")] = 17500

    drives_misc = np.zeros((3, N))
    # CO2 level drive to BotC/PreBotC
    drives_misc[0, pnames.index("Insp")] = 0.3

    # wakefullness drive
    drives_misc[1, pnames.index("Sw1")] = 0.16
    drives_misc[1, pnames.index("Sw2")] = 0.35

    #drive from pons
    drives_misc[2, pnames.index("Exp")] = 0.35
    drives_misc[2, pnames.index("Insp")] = 0.05

    param_dict = dict()
    param_dict["pnames"] = pnames
    param_dict["N"] = len(pnames)
    param_dict["dt"] = dt
    param_dict["tau"] = tau
    param_dict["W"] = W
    param_dict["drives_misc"] = drives_misc
    pickle.dump(param_dict, open(os.path.join(params_folder, "params_model_backbone.pkl"), 'wb+'))
    return None

if __name__ == '__main__':
    generate_paramconfig_full_model()








