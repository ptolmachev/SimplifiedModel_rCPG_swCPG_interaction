# a script which generates parameters for the "model 4"
import numpy as np
import pickle
from rCPGswCPG.utils.gen_utils import get_project_root
import os

def generate_paramconfig_full_model():
    params_folder = os.path.join(get_project_root(), "data", "model_params")
    os.makedirs(params_folder, exist_ok=True)
    dt = 0.5

    pnames = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
    N = len(pnames)

    W = np.zeros((N, N))
    W[pnames.index("Sensory_relay"), pnames.index("Sw1")] = 0.12
    W[pnames.index("Sensory_relay"), pnames.index("Sw2")] = 0.07
    W[pnames.index("Sensory_relay"), pnames.index("KF_phasic")] = 0.03
    W[pnames.index("Sensory_relay"), pnames.index("Insp")] = -0.4
    W[pnames.index("Sensory_relay"), pnames.index("RampI")] = -0.4
    W[pnames.index("Sensory_relay"), pnames.index("LateExp")] = -0.4 #was -0.8?

    W[pnames.index("Sw1"), pnames.index("Sw2")] = -0.9
    W[pnames.index("Sw1"), pnames.index("Insp")] = -0.8
    W[pnames.index("Sw1"), pnames.index("RampI")] = -0.8
    W[pnames.index("Sw1"), pnames.index("LateExp")] = -0.2
    W[pnames.index("Sw2"), pnames.index("Sw1")] = -0.65
    W[pnames.index("Insp"), pnames.index("Exp")] = -0.2
    W[pnames.index("Insp"), pnames.index("LateExp")] = -1.00
    W[pnames.index("Insp"), pnames.index("Sw1")] = -0.01
    W[pnames.index("Insp"), pnames.index("KF_phasic")] = -0.35
    W[pnames.index("Insp"), pnames.index("RampI")] = -0.22
    W[pnames.index("Exp"), pnames.index("Insp")] = -0.5
    W[pnames.index("Exp"), pnames.index("RampI")] = -1.00
    W[pnames.index("Exp"), pnames.index("LateExp")] = -1.3
    W[pnames.index("Exp"), pnames.index("KF_phasic")] = 0.03
    W[pnames.index("LateExp"), pnames.index("Exp")] = -0.01
    W[pnames.index("LateExp"), pnames.index("KF_phasic")] = -0.65
    W[pnames.index("LateExp"), pnames.index("Insp")] = -0.7
    W[pnames.index("LateExp"), pnames.index("RampI")] = -0.75
    W[pnames.index("KF_phasic"), pnames.index("Exp")] = 0.03
    W[pnames.index("KF_gate"), pnames.index("Sw1")] = -0.15

    tau = np.zeros(N)
    tau[pnames.index("Insp")] = 5000
    tau[pnames.index("RampI")] = 5000
    tau[pnames.index("Exp")] = 5000
    tau[pnames.index("LateExp")] = 5000
    tau[pnames.index("Sw1")] = 1500
    tau[pnames.index("Sw2")] = 1500
    tau[pnames.index("Sensory_relay")] = 17500
    tau[pnames.index("KF_phasic")] = 10000
    tau[pnames.index("KF_gate")] = 5000

    drives_misc = np.zeros((3, N))
    # CO2 level drive to BotC/PreBotC
    drives_misc[0, pnames.index("Insp")] = 0.3
    drives_misc[0, pnames.index("RampI")] = 0.3
    drives_misc[0, pnames.index("LateExp")] = 0.45
    drives_misc[0, pnames.index("KF_phasic")] = 0.05

    # wakefullness drive
    drives_misc[1, pnames.index("Sw1")] = 0.19
    drives_misc[1, pnames.index("Sw2")] = 0.35
    drives_misc[1, pnames.index("KF_gate")] = 0.25

    #drive from pons
    drives_misc[2, pnames.index("Exp")] = 0.35
    drives_misc[2, pnames.index("KF_phasic")] = 0.35
    drives_misc[2, pnames.index("Insp")] = 0.05
    drives_misc[2, pnames.index("RampI")] = 0.05


    param_dict = dict()
    param_dict["pnames"] = pnames
    param_dict["N"] = len(pnames)
    param_dict["dt"] = dt
    param_dict["tau"] = tau
    param_dict["W"] = W
    param_dict["drives_misc"] = drives_misc
    pickle.dump(param_dict, open(os.path.join(params_folder, "params_model_complex.pkl"), 'wb+'))
    return None

if __name__ == '__main__':
    generate_paramconfig_full_model()








