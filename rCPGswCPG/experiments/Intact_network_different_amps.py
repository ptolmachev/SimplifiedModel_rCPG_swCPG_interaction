import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
from rCPGswCPG.Network_ import firing_rate
from rCPGswCPG.construct_model import construct_model
from rCPGswCPG.protocols.protocols import run_full_protocol, run_KF_inhibited_protocol
from rCPGswCPG.utils.gen_utils import get_project_root, create_dir_if_not_exist
from tqdm.auto import tqdm

#test
model_name = "model_complex"
W_Insp_Sw1 = 0.005
exp_name = f"Intact_network_different_amplitude_stim_{W_Insp_Sw1}"
param_folder = os.path.join(f"{get_project_root()}", "data", "model_params")
base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")
create_dir_if_not_exist(os.path.join(base_folder, "runs"))
model_params = pickle.load(open(os.path.join(f'{param_folder}', f'params_{model_name}.pkl'), 'rb+'))
model = construct_model(model_params)
external_inputs = np.zeros(model_params["N"])
pnames = model_params["pnames"]
#modify parameters if needed

for i, amp in tqdm(enumerate(np.linspace(0, 0.4, 21))):
    # for protocols in ["Protocol_noSI", "Protocol_longSI", "Protocol_shortSI"]
    run_full_protocol(model, amp=amp)
    v_history = model.get_raw_history()
    fr_history = firing_rate(v_history)
    t = (model_params['dt'] * np.arange(fr_history.shape[0]) / 1000)  # in sec

    recordings = dict()
    recordings["protocol_runs"] = dict()
    file_name = f"{str.zfill(str(i), 3)}_recordings_{np.round(amp,2)}.pkl"
    # run experimental_protocols
    recordings["protocol_runs"]["full_protocol"] = model.get_recordings()
    save_to = os.path.join(base_folder, "runs", file_name)
    pickle.dump(recordings, open(save_to, "wb+"))
