'''
Prepares the data necessary for the PRC extraciton
the actual PRC extraction goes in PRC_estimation project ('PRC_extraction_CPGswCPG')
'''

from rCPGswCPG.Network_ import firing_rate
from rCPGswCPG.protocols.protocols import run_full_protocol
from rCPGswCPG.utils.sp_utils import *
from rCPGswCPG.construct_model import construct_model
from rCPGswCPG.utils.gen_utils import create_dir_if_not_exist, get_project_root, put
import numpy as np
import pickle
from tqdm.auto import tqdm
import os

def run_model_apply_stimulus(model, t_stim_start, stim_duration, stim_amp, t_stop):
    external_inputs = np.zeros(model.N)
    pnames = model.populations
    model.run(t_stim_start, input=put(external_inputs, pnames.index("Sensory_relay"), 0))
    model.run(stim_duration, input=put(external_inputs, pnames.index("Sensory_relay"), stim_amp))
    model.run(t_stop - (t_stim_start + stim_duration), input=put(external_inputs, pnames.index("Sensory_relay"), 0))
    recordings = model.get_recordings()
    return recordings

def get_stimuli_schedule(model_params, n_stim, t_settle, t_stop):
    '''
    To apply the stimulus throughout the phase one has to define the period and define the timings, which should be spread
    out evenly
    '''
    data_folder = os.path.join(get_project_root(), "data")
    dt = model_params["dt"]
    t_settle_inds = int(t_settle * 1000 / dt)
    model = construct_model(model_params)
    # run the model without any stimulus for t_stop time
    recordings = run_model_apply_stimulus(model=model, t_stim_start=0, stim_duration=0, stim_amp=0, t_stop=t_stop)
    # start from the end of expiration (begin of inspiration)
    pnames = model.populations
    Insp = recordings["fr_history"][t_settle_inds:, pnames.index("Insp")]
    t = recordings["t"][t_settle_inds:] * 1000 # to be in ms rather than in sec
    T = get_period(t, Insp)
    t_start_insp_inds, t_end_insp_inds = (get_insp_starts_and_ends(Insp))
    t_start_insp = (t_start_insp_inds + t_settle_inds) * dt
    t_end_insp = (t_end_insp_inds + t_settle_inds) * dt
    # shifts in ms
    stim_times = np.array([T * i / n_stim for i in range(n_stim)])

    stim_schedule_data = dict()
    stim_schedule_data["T"] = T
    stim_schedule_data["t_start_insp"] = t_start_insp
    stim_schedule_data["t_end_insp"] = t_end_insp
    stim_schedule_data["stim_times"] = stim_times
    #save the stimuli schedule to a separate file
    path_to_save = os.path.join(data_folder, "auxiliary_data", f"stim_schedule_{model_name}.pkl")
    pickle.dump(stim_schedule_data, open(path_to_save, "wb+"))
    return None

def run_simulations(experiment_params, model_params, stim_schedule_data, folder_save_to):
    stim_duration = experiment_params["stim_duration"]
    stim_amp = experiment_params["stim_amp"]
    t_stop = experiment_params["t_stop"]
    N_cycles = experiment_params["N_cycles"]

    t_start_insp = stim_schedule_data["t_start_insp"]
    stim_times = stim_schedule_data["stim_times"]
    T = stim_schedule_data["T"]

    t_insp_s = t_start_insp[:N_cycles] # run the stim application over N_cycles cycles
    for i in tqdm(range(len(stim_times))[::-1]):
        for j in range(len(t_insp_s)):
            stim_time = stim_times[i]
            t_stim_start = int(t_insp_s[j] + stim_time) / 1000 # in s rather than in ms
            # create and run a model
            model = construct_model(model_params)
            recordings = run_model_apply_stimulus(model=model, t_stim_start=t_stim_start,
                                                  stim_duration=stim_duration,
                                                  stim_amp=stim_amp, t_stop=t_stop)
            recording_data = dict()
            recording_data['signals'] = recordings["fr_history"]
            recording_data['t'] = recordings["t"]
            recording_data['population_names'] = model.populations
            recording_data['dt'] = model.dt
            recording_data['phase'] = np.round((2 * np.pi) * (i / len(stim_times)), 2)
            recording_data['T'] = T
            recording_data['t_stim_start'] = t_stim_start
            recording_data['stim_duration'] = stim_duration
            recording_data['stim_amp'] = stim_amp

            # to see if everything is alright
            # VNA_components = {"KF_phasic": 1.0, "Sw1": 0.6, "Insp": 0.65}
            # pnames = ["Insp", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
            # fig, axes = plot_data(recordings["t"], recordings["fr_history"], pnames, pnames, VNA_components)
            # plt.show(block=True)
            # plt.close(fig)

            pickle.dump(recording_data,
                        open(os.path.join(folder_save_to, f"run_{amp}_{stim_duration}_{recording_data['phase']}_trial_{j}.pkl"),
                             "wb+"))
    return None

if __name__ == '__main__':
    data_path = os.path.join(get_project_root(), "data")
    img_path = os.path.join(get_project_root(), "img")

    experiment_params = {}
    experiment_params["t_stop"] = t_stop = 75 #s
    experiment_params["n_stim"] = n_stim = 100
    experiment_params["t_settle"] = t_settle = 20 #s
    experiment_params["N_cycles"] = 5
    model_name = "model_complex"
    stim_descriptor = 'short'
    stim_amps = [0.45]
    stim_durations = [0.25]

    model_param_folder = os.path.join(f'{get_project_root()}', 'data', 'model_params')
    model_params = pickle.load(open(os.path.join(f'{model_param_folder}', f'params_{model_name}.pkl'), 'rb+'))
    pnames = model_params["pnames"]

    #if you need apneusis, modify the parameters here
    # KF_populations = ["KF_gate", "KF_phasic"]
    # for KF_pop in KF_populations:
    #     model_params["drives_misc"][0, pnames.index(KF_pop)] = 0
    #     model_params["drives_misc"][1, pnames.index(KF_pop)] = 0
    #     model_params["drives_misc"][2, pnames.index(KF_pop)] = 0
    #     for name in pnames:
    #         model_params["W"][pnames.index(name), pnames.index(KF_pop)] = 0.0
    #         model_params["W"][pnames.index(KF_pop), pnames.index(name)] = 0.0
    # model_params["drives_misc"][0, pnames.index("Exp")] = 0
    # model_params["drives_misc"][1, pnames.index("Exp")] = 0
    # model_params["drives_misc"][2, pnames.index("Exp")] = 0
    # model_params["drives_misc"][0, pnames.index("Insp")] = 0.3
    # model_params["drives_misc"][1, pnames.index("Insp")] = 0
    # model_params["drives_misc"][2, pnames.index("Insp")] = 0
    # model_params["drives_misc"][0, pnames.index("RampI")] = 0.3
    # model_params["drives_misc"][1, pnames.index("RampI")] = 0
    # model_params["drives_misc"][2, pnames.index("RampI")] = 0

    # load the schedule of the stimuli
    get_stimuli_schedule(model_params, n_stim, t_settle, t_stop)
    load_schedule_from = os.path.join(data_path, "auxiliary_data", f"stim_schedule_{model_name}.pkl")
    stim_schedule_data = pickle.load(open(load_schedule_from, 'rb+'))

    root_folder_save_recordings_to = os.path.join(data_path, "experiments", "PRC_experiments",
                                       model_name, f"PRC_{stim_descriptor}_stim")
    create_dir_if_not_exist(root_folder_save_recordings_to)

    for stim_duration in stim_durations:
        for amp in stim_amps:
            experiment_params["stim_amp"] = amp
            experiment_params["stim_duration"] = stim_duration
            print(amp, stim_duration)
            folder_recordings = os.path.join(root_folder_save_recordings_to, f"num_run_{stim_descriptor}_stim_{amp}_{stim_duration}")
            create_dir_if_not_exist(folder_recordings)
            run_simulations(experiment_params, model_params, stim_schedule_data, folder_recordings)
