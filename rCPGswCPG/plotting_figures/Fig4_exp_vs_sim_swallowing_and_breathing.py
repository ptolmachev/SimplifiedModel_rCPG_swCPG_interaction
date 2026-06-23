##plotfrom matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from rCPGswCPG.plotting_figures.plotting_utils import run_sim, resample, plot_recordings
from rCPGswCPG.utils.gen_utils import get_project_root
from scipy.signal import savgol_filter as sg


if __name__ == '__main__':
    # ---- experiment config (indices & windows) ----
    stim_start_ind, stim_end_ind = 15000, 25000
    t1, t2, t3, t4 = 6600, 7040, 8168, 9080
    experiment_ind_start, experiment_ind_end = 5600, 18900
    simulation_ind_start, simulation_ind_end = 0, -1
    rerun = True
    exp_folder = os.path.abspath(
        r"/Users/tolmach/Documents/GitHub/Exp_Data_Processing_rCPG/data/sln_prc_filtered/2019-08-22_15-59-55_t1")

    # ---- simulation (cached) ----
    sim_path = os.path.join(get_project_root(), "data", f"ExpEupnVsSimEupn_recordings_eupneic.pkl")
    if not os.path.exists(sim_path) or rerun:
        rec = run_sim(mode='eupneic', settle_time=29.37) # for the best visual match with experiment
        with open(sim_path, 'wb') as f:
            pickle.dump(rec, f)
    with open(sim_path, 'rb') as f:
        rec = pickle.load(f)

    sim_PNA = rec['PNA']
    sim_VNA = rec['VNA']
    pnames = rec['pnames']
    neural_traces = {pname: rec[pname] for pname in pnames}  # keep names and traces aligned

    # ---- experimental data (optional: lives in a separate repo; skip if absent) ----
    have_exp = os.path.exists(os.path.join(exp_folder, "100_CH10_processed.pkl"))
    if have_exp:
        exp_PNA = pickle.load(open(os.path.join(exp_folder, "100_CH10_processed.pkl"), "rb"))["signal"]
        exp_VNA = pickle.load(open(os.path.join(exp_folder, "100_CH15_processed.pkl"), "rb"))["signal"]
    else:
        print(f"[Fig4] experimental data not found at {exp_folder}; producing simulation panels only")

    # ---- unify lengths (resample to common grid) ----
    interp_length = 40000
    t_new = np.linspace(0, interp_length, interp_length)
    PNA_s = resample(sim_PNA[simulation_ind_start:simulation_ind_end], t_new, interp_length)
    VNA_s = resample(sim_VNA[simulation_ind_start:simulation_ind_end], t_new, interp_length)
    if have_exp:
        PNA_e = resample(sg(exp_PNA[experiment_ind_start:experiment_ind_end], 51, 3), t_new, interp_length)
        VNA_e = resample(sg(exp_VNA[experiment_ind_start:experiment_ind_end], 51, 3), t_new, interp_length)

    # ---- common styling ----
    times = [t1, t2, t3, t4]
    spans = [(t1, t2, 'r', 0.05), (t2, t3, 'b', 0.05), (t3, t4, 'g', 0.05)]
    colors = {'PNA': 'r', 'VNA': 'b'}
    img_folder = os.path.join(get_project_root(), "img")

    # ---- plot nerves (simulation vs experiment) ----
    data_sim = {'PNA': PNA_s, 'VNA': VNA_s}
    ylims_sim = {k: (-0.01, 1.05 * np.max(v)) for k, v in data_sim.items()}

    if have_exp:
        data_exp = {'PNA': PNA_e, 'VNA': VNA_e}
        ylims_exp = {k: (-0.01, 1.05 * np.max(v)) for k, v in data_exp.items()}
        plot_recordings(time_points=t_new, data=data_exp, keys=['PNA', 'VNA'], color_map=colors,
                        label_map={'PNA': 'PNA, experiment', 'VNA': 'VNA, experiment'},
                        phase_spans=spans, phase_times=times, stim_start=stim_start_ind, stim_end=stim_end_ind,
                        ylims=ylims_exp, outpath=os.path.join(img_folder, "eupneic_motor_outputs_experiment.pdf"))

    plot_recordings(time_points=t_new, data=data_sim, keys=['PNA', 'VNA'], color_map=colors,
                    label_map={'PNA': 'PNA, simulation', 'VNA': 'VNA, simulation'},
                    phase_spans=spans, phase_times=times, stim_start=stim_start_ind, stim_end=stim_end_ind,
                    aspect_ratios=[1, 2],
                    ylims=ylims_sim, outpath=os.path.join(img_folder, "eupneic_motor_outputs_simulation.pdf"))

    # ---- traces panel (sim) using the same generic plotter ----
    pnames_to_plot = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
    labels = ["I (early-I)", "ramp-I", "E (post-I)", "late-E", r"$Sw_{1}$", r"$Sw_{2}$", "Sensory relay", "pontine post-I", "sw.-gate control"]
    label_map_tr = {k: v for k, v in zip(pnames_to_plot, labels)}
    color_map_tr = {k: 'k' for k in pnames_to_plot}

    # resample each trace to t_new
    data_traces = {}
    for name in pnames_to_plot:
        raw = neural_traces[name][simulation_ind_start:simulation_ind_end]
        data_traces[name] = resample(raw, t_new, interp_length)

    ylims_tr = {k: (-0.01, max(1.0, 1.05 * np.max(v))) for k, v in data_traces.items()}
    print(stim_start_ind, stim_end_ind)
    
    plot_recordings(time_points=t_new,
                    data=data_traces, 
                    keys=pnames_to_plot,
                    color_map=color_map_tr,
                    label_map=label_map_tr,
                    phase_spans=spans, phase_times=times,
                    stim_start=stim_start_ind, stim_end=stim_end_ind,
                    ylims=ylims_tr, outpath=os.path.join(img_folder, "eupneic_traces_simulation.pdf"))