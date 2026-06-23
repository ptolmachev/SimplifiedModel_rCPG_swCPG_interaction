from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from rCPGswCPG.plotting_figures.plotting_utils import run_sim, resample, plot_recordings
from rCPGswCPG.utils.gen_utils import get_project_root

if __name__ == '__main__':
    rerun = True
    modes = ['eupneic', 'apneustic']

    recordings = {}
    data_dir   = os.path.join(get_project_root(), 'data')
    for m in modes:
        path = os.path.join(data_dir, f'SimApnVsSimEupn_recordings_{m}.pkl')
        if rerun or not os.path.exists(path):
            rec = run_sim(mode=m)
            with open(path, 'wb') as f:
                pickle.dump(rec, f)
        with open(path, 'rb') as f:
            recordings[m] = pickle.load(f)

    # plotting setup (easy to tweak)
    interp_length = 40000
    time = np.linspace(0, interp_length, interp_length)
    stim_start, stim_end = 15000, 25000

    populations       = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay", "KF_phasic", "KF_gate"]
    populations2plot  = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay"]
    motor_outputs  = ["PNA", "VNA"]
    lbls         = ["I (early-I)", "ramp-I", "E (post-I)", "late-E", r"$Sw_{1}$", r"$Sw_{2}$", "Sensory relay"]

    label_map = {k: v for k, v in zip(populations2plot, lbls)}
    label_map.update({n: n for n in motor_outputs})

    colors              = ['r', 'b', 'g', 'yellow', 'magenta', 'cyan']
    color_map_nerves    = dict(zip(motor_outputs, colors[:len(motor_outputs)]))
    color_map_traces    = {k: 'k' for k in populations2plot}

    img_folder = os.path.join(get_project_root(), 'img')

    # resample once per mode
    for m in modes:
        # resample all signals to a common time grid
        data_m = {k: resample(recordings[m][k], time, interp_length) for k in (motor_outputs + populations2plot)}
        # y-limits anchored to eupneic maxima for consistency
        yref   = {k: 1.05 * np.max(resample(recordings[m][k], time, interp_length)) for k in data_m}
        # ylims  = {k: (-0.01, yref[k] if yref[k] > 0.1 else 1.0) for k in data_m}
        ylims = {k: (-0.01, yref[k] if yref[k] > 0.1 else 1.0) for k in data_m}

        # phase spans (derive from phase times)
        phase_times = [4380, 4790, 6060, 6930] if m == 'eupneic' else [7940, 9240, 12800] #2670, 3960,
        spans = [(phase_times[i], phase_times[i + 1], colors[i % len(colors)], 0.05)
                 for i in range(len(phase_times) - 1)]
        print(stim_start, stim_end)
        # plotting motor outputs

        plot_recordings(time_points=time, data=data_m, keys=motor_outputs,
                        color_map=color_map_nerves, label_map={"PNA": f"PNA, {m}", "VNA": f"VNA, {m}"},
                        phase_spans=spans, phase_times=phase_times,
                        stim_start=stim_start, stim_end=stim_end,
                        ylims=ylims, outpath=os.path.join(img_folder, f'{m}_MotorOutputs_transection.pdf'))

        # plotting individual neural traces
        plot_recordings(time_points=time, data=data_m, keys=populations2plot, color_map=color_map_traces, label_map=label_map,
                        phase_spans=spans, phase_times=phase_times, stim_start=stim_start, stim_end=stim_end,
                        ylims=ylims, outpath=os.path.join(img_folder, f'{m}_traces_transection.pdf'))


