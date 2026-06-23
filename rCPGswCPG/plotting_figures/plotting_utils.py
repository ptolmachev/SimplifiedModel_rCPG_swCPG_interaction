import pickle
import numpy as np
import os
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from rCPGswCPG.Network_ import firing_rate
from rCPGswCPG.Network import construct_model
from rCPGswCPG.utils.gen_utils import get_project_root

# ------- helpers
def idx_map(names): return {n: i for i, n in enumerate(names)}

def zero_like(t): return np.zeros_like(t)

def resample(x, t_out, L): return interp1d(np.linspace(0, L, len(x)), x)(t_out)

def apply_mode(model, idx, mode):
    if mode == 'eupneic':
        return
    if mode == 'apneustic':
        kf = ['KF_gate', 'KF_phasic']
        for name in kf:
            j = idx[name]
            model.populations[j].drive = 0.0
            model.W[:, j] = 0.0
            model.W[j, :] = 0.0
        model.populations[idx['Exp']].drive    = 0.0
        model.populations[idx['Insp']].drive   = 0.3
        model.populations[idx['RampI']].drive  = 0.3

def run_schedule(model, idx, sensory_name, schedule):
    for dur, val in schedule:
        u = np.zeros(len(idx))
        u[idx[sensory_name]] = val
        model.run(dur, input=u)

def build_recordings(fr, pnames):
    d = {name: fr[:, i] for i, name in enumerate(pnames)}
    d['PNA'] = fr[:, pnames.index('RampI')]
    d['VNA'] = (0.75 * fr[:, pnames.index('KF_phasic')]
                + 0.90 * fr[:, pnames.index('RampI')]
                + 0.60 * fr[:, pnames.index('Sw1')])
    return d

def run_sim(mode='eupneic',
            settle_time=30.0,
            B4StimDuration=15.0,
            StimDuration=10.0,
            AfterStimDuration=15.0):
    model_name   = 'model_complex'
    param_folder = os.path.join(get_project_root(), 'data', 'model_params')
    with open(os.path.join(param_folder, f'params_{model_name}.pkl'), 'rb') as f:
        mp = pickle.load(f)

    model   = construct_model(mp)
    pnames  = mp['pnames']
    idx     = idx_map(pnames)

    apply_mode(model, idx, mode)

    # settle to limit cycle
    run_schedule(model, idx, 'Sensory_relay', [(settle_time, 0.0)])
    model.clear_history()

    # experiment schedule (easy to modify)
    run_schedule(model, idx, 'Sensory_relay',
                 [(B4StimDuration, 0.0),
                  (StimDuration, 0.45),
                  (AfterStimDuration, 0.0)])

    v, m = model.get_raw_history()
    fr = firing_rate(v)
    t_sim = mp['dt'] * np.arange(fr.shape[0]) / 1000.0  # seconds

    rec = build_recordings(fr, pnames)
    rec.update({'pnames': pnames, 't': t_sim})
    return rec

def plot_recordings(time_points, data, keys, color_map, label_map,
                    stim_start, stim_end, phase_spans=None, phase_times=None,
                    ylims=None, outpath=None, aspect_ratios=None):
    n = len(keys)
    hr = [1] * n if aspect_ratios is None else aspect_ratios
    fig_h = 0.8 * sum(hr)
    fig, axs = plt.subplots(n, 1, figsize=(15, fig_h), gridspec_kw={'height_ratios': hr})
    axs = [axs] if n == 1 else axs
    plt.subplots_adjust(wspace=0, hspace=0.25)

    z = np.zeros_like(time_points)

    for ax, name in zip(axs, keys):
        trace = data[name]
        ax.plot(trace, linewidth=2, label=label_map.get(name, name), c=color_map.get(name, 'k'))
        ax.plot(z, c='k', alpha=0.1)

        ax.axvline(float(stim_start), c='k', ls='--')
        ax.axvline(float(stim_end),   c='k', ls='--')
        ax.axvspan(float(stim_start), float(stim_end), color='k', alpha=0.05)

        if phase_times is not None:
            for t in phase_times:
                if t is not None:
                    ax.axvline(t, c='k', ls='--')
        if phase_spans is not None:
            for a, b, col, al in phase_spans:
                if a is not None:
                    ax.axvspan(a, b, color=col, alpha=al)

        if ylims is None:
            ymax = 1.05 * float(np.nanmax(trace))
            ax.set_ylim(0.0, ymax if ymax > 0 else 1.0)
        elif isinstance(ylims, tuple):
            ax.set_ylim(*ylims)
        else:
            ax.set_ylim(*ylims.get(name, (0.0, 1.0)))

        legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                           facecolor='white', edgecolor='none', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axis('off')

    if outpath:
        plt.savefig(outpath, bbox_inches='tight', transparent=True)
    plt.show(block=True)
    plt.close()

