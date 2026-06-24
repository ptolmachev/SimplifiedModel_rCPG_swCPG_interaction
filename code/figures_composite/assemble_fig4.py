"""Assemble the full Fig. 4 composite (experiment + simulation, one shared x-axis).

Panels, stacked top-to-bottom, all sharing the same time grid:
  A  experimental PNA / VNA            (SLN stimulation recording)
  B  simulated PNA / VNA
  C  simulated neural-population traces

Alignment principle
-------------------
Every signal is resampled onto a common 40000-point grid scaled to 1 ms/point.
The experiment is windowed using its stored stimulus markers (15 s before stim
onset + 40 s total), which pins the experimental stim onset to grid index 15000
-- exactly where the simulation's 15 s pre-stim window ends. The phase bands
(I / E1 / E2) are placed on the experimental reference breath; the simulation is
slid under those same bands with the single knob SETTLE_TIME. Both the bands and
the stim window are drawn identically on every panel, so phases line up across
experiment and simulation by construction.
"""
import os
import pickle
import numpy as np
from scipy.signal import savgol_filter as sg
from matplotlib import pyplot as plt
from rCPGswCPG.plotting_figures.plotting_utils import run_sim, resample
from rCPGswCPG.utils.gen_utils import get_project_root

# ------------------------------------------------------------------ tunables
L = 40000                       # common grid length (1 ms / point -> 40 s window)
STIM_START, STIM_END = 15000, 25000     # 10 s SLN stim window on the grid

# Simulation alignment: slides the simulated breath train under the phase bands.
SETTLE_TIME = 30.52             # s

# Phase bands of the reference breath (grid indices), placed on the experimental
# inspiratory burst: t1/t2/t4 = burst onset / offset / next onset, t3 = E midpoint.
PHASE_TIMES = [5230, 5659, 6816, 7973]

# Experimental recording (CH10 = PNA, CH15 = VNA); override root via EXP_DATA_ROOT.
EXP_ROOT = os.environ.get(
    "EXP_DATA_ROOT",
    "/Users/pt1290/Documents/GitHub/Experimental_data_processing_rCPG_SwCPG")
EXP_REC = os.path.join(EXP_ROOT, "data", "sln_prc_filtered", "2019-08-22_15-59-55_t1")


# ------------------------------------------------------------------ data prep
def load_experiment():
    """Load and grid-align experimental PNA/VNA.

    Returns dict {'PNA','VNA'} of 1-D arrays on the common 40000-pt grid, windowed
    so the recorded stim onset lands at grid index STIM_START (15000).
    """
    grid = np.linspace(0, L, L)
    out = {}
    for ch, name in [("CH10", "PNA"), ("CH15", "VNA")]:
        d = pickle.load(open(os.path.join(EXP_REC, f"100_{ch}_processed.pkl"), "rb"))
        sig, fr, s0 = np.asarray(d["signal"]), d["fr"], d["stim_start"]
        pre, total = int(15 * fr), int(40 * fr)         # 15 s pre-stim, 40 s total
        a = int(round(s0)) - pre
        out[name] = resample(sg(sig[a:a + total], 51, 3), grid, L)
    return out


def load_simulation():
    """Run the eupneic simulation and grid-align PNA/VNA + the 9 population traces.

    Returns (data, pnames) where data maps each signal name to a 40000-pt array.
    """
    rec = run_sim(mode='eupneic', settle_time=SETTLE_TIME)
    grid = np.linspace(0, L, L)
    pnames = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2",
              "Sensory_relay", "KF_phasic", "KF_gate"]
    data = {k: resample(rec[k], grid, L) for k in (["PNA", "VNA"] + pnames)}
    return data, pnames


# ------------------------------------------------------------------ plotting
def overlay(ax):
    """Draw the shared stim window and phase bands/dividers on one axis."""
    ax.axvspan(STIM_START, STIM_END, color='k', alpha=0.05)
    for x in (STIM_START, STIM_END):
        ax.axvline(x, c='k', ls='--', lw=1)
    band_colors = ['r', 'b', 'g']
    for i in range(len(PHASE_TIMES) - 1):
        ax.axvspan(PHASE_TIMES[i], PHASE_TIMES[i + 1], color=band_colors[i % 3], alpha=0.06)
    for x in PHASE_TIMES:
        ax.axvline(x, c='k', ls='--', lw=1)


def main():
    exp = load_experiment()
    sim, pnames = load_simulation()

    trace_labels = ["I (early-I)", "ramp-I", "E (post-I)", "late-E",
                    r"$Sw_{1}$", r"$Sw_{2}$", "Sensory relay",
                    "pontine post-I", "sw.-gate control"]

    # (group_letter, key, source, color, right-label)
    rows = [
        ("A", "PNA", exp, 'r', "PNA, experiment"),
        ("A", "VNA", exp, 'b', "VNA, experiment"),
        ("B", "PNA", sim, 'r', "PNA, simulation"),
        ("B", "VNA", sim, 'b', "VNA, simulation"),
    ]
    rows += [("C", k, sim, 'k', lbl) for k, lbl in zip(pnames, trace_labels)]

    # nerve panels a touch taller than the population traces
    hr = [1.5, 1.5, 1.5, 1.5] + [1.0] * len(pnames)
    fig, axs = plt.subplots(len(rows), 1, figsize=(15, 0.55 * sum(hr)),
                            gridspec_kw={'height_ratios': hr})
    plt.subplots_adjust(hspace=0.0)   # seamless stacked panels (continuous stim/phase shading)

    active_ref = max(float(np.nanmax(sim[k])) for k in pnames)   # scale for near-silent traces

    prev_group = None
    for ax, (grp, key, src, col, lbl) in zip(axs, rows):
        y = src[key]
        ax.plot(y, lw=1.6, c=col)
        overlay(ax)
        ax.set_xlim(0, L)
        # leave clear headroom so the peak fills ~2/3 of the (seamless) panel and never
        # crowds the panel above; near-silent traces use the figure's active scale.
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if ymax - ymin > 0.05:
            amp = ymax - ymin
            ax.set_ylim(ymin - 0.05 * amp, ymax + 0.45 * amp)
        else:
            ax.set_ylim(-0.05 * active_ref, 1.45 * active_ref)
        ax.text(1.012, 0.5, lbl, transform=ax.transAxes, va='center', ha='left', fontsize=11)
        ax.axis('off')
        # panel letter at the first row of each group
        if grp != prev_group:
            ax.text(-0.01, 1.0, grp, transform=ax.transAxes, va='top', ha='right',
                    fontsize=18, fontweight='bold')
            prev_group = grp

    # top annotations: phase labels, stim label, scale bar (on the very top axis)
    top = axs[0]
    phase_names = ["I", r"$E_1$", r"$E_2$"]
    for i, nm in enumerate(phase_names):
        xc = 0.5 * (PHASE_TIMES[i] + PHASE_TIMES[i + 1])
        top.text(xc, 1.15, nm, transform=top.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=12)
    top.text(0.5 * (STIM_START + STIM_END), 1.15, "10 s SLN stimulation",
             transform=top.get_xaxis_transform(), ha='center', va='bottom', fontsize=12)
    # 2.5 s scale bar (2500 grid pts) at bottom-right of panel A
    x0 = L - 3000
    top.plot([x0, x0 + 2500], [-0.15, -0.15], transform=top.get_xaxis_transform(),
             c='k', lw=2, clip_on=False)
    top.text(x0 + 1250, -0.30, "2.5 s", transform=top.get_xaxis_transform(),
             ha='center', va='top', fontsize=10)

    out = os.path.join(get_project_root(), "img", "Fig4_composite.pdf")
    fig.savefig(out, bbox_inches='tight', transparent=False)
    plt.close(fig)
    print("wrote", out)


if __name__ == '__main__':
    main()
