"""Assemble Fig. 7 composite (intact vs. lesioned model, shared stim window).

Data panels (panel A of the paper is a static circuit schematic, composited
separately):
  B  intact (eupneic) PNA / VNA
  C  lesioned (apneustic) PNA / VNA
  D  lesioned neural-population traces (pontine populations are removed by the
     lesion, so only the 7 surviving populations are shown)

Alignment principle
-------------------
All signals share the same 40000-pt, 1 ms/point grid; each simulation records
15 s pre-stim + 10 s stim + 15 s post-stim, so the SLN-stim window lands at grid
[15000, 25000] on every panel. The intact and lesioned rhythms have different
periods, so each carries its OWN phase bands (eupnea: I/E1/E2; apneusis: I/E),
placed on its reference pre-stim breath -- exactly as in the published figure.
SETTLE_TIME per mode slides each rhythm's reference breath under its bands.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from rCPGswCPG.plotting_figures.plotting_utils import run_sim, resample
from rCPGswCPG.utils.gen_utils import get_project_root

# ------------------------------------------------------------------ tunables
L = 40000
STIM_START, STIM_END = 15000, 25000

# per-mode settle times (slide each reference breath under its bands)
SETTLE = {'eupneic': 31.561, 'apneustic': 31.092}

# phase bands (grid indices), snapped to each mode's PNA reference breath:
#   eupnea  -> I / E1 / E2 (E split at expiratory midpoint)
#   apneusis-> I (prolonged plateau) / E
BANDS = {'eupneic': [4186, 4633, 5804, 6975], 'apneustic': [7539, 8852, 12549]}


# ------------------------------------------------------------------ data prep
def load_mode(mode):
    """Run a simulation for `mode` and grid-align PNA/VNA + the 7 surviving traces.

    Returns dict mapping signal name -> 40000-pt array on the common grid.
    """
    rec = run_sim(mode=mode, settle_time=SETTLE[mode])
    grid = np.linspace(0, L, L)
    keys = ["PNA", "VNA", "Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay"]
    return {k: resample(rec[k], grid, L) for k in keys}


# ------------------------------------------------------------------ plotting
def overlay(ax, phase_times):
    """Draw the shared stim window plus this panel's phase bands/dividers."""
    ax.axvspan(STIM_START, STIM_END, color='k', alpha=0.05)
    for x in (STIM_START, STIM_END):
        ax.axvline(x, c='k', ls='--', lw=1)
    band_colors = ['r', 'b', 'g']
    for i in range(len(phase_times) - 1):
        ax.axvspan(phase_times[i], phase_times[i + 1], color=band_colors[i % 3], alpha=0.06)
    for x in phase_times:
        ax.axvline(x, c='k', ls='--', lw=1)


def main():
    eup = load_mode('eupneic')
    apn = load_mode('apneustic')

    trace_keys = ["Insp", "RampI", "Exp", "LateExp", "Sw1", "Sw2", "Sensory_relay"]
    trace_lbls = ["I (Early-I)", "Ramp-I", "E (Post-I)", "Late-E",
                  r"$Sw_{1}$", r"$Sw_{2}$", "Sensory Relay"]

    # (group, key, source, mode-for-bands, color, right-label)
    rows = [
        ("B", "PNA", eup, 'eupneic', 'r', "PNA, eupneic"),
        ("B", "VNA", eup, 'eupneic', 'b', "VNA, eupneic"),
        ("C", "PNA", apn, 'apneustic', 'r', "PNA, apneustic"),
        ("C", "VNA", apn, 'apneustic', 'b', "VNA, apneustic"),
    ]
    rows += [("D", k, apn, 'apneustic', 'k', lbl) for k, lbl in zip(trace_keys, trace_lbls)]

    hr = [1.5, 1.5, 1.5, 1.5] + [1.0] * len(trace_keys)
    fig, axs = plt.subplots(len(rows), 1, figsize=(15, 0.55 * sum(hr)),
                            gridspec_kw={'height_ratios': hr})
    plt.subplots_adjust(hspace=0.0)   # seamless stacked panels (continuous stim/phase shading)

    prev = None
    for ax, (grp, key, src, mode, col, lbl) in zip(axs, rows):
        y = src[key]
        ax.plot(y, lw=1.6, c=col)
        overlay(ax, BANDS[mode])
        ax.set_xlim(0, L)
        # leave clear headroom so the peak fills ~2/3 of the (seamless) panel and never
        # crowds the panel above. A near-silent trace (amp < 0.05) -- e.g. post-I during
        # apneusis -- is plotted on its eupneic amplitude so it stays small instead of
        # being inflated to fill the panel.
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if ymax - ymin > 0.05:
            amp = ymax - ymin
            ax.set_ylim(ymin - 0.05 * amp, ymax + 0.45 * amp)
        else:
            ref = max(float(np.nanmax(eup[key])), 0.05) if key in eup else 0.3
            ax.set_ylim(-0.05 * ref, 1.45 * ref)
        ax.text(1.012, 0.5, lbl, transform=ax.transAxes, va='center', ha='left', fontsize=11)
        ax.axis('off')
        if grp != prev:
            ax.text(-0.01, 1.0, grp, transform=ax.transAxes, va='top', ha='right',
                    fontsize=18, fontweight='bold')
            prev = grp

    # stim label above the very top axis; scale bar bottom-right of B
    top = axs[0]
    top.text(0.5 * (STIM_START + STIM_END), 1.15, "10 s SLN stimulation",
             transform=top.get_xaxis_transform(), ha='center', va='bottom', fontsize=12)
    x0 = L - 3000
    top.plot([x0, x0 + 2500], [-0.15, -0.15], transform=top.get_xaxis_transform(),
             c='k', lw=2, clip_on=False)
    top.text(x0 + 1250, -0.30, "2.5 s", transform=top.get_xaxis_transform(),
             ha='center', va='top', fontsize=10)

    out = os.path.join(get_project_root(), "img", "Fig7_composite.pdf")
    fig.savefig(out, bbox_inches='tight', transparent=False)
    plt.close(fig)
    print("wrote", out)


if __name__ == '__main__':
    main()
