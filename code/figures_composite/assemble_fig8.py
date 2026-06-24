"""Assemble Fig. 8 composite: pathological breathing-swallowing interactions.

Three conditions, each PNA/VNA under a 10 s SLN stimulation, sharing one x-axis:
  A  healthy swallowing response (intact network)
  B  inspiratory breakthroughs  (weakened sensory inhibition of I/Ramp-I -> -0.22)
  C  weak/delayed swallow        (weakened sensory drive to Sw1 -> 0.05)
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from rCPGswCPG.Network import firing_rate, construct_model
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
from rCPGswCPG.utils.gen_utils import get_project_root, put

T_PRE, T_STIM, T_POST, AMP = 7.5, 10.0, 7.5, 0.4
VNA_COEFF = {"KF_phasic": 0.75, "Sw1": 0.6, "RampI": 0.9}


def run_condition(mod):
    """Run the complex model under condition `mod` and return (PNA, VNA, stim_idx_start/end).

    mod in {'healthy','insp_breakthroughs','weak_swallow'}.
    """
    mp = model_params_from_cfg(load_model_cfg_file('complex'))
    model = construct_model(mp)
    pnames = mp['pnames']
    if mod == 'insp_breakthroughs':
        model.W[pnames.index("Insp"), pnames.index("Sensory_relay")] = -0.22
        model.W[pnames.index("RampI"), pnames.index("Sensory_relay")] = -0.22
    elif mod == 'weak_swallow':
        model.W[pnames.index("Sw1"), pnames.index("Sensory_relay")] = 0.05
    model.sync_params()

    ext = np.zeros(len(pnames))
    model.run(T_PRE, put(ext, pnames.index("Sensory_relay"), 0))
    model.run(T_STIM, put(ext, pnames.index("Sensory_relay"), AMP))
    model.run(T_POST, put(ext, pnames.index("Sensory_relay"), 0))

    v, _ = model.get_raw_history()
    fr = firing_rate(v, model.beta)
    PNA = fr[:, pnames.index("RampI")]
    VNA = sum(c * fr[:, pnames.index(p)] for p, c in VNA_COEFF.items())
    i0 = int(T_PRE * 1000 / mp['dt'])
    i1 = int((T_PRE + T_STIM) * 1000 / mp['dt'])
    return PNA, VNA, i0, i1


def main():
    conds = [('A', 'healthy', 'healthy'),
             ('B', 'insp_breakthroughs', 'insp. breakthroughs'),
             ('C', 'weak_swallow', 'weak swallow')]
    rows = []
    for letter, mod, title in conds:
        PNA, VNA, i0, i1 = run_condition(mod)
        rows.append((letter, 'PNA', PNA, 'r', f"PNA, {title}", i0, i1))
        rows.append((letter, 'VNA', VNA, 'b', f"VNA, {title}", i0, i1))

    fig, axs = plt.subplots(len(rows), 1, figsize=(13, 0.9 * len(rows)),
                            gridspec_kw={'height_ratios': [1, 2] * len(conds)})
    plt.subplots_adjust(hspace=0.15)

    prev = None
    for ax, (letter, key, y, col, lbl, i0, i1) in zip(axs, rows):
        ax.plot(y, lw=1.5, c=col, zorder=3)
        ax.axvspan(i0, i1, color=(0.93, 0.93, 0.93), lw=0, zorder=0)   # opaque (vector, no alpha)
        for x in (i0, i1):
            ax.axvline(x, c='k', ls='--', lw=1, zorder=1)
        ax.set_xlim(0, len(y))
        ymax = 1.05 * float(np.nanmax(y))
        ax.set_ylim(-0.02 * ymax, ymax if ymax > 0 else 1.0)
        ax.text(1.012, 0.5, lbl, transform=ax.transAxes, va='center', ha='left', fontsize=11)
        ax.axis('off')
        if letter != prev:
            ax.text(-0.01, 1.0, letter, transform=ax.transAxes, va='top', ha='right',
                    fontsize=18, fontweight='bold')
            prev = letter

    axs[0].text(0.5 * (rows[0][5] + rows[0][6]), 1.2, "10 s SLN stimulation",
                transform=axs[0].get_xaxis_transform(), ha='center', va='bottom', fontsize=12)
    out = os.path.join(get_project_root(), "img", "Fig8_composite.pdf")
    fig.savefig(out, bbox_inches='tight', transparent=False)
    plt.close(fig)
    print("wrote", out)


if __name__ == '__main__':
    main()
