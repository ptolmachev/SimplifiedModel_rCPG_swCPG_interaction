"""Assemble Fig. 9: effect of varying swallowing-neuron parameters.

Four parameter rows (A-D); each row = an analytics panel (number of swallows vs the
parameter, with the PIR / spontaneous-swallow regions and the intact value marked)
plus two example PNA/VNA recordings (a low and a high parameter value, marked 1 / 2
on the analytics panel), each spanning a 10 s + 100 ms SLN stimulation protocol.

Note on labels: get_short_name (and hence the experiment folder names) was written for
the old W[from,to] convention; under the current W[to,from] convention the two cross-
connection sweeps are labeled with their CORRECT direction here (folder
Varying_W_Sw1_to_Sw2 actually varies Sw2->Sw1, and vice-versa). The computation is
unaffected -- only the displayed label is corrected.
"""
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
from rCPGswCPG.utils.utils import get_val_of_param
from rCPGswCPG.utils.gen_utils import get_project_root

VNA_COEFF = {"KF_phasic": 0.75, "Sw1": 0.6, "RampI": 0.9}

# (folder, row letter, corrected x-label, is_arousal)
ROWS = [
    ("Varying_Sw1_tau_m",     "A", r"$\tau_m$ of $Sw_1$ (ms)", False),
    ("Experiment_arousal",    "B", "arousal drive, %",         True),
    ("Varying_W_Sw1_to_Sw2",  "C", r"$W\ Sw_2\!\to\!Sw_1$",    False),  # folder name is reversed
    ("Varying_W_Sw2_to_Sw1",  "D", r"$W\ Sw_1\!\to\!Sw_2$",    False),  # folder name is reversed
]


def gidx(T, dt):
    return int(T * 1000 / dt)


def find_bounds(vals, pv):
    """Return [first, last] param value where vals==1 (the active region), else [None,None]."""
    if np.any(vals == 1):
        w = np.where(vals == 1)[0]
        return pv[w[0]], pv[w[-1]]
    return None, None


def load_row(folder, is_arousal, mp):
    """Load analytics + the two example recordings for one parameter row."""
    base = os.path.join(get_project_root(), "data", "experiments", folder)
    dt = pickle.load(open(os.path.join(base, "config_file.pkl"), "rb"))["model_params"]["dt"]
    d = pickle.load(open(os.path.join(base, "data_table.pkl"), "rb"))
    cols, vals = d["columns"], d["vals"]
    pv = vals[:, -1].astype(float)            # last column = (first) swept parameter
    Nsw = vals[:, cols.index("N_sw")].astype(float)
    pir = find_bounds(vals[:, cols.index("PIR")].astype(float), pv)
    spw = find_bounds(vals[:, cols.index("spont_swallows")].astype(float), pv)
    normal = get_val_of_param(cols[-1], mp)

    if is_arousal:
        x = 100.0 * pv / normal               # express as % of intact drive
        xnorm = 100.0
        targets = [pv[len(pv) // 4], pv[3 * len(pv) // 4]]
    else:
        x = pv
        xnorm = normal
        targets = [normal * 0.6, normal * 1.3]

    # pick the two example recordings closest to the targets
    rec_dir = os.path.join(base, "runs", "recordings")
    files = os.listdir(rec_dir)
    fvals = np.array([float(f.split("_")[2].split(".pkl")[0]) for f in files])
    examples = []
    for t in targets:
        f = files[int(np.argmin(np.abs(fvals - t)))]
        pr = pickle.load(open(os.path.join(rec_dir, f), "rb"))["protocol_runs"]["Protocol_LongShortSI"]
        pn = pr["population_names"]
        a, b = gidx(7.5, dt), -gidx(7.5, dt)
        PNA = pr["fr_history"][:, pn.index("RampI")][a:b]
        VNA = sum(c * pr["fr_history"][:, pn.index(p)] for p, c in VNA_COEFF.items())[a:b]
        stim = (gidx(10, dt) - a, gidx(20, dt) - a, gidx(30, dt) - a, gidx(30.1, dt) - a)
        examples.append((PNA, VNA, stim, float(fvals[int(np.argmin(np.abs(fvals - t)))])))
    return dict(x=x, xnorm=xnorm, Nsw=Nsw, pir=pir, spw=spw, examples=examples,
                xtargets=targets, is_arousal=is_arousal, x_of=lambda p: 100.0 * p / normal if is_arousal else p)


def main():
    mp = model_params_from_cfg(load_model_cfg_file("complex"))
    data = [(L, lt, xl, load_row(L, ar, mp)) for (L, lt, xl, ar) in ROWS]

    fig = plt.figure(figsize=(13, 11))
    outer = gridspec.GridSpec(4, 1, hspace=0.45)
    for r, (folder, letter, xlabel, R) in enumerate(data):
        # row = analytics (col 0) + example1 (col 1) + example2 (col 2); examples have PNA/VNA sub-rows
        gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[r],
                                              width_ratios=[1.1, 2, 2], hspace=0.0, wspace=0.15)
        # --- analytics ---
        axa = fig.add_subplot(gs[:, 0])
        # opaque pale fills / muted guide lines (no alpha -> vector, no transparency)
        for (lo, hi), fill, line in [(R["pir"], (1.0, 0.90, 0.90), (0.85, 0.45, 0.45)),
                                     (R["spw"], (0.90, 0.90, 1.0), (0.45, 0.45, 0.85))]:
            if lo is not None:
                a, b = R["x_of"](lo), R["x_of"](hi)
                axa.axvspan(min(a, b), max(a, b), color=fill, lw=0, zorder=0)
                axa.axvline(a, color=line, ls="--", zorder=1); axa.axvline(b, color=line, ls="--", zorder=1)
        axa.axvline(R["xnorm"], color=(0.55, 0.55, 0.55), lw=3, zorder=1)
        axa.scatter(R["x"], R["Nsw"], s=14, color="g", zorder=3)
        # arrows point at the exact data points used for example recordings 1 and 2
        for k, ex in enumerate(R["examples"]):
            ev = ex[3]                              # actual param value of this example
            j = int(np.argmin(np.abs(R["x"] - R["x_of"](ev))))
            xa, ya = R["x"][j], R["Nsw"][j]
            axa.annotate(str(k + 1), xy=(xa, ya), xytext=(xa, ya + 0.18 * (R["Nsw"].max() + 1)),
                         ha="center", fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.2))
        axa.set_xlabel(xlabel, fontsize=11); axa.set_ylabel("N swallows", fontsize=10)
        axa.spines["top"].set_visible(False); axa.spines["right"].set_visible(False)
        axa.margins(y=0.18)
        axa.text(-0.32, 1.0, letter, transform=axa.transAxes, fontsize=18, fontweight="bold", va="top")

        # --- two example recordings ---
        for k, (PNA, VNA, stim, _) in enumerate(R["examples"]):
            for sub, (sig, col) in enumerate([(PNA, "r"), (VNA, "b")]):
                ax = fig.add_subplot(gs[sub, 1 + k])
                ls, le, ss, se = stim
                ax.axvspan(ls, le, color=(1.0, 0.90, 0.90), lw=0, zorder=0)   # long stim (opaque)
                ax.axvspan(ss, se, color=(1.0, 0.78, 0.78), lw=0, zorder=0)   # short stim (opaque)
                ax.plot(sig, c=col, lw=1.0, zorder=3)
                ax.set_xlim(0, len(sig)); ax.margins(x=0)
                ax.axis("off")
                if sub == 0:
                    ax.set_title(f"{'10 s' if False else ''}", fontsize=9)
                    ax.text(0.0, 1.05, f"{k+1}", transform=ax.transAxes, fontsize=12, fontweight="bold")
        # column headers on the top row only
        if r == 0:
            R["examples"]  # headers via figure text below

    fig.text(0.45, 0.905, "10 s + 100 ms SLN stimulation", ha="center", fontsize=12)
    out = os.path.join(get_project_root(), "img", "Fig9_composite.pdf")
    fig.savefig(out, bbox_inches="tight", transparent=False, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
