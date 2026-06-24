"""Assemble Fig. 6 composite: solitary-swallow PIR mechanism + cusp phase space.

Layout (2 rows x 2 cols), one row per stimulus scenario:
  A  Sw1/Sw2 membrane-potential traces      B  cusp equilibrium surface + trajectory   (scenario 1)
  C  Sw1/Sw2 membrane-potential traces      D  cusp equilibrium surface + trajectory   (scenario 2)

The equilibrium surface (scenario-independent) is loaded from the cache produced by
PIR_solitary_swallow_phase_space_analysis.py; trajectories are re-simulated here. All
quantities are in the rescaled (new) coordinates, so the trajectory lies on the surface.
"""
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from rCPGswCPG.Network import construct_model
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
from rCPGswCPG.utils.gen_utils import get_project_root

DT = 0.01                     # ms
T_BEFORE, T_STIM, T_AFTER = 0.5, 0.15, 2.0    # s
SCENARIOS = [(0.12, 0.07), (0.19, 0.07)]      # (Sw1, Sw2) stim -> rows (A,B) and (C,D)


def run_trajectory(stim):
    """Simulate the swHCO with a brief (Sw1,Sw2) stimulus.

    Returns (t, v, m, i0, i1): time (s), v/m histories (Nx2), stim start/end indices.
    """
    mp = model_params_from_cfg(load_model_cfg_file('swHCO'))
    hco = construct_model(mp)
    hco.dt = DT
    hco.run(T_BEFORE, np.zeros(2))
    hco.run(T_STIM, np.array(stim))
    hco.run(T_AFTER, np.zeros(2))
    r = hco.get_recordings()
    i0 = int(T_BEFORE * 1000 / DT)
    i1 = int((T_BEFORE + T_STIM) * 1000 / DT)
    return r['t'], r['v_history'], r['m_history'], i0, i1


def load_surface():
    """Load the cached equilibrium surface; return stable/unstable (m1,m2,v1) arrays."""
    tag = "(np.float64(0.65), np.float64(0.9), np.float64(0.175), np.float64(0.35))"
    f = os.path.join(get_project_root(), "data", f"eq_surface_simplified_HCO_{tag}.pkl")
    d = pickle.load(open(f, 'rb'))
    m1, m2, v1, lab = d[:, 0].astype(float), d[:, 1].astype(float), d[:, 2].astype(float), d[:, 4].astype(str)
    st = lab == 'stable'
    return (m1[st], m2[st], v1[st]), (m1[~st], m2[~st], v1[~st])


def get_points(v, i0, i1, n):
    """Indices of the n labeled trajectory stages of v1 (text 'points 1-2-3-4' / '1-2-3').

    Scenario 1 (n=4): 1 = rest, 2 = top of the upper sheet (post-inhibitory rebound
    peak), 3 = cusp / abrupt fall, 4 = return to rest.
    Scenario 2 (n=3): 1 = rest, 2 = driven peak during the stimulus, 3 = return to rest.
    """
    if n == 4:
        post = v[i1:, 0]
        p2 = i1 + int(np.argmax(post))                  # rebound onto the upper sheet
        p3 = p2 + int(np.argmin(np.diff(v[p2:, 0])))    # steepest drop = cusp/termination
        # point 4 = partway along the lower-sheet return (~1.5 s), not all the way to rest
        p4 = min(int(1.5 * 1000 / DT), len(v) - 1)
        return [i0, p2, p3, p4]
    # scenario 2: point 2 in the middle of the stimulation
    p2 = (i0 + i1) // 2
    return [i0, p2, len(v) - 1]


def plot_traces(ax, t, v, i0, i1, letter, points, show_x=True):
    """Sw1/Sw2 membrane-potential traces around the stimulus, with numbered stages.

    show_x=False suppresses the (shared) time axis on the upper panel.
    """
    a = i0 - int(0.3 * 1000 / DT)
    ax.plot(t[a:], v[a:, 0], 'r', lw=1.6, label=r'$\text{Sw}_1$ ($v_1$)')
    ax.plot(t[a:], v[a:, 1], 'b', lw=1.6, label=r'$\text{Sw}_2$ ($v_2$)')
    ax.axvspan(t[i0], t[i1], color=(1.0, 0.90, 0.90), lw=0, zorder=0)   # opaque (vector)
    for n, pi in enumerate(points):                     # numbered trajectory stages on Sw1
        ax.scatter(t[pi], v[pi, 0], color='k', s=18, zorder=5)
        ax.annotate(str(n + 1), (t[pi], v[pi, 0]), textcoords='offset points',
                    xytext=(2, 6), fontsize=10, fontweight='bold')
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.set_ylabel(r'$v_1$ (Sw$_1$),  $v_2$ (Sw$_2$)')
    ax.yaxis.set_major_locator(MaxNLocator(3))          # 2-3 y ticks only
    if show_x:
        ax.set_xlabel('t (s)')
    else:
        ax.tick_params(labelbottom=False)               # time ticks live on the bottom panel
    ax.legend(fontsize=9, loc='lower right', frameon=False)
    ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=18, fontweight='bold')


def plot_phase(ax, stable, unstable, v, m, i0, i1, letter, points):
    """Cusp equilibrium surface (stable navy / unstable sky) with the trajectory on it."""
    ax.computed_zorder = False          # respect manual zorder so the trajectory stays in front
    # opaque light-colored surface points (no alpha) so the figure stays vector
    ax.scatter(*stable, c=[(0.62, 0.66, 0.95)], marker='o', s=4, edgecolors='none', zorder=0)
    ax.scatter(*unstable, c=[(0.62, 0.83, 0.96)], marker='o', s=4, edgecolors='none', zorder=0)
    ax.plot3D(m[i0:i1, 0], m[i0:i1, 1], v[i0:i1, 0], color='red', lw=2, zorder=10)   # stim
    ax.plot3D(m[i1:, 0], m[i1:, 1], v[i1:, 0], color='k', lw=2, zorder=10)           # post-stim
    for n, pi in enumerate(points):                     # numbered trajectory stages, in front
        ax.scatter(m[pi, 0], m[pi, 1], v[pi, 0], color='k', s=24, zorder=15)
        ax.text(m[pi, 0], m[pi, 1], v[pi, 0] + 0.05, str(n + 1), fontsize=11, fontweight='bold', zorder=20)
    ax.set_zlim(-0.55, 0.25)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    ax.set(xlabel=r'$m_1$', ylabel=r'$m_2$', zlabel=r'$v_1$')
    ax.set_xticks(np.linspace(*ax.get_xbound(), 3)); ax.set_yticks(np.linspace(*ax.get_ybound(), 3))
    ax.set_zticks(np.linspace(*ax.get_zbound(), 3))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.view_init(elev=20, azim=-127)
    ax.set_box_aspect(None, zoom=1.45)                  # enlarge the 3D content in the panel
    ax.text2D(0.0, 0.92, letter, transform=ax.transAxes, fontsize=18, fontweight='bold')


def main():
    stable, unstable = load_surface()
    fig = plt.figure(figsize=(15, 8))
    # give the 3D phase-space column more width; traces stay compact on the left
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.7], wspace=0.02, hspace=0.3)
    letters = [('A', 'B'), ('C', 'D')]
    for row, stim in enumerate(SCENARIOS):
        t, v, m, i0, i1 = run_trajectory(stim)
        pts = get_points(v, i0, i1, 4 if row == 0 else 3)   # scenario 1: 4 stages, scenario 2: 3
        ax_tr = fig.add_subplot(gs[row, 0])
        plot_traces(ax_tr, t, v, i0, i1, letters[row][0], pts, show_x=(row == len(SCENARIOS) - 1))
        ax_ph = fig.add_subplot(gs[row, 1], projection='3d')
        plot_phase(ax_ph, stable, unstable, v, m, i0, i1, letters[row][1], pts)
    out = os.path.join(get_project_root(), "img", "Fig6_composite.pdf")
    fig.savefig(out, bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == '__main__':
    main()
