import pickle
import os
import numpy as np
from scipy.optimize import fsolve
from tqdm.auto import tqdm
import numdifftools as nd
import warnings
from rCPGswCPG.Network import firing_rate
from rCPGswCPG.Network import construct_model
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
from rCPGswCPG.utils.gen_utils import get_project_root
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
mpl.use('MacOSX')  # on macOS built-in backend
import matplotlib.tri as mtri
# mpl.use('QtAgg')  # if you have PyQt5/PySide6 installed

# find a solution of a system of nonlinear equations
def find_solutions(dim, equations, args, bounds, num_iter):
    sols = []
    for i in range(num_iter):
        init_guess = np.array([bounds[j, 0] + (bounds[j, 1] - bounds[j, 0]) * np.random.rand() for j in range(dim)])
        roots = fsolve(equations, init_guess, args=args)
        if np.allclose(equations(roots, *args), np.zeros(dim)):
            roots = np.round(roots, 4)
            if not np.array_str(roots) in [np.array_str(sol) for sol in sols]:
                sols.append(roots)
    return sols

def rhs(vs, ms, ws, drives, inps):
    # NOTE: this fixed-point computation still hardcodes the OLD parametrization
    # (scale=1/tau_v_old=200, alpha=0.01, beta=0.3). It is internally consistent
    # in the old v~O(40) coordinates, but mismatched with the rescaled model
    # trajectory below; reworking to the new coordinates is a separate task.
    scale = 200
    alpha = 0.01
    bias = -0.2
    fr = firing_rate(vs, 0.3)
    rhs_v = scale * (-alpha * vs - ms + (drives + bias) + ws @ fr + inps)
    return rhs_v

def determine_stability(point, rhs_eq, ms, ws, drives, inps):
    # linearize equations first
    A = (nd.Jacobian(rhs_eq)(point, ms, ws, drives, inps))
    eigenvals = np.round(np.linalg.eig(A)[0], 4)
    label = 'unstable'
    if np.all(np.real(eigenvals) < 0):
        label = 'stable'
    return label, eigenvals

def find_fixed_points(ms, drives, ws, inps, bounds):
    fixed_points = find_solutions(dim=2, equations=rhs, args=(ms, ws, drives, inps), bounds=bounds, num_iter=100)
    data = np.empty((0, 5), dtype=object)
    for fp in fixed_points:
        label, eigenvals = determine_stability(fp, rhs, ms, ws, drives, inps)
        data = np.append(data, np.array([[*ms, *fp, label]]), axis = 0)
    return data

def calculate_equilibrium_surface(drives, ws, inps, bounds):
    m1s = np.linspace(0.0, 0.04, 40)
    m2s = np.linspace(0.17, 0.23, 40)

    # for every point find solutions of the fast subsystem:
    data_surf = np.empty((0, 5), dtype=object)
    for i in tqdm(range(len(m1s))):
        for j in range(len(m2s)):
            ms = np.array([m1s[i], m2s[j]])
            # calculate fixed points of the fast subsystem:
            data = find_fixed_points(ms, drives=drives, ws=ws, inps=inps, bounds=bounds)
            data_surf = np.append(data_surf, data, axis = 0)
    return data_surf

def plot_sheet_trisurf(x, y, z, ax=None, color=None, alpha=0.12, len_pct=95, dz_pct=95, lw=0.2):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7, 5))
    tri = mtri.Triangulation(x, y)
    T = tri.triangles
    P3 = np.c_[x, y, z]
    e0 = np.linalg.norm(P3[T[:, 0]] - P3[T[:, 1]], axis=1)
    e1 = np.linalg.norm(P3[T[:, 1]] - P3[T[:, 2]], axis=1)
    e2 = np.linalg.norm(P3[T[:, 2]] - P3[T[:, 0]], axis=1)
    Lmax = np.maximum(e0, np.maximum(e1, e2))
    dz0 = np.abs(z[T[:, 0]] - z[T[:, 1]])
    dz1 = np.abs(z[T[:, 1]] - z[T[:, 2]])
    dz2 = np.abs(z[T[:, 2]] - z[T[:, 0]])
    DZmax = np.maximum(dz0, np.maximum(dz1, dz2))
    tri.set_mask((Lmax > np.percentile(Lmax, len_pct)) | (DZmax > np.percentile(DZmax, dz_pct)))
    ax.plot_trisurf(tri, z, linewidth=lw, antialiased=True, alpha=alpha, color=color)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    ax.set(xlabel=r'$m_1$', ylabel=r'$m_2$', zlabel=r'$v_1$')
    return ax

if __name__ == '__main__':
    #plotting PPA
    recalculate = False
    img_folder = os.path.join(get_project_root(), "img")
    data_folder = os.path.join(get_project_root(), "data")

    model_name = "swHCO"
    model_params = model_params_from_cfg(load_model_cfg_file(model_name.replace("model_", "")))
    hco = construct_model(model_params)
    external_inputs = np.zeros(model_params["N"])
    pnames = model_params["pnames"]
    weights = np.array([model_params["W"][0, 1], model_params["W"][1, 0]])
    drives = np.array([model_params["drives_misc"][0, 0], model_params["drives_misc"][0, 1]])
    inputs = np.array([0.12, 0.07])
    # inputs = np.array([0.19, 0.07])
    dt = 0.01
    hco.dt = dt
    T_before_stim = 0.5
    T_stim = 0.15
    T_after_stim = 2

    hco.run(T_before_stim, np.zeros(2))
    hco.run(T_stim, inputs)
    hco.run(T_after_stim, np.zeros(2))
    recordings = hco.get_recordings()

    v1_traj = recordings["v_history"][:, 0]
    v2_traj = recordings["v_history"][:, 1]
    m1_traj = recordings["m_history"][:, 0]
    m2_traj = recordings["m_history"][:, 1]
    print(np.min(v1_traj), np.max(v1_traj))
    print(np.min(v2_traj), np.max(v2_traj))
    print(np.min(m1_traj), np.max(m1_traj))
    print(np.min(m2_traj), np.max(m2_traj))

    lim = np.array([-40, 20])
    inps = np.zeros(2)
    bounds = np.array([[-40, 20], [-40, 20]])
    tag = (np.abs(weights[0]), np.abs(weights[1]), drives[0], drives[1])
    file_name = os.path.join(data_folder, f"eq_surface_simplified_HCO_{tag}.pkl")
    if not os.path.exists(file_name) or recalculate:
        data_surf = calculate_equilibrium_surface(drives, hco.W, inps, bounds)
        pickle.dump(data_surf, open(file_name, 'wb+'))
    data_surf = pickle.load(open(file_name, 'rb+'))

    m1s = data_surf[:, 0].astype(float)
    m2s = data_surf[:, 1].astype(float)
    v1s = data_surf[:, 2].astype(float)
    v2s = data_surf[:, 3].astype(float)
    labels = data_surf[:, 4].astype(str)

    m1s_stable = m1s[labels == 'stable']
    m2s_stable = m2s[labels == 'stable']
    v1s_stable = v1s[labels == 'stable']
    v2s_stable = v2s[labels == 'stable']
    m1s_unstable = m1s[labels == 'unstable']
    m2s_unstable = m2s[labels == 'unstable']
    v1s_unstable = v1s[labels == 'unstable']
    v2s_unstable = v2s[labels == 'unstable']


    def split_upper_lower(z_stable, z_unstable):
        thr = float(np.mean(z_unstable))
        upper = z_stable >= thr
        lower = ~upper
        return upper, lower, thr

    upper, lower, thr = split_upper_lower(v1s_stable, v1s_unstable)
    m1_up, m2_up, v1_up = m1s_stable[upper], m2s_stable[upper], v1s_stable[upper]
    m1_lo, m2_lo, v1_lo = m1s_stable[lower], m2s_stable[lower], v1s_stable[lower]

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(7, 5))
    ax.scatter(m1s_stable, m2s_stable, v1s_stable, c='blue', marker='o', alpha = 0.1)
    ax.scatter(m1s_unstable, m2s_unstable, v1s_unstable, c='deepskyblue', marker='o', alpha = 0.1)
    # ax = plot_sheet_trisurf(m1_up, m2_up, v1_up, alpha=0.20, color='blue', ax=ax)
    # ax = plot_sheet_trisurf(m1s_unstable, m2s_unstable, v1s_unstable, alpha=0.30, color='deepskyblue', ax=ax)
    # ax = plot_sheet_trisurf(m1_lo, m2_lo, v1_lo, alpha=0.20, color='blue', ax=ax)

    start_stim = int(T_before_stim * 1000 / dt)
    stop_stim = int((T_stim) * 1000 / dt) + start_stim
    ax.scatter(m1_traj[start_stim+1], m2_traj[start_stim+1], v1_traj[start_stim+1], color='r', s=20)
    ax.scatter(m1_traj[stop_stim+1], m2_traj[stop_stim+1], v1_traj[stop_stim+1], color='g', s=20)
    ax.scatter(m1_traj[-1], m2_traj[-1], v1_traj[-1], color='b', s=20)

    ax.plot3D(m1_traj[start_stim:stop_stim], m2_traj[start_stim:stop_stim], v1_traj[start_stim:stop_stim], color='red', alpha=0.5)
    ax.plot3D(m1_traj[stop_stim:], m2_traj[stop_stim:], v1_traj[stop_stim:], color='k', alpha=0.5)

    ax.set_zlim(-40, 15)

    # three ticks per axis
    xt = np.linspace(*ax.get_xbound(), 3)
    yt = np.linspace(*ax.get_ybound(), 3)
    zt = np.linspace(*ax.get_zbound(), 3)

    ax.set_xticks(xt);
    ax.set_yticks(yt);
    ax.set_zticks(zt)

    # formatters: x/y with two decimals, z as integers
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.2f}"))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(round(v))}"))

    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    ax.set(xlabel=r'$m_1$', ylabel=r'$m_2$', zlabel=r'$v_1$')
    ax.view_init(elev=20, azim=-127)
    tag = (np.abs(weights[0]), np.abs(weights[1]), drives[0], drives[1], inputs[0], inputs[1])
    plt.savefig(os.path.join(img_folder, f'3Dppa_{tag}.pdf'), bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(img_folder, f'3Dppa_{tag}.png'), bbox_inches='tight', transparent=True, dpi=500)
    plt.show()