from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from scipy.signal import savgol_filter as sg
from rCPGswCPG.utils.gen_utils import get_project_root
from rCPGswCPG.utils.utils import get_val_of_param
from rCPGswCPG.model_params.config_loader import load_model_cfg_file, model_params_from_cfg
import re

def plot_analytics(base_folder, xlabel, title, x_vline, PIR_bnds, sp_sw_bnds, outfile, show=False):
    data_file = os.path.join(base_folder, "data_table.pkl")
    img_folder = os.path.join(base_folder, "imgs", "analysis")
    os.makedirs(img_folder, exist_ok=True)
    data_table = pickle.load(open(data_file, "rb+"))

    columns = data_table["columns"]
    vals = data_table["vals"]
    x = vals[:, -1]

    # plotting 'spont_swallows', 'N_sw', 'N_br', N_sw_shortSI
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    PIR_start, PIR_end = PIR_bnds
    if PIR_start is not None:
        ax.axvline(PIR_start, color ='r', linestyle='--', alpha=0.5)
    if PIR_end is not None:
        ax.axvline(PIR_end, color ='r', linestyle='--', alpha=0.5)
    if PIR_start is None or PIR_end is None:
        pass
    else:
        ax.axvspan(PIR_start, PIR_end, color=None, facecolor='r', alpha=0.05)

    # analogously for spontaneous swallows
    sp_sw_start, sp_sw_end = sp_sw_bnds
    if sp_sw_start is not None:
        ax.axvline(sp_sw_start, color ='b', linestyle='--', alpha=0.5)
    if sp_sw_end is not None:
        ax.axvline(sp_sw_end, color ='b', linestyle='--', alpha=0.5)
    if sp_sw_start is None or sp_sw_end is None:
        pass
    else:
        ax.axvspan(sp_sw_start, sp_sw_end, color=None, facecolor='b', alpha=0.05)

    ax.axvline(x_vline, linestyle='-', color='k', alpha = 0.5, linewidth = 3)
    ax.scatter(x, vals[:, columns.index("N_sw")], s=15, color='g', label="N swallows")
    # ax.scatter(x, vals[:, columns.index("N_br")], s=15, color='r', label="N Insp. breakthroughs", marker='x')
    # ax.set_ylabel("Number of swallows")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.3)
    print(title)
    # if "Varying" in title and r"$\tau_m$" in title and r"$\text{Sw}_{1}$" in title:
    #     plt.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    if "arousal" in title:
        ax.set_xticklabels([])
    else:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)

        print("  Legend added.".capitalize())
    if show:
        plt.show()
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0.0, transparent=True)
    # fig.savefig(outfile.split(".pdf")[0] + ".png", bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.close()
    return None

def plot_responses(VNA, PNA, long_stim_start_ind, long_stim_stop_ind,
                   short_stim_start_ind, short_stim_stop_ind, outpath=None, show=False):
    fig, axes = plt.subplots(2, 1, figsize=(6, 2), sharex=True, gridspec_kw={'hspace': 0})
    axes[0].plot(PNA,  c='r', label='PNA')
    axes[1].plot(VNA,  c='b', label='VNA')

    for ax in axes:
        ax.axvline(long_stim_start_ind,  c='r'); ax.axvline(long_stim_stop_ind,  c='r')
        ax.axvspan(long_stim_start_ind,  long_stim_stop_ind,  color='r', alpha=0.1)
        ax.axvline(short_stim_start_ind, c='r'); ax.axvline(short_stim_stop_ind, c='r')
        ax.axvspan(short_stim_start_ind, short_stim_stop_ind, color='r', alpha=0.1)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(x=0, y=0)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    if outpath:
        fig.savefig(outpath, bbox_inches='tight', pad_inches=0, transparent=True)
    if show:
        plt.show()

def find_bounds(vals, param_vals):
    if np.any(vals == 1):  
        start = param_vals[np.where(vals == 1)[0][0]]
        end = param_vals[np.where(vals == 1)[0][-1]]
    else:
        start = None
        end = None
    return [start, end]


def _fmt_tok(t):
    m = re.fullmatch(r'([A-Za-z]+)(\d+)', t)
    if m: return fr"$\text{{{m.group(1)}}}_{{{m.group(2)}}}$"
    if t.isdigit(): return fr"$_{{{t}}}$"
    return t

def param_mapping(s):
    if s.startswith("W_") and "_to_" in s:
        pre, post = s.split("W_")[1].split("_to_")
        return f"W {_fmt_tok(pre)} → {_fmt_tok(post)}"
    if s.endswith("_drive"):
        return f"drive to {_fmt_tok(s[:-6])}"
    if s.endswith("_tau"):
        return fr"$\tau_m$ of {_fmt_tok(s[:-4])}"
    return _fmt_tok(s)

def get_index(T, dt):
    return int(T*1000 / dt)

if __name__ == '__main__':
    VNA_coeffs = {"KF_phasic" : 0.75, "Sw1" : 0.6, "RampI" : 0.9}
    img_folder = os.path.join(get_project_root(), "img")
    exp_folder = os.path.join(get_project_root(), "data", "experiments")
    model_name = "model_complex"
    model_params = model_params_from_cfg(load_model_cfg_file(model_name.replace("model_", "")))
    
    for exp_name in os.listdir(exp_folder):
        if not exp_name.startswith("Experiment_") and not exp_name.startswith("Varying_"):
            continue
        prefix = exp_name.split("_")[0]
        param_name = "_".join(exp_name.split("_")[1:])
        param_str = param_mapping(param_name)
        xlabel = param_str
        title = rf"Varying {param_str}"
        
        data_table_file = os.path.join(get_project_root(), "data", "experiments", exp_name, "data_table.pkl")
        with open(data_table_file, "rb") as f:
            data_dict = pickle.load(f)

        normal_param_val = get_val_of_param(data_dict["columns"][-1], model_params)
        x_vline = normal_param_val

        param_vals = data_dict["vals"][:, data_dict["columns"].index(data_dict["columns"][-1])]
        
        PIR_vals = data_dict["vals"][:, data_dict["columns"].index("PIR")]
        sp_sw_vals = data_dict["vals"][:, data_dict["columns"].index("spont_swallows")]

        x_vline = normal_param_val
        PIR_bnds = find_bounds(PIR_vals, param_vals)
        sp_sw_bnds = find_bounds(sp_sw_vals, param_vals)

        outfile = os.path.join(img_folder, f"{exp_name}.pdf")
        base_folder = os.path.join(get_project_root(), "data", "experiments", exp_name)
        plot_analytics(base_folder, xlabel, title, x_vline, PIR_bnds, sp_sw_bnds, outfile)

        # plot VNA recordings:
        data_folder = os.path.join(f"data", "experiments", f"{prefix}_{param_name}", "runs", "recordings")
        files = os.listdir(data_folder)
        files_filtered = []

        vals2plot = [param_vals[np.argmin(np.abs(param_vals - normal_param_val))] * 0.6,
                     param_vals[np.argmin(np.abs(param_vals - normal_param_val))] * 1.3]
        vals = np.array([float(files.split("_")[2].split(".pkl")[0]) for files in files])
        for v2p in vals2plot:
            files_filtered.append(files[np.argmin(np.abs(vals - v2p))])

        conf_file_path = os.path.join("data", "experiments", f"{prefix}_{param_name}", "config_file.pkl")
        config = pickle.load(open(conf_file_path, 'rb'))
        dt = config["model_params"]["dt"]
        noSI_T = config["protocol_dict"]["Protocol_LongShortSI"]["noSI_T"]
        longSI_T = config["protocol_dict"]["Protocol_LongShortSI"]["longSI_T"]
        interim_T = config["protocol_dict"]["Protocol_LongShortSI"]["interim_T"]
        stim_duration = config["protocol_dict"]["Protocol_LongShortSI"]["stim_duration"]

        t_start_ind = get_index(3 * noSI_T / 4, dt)
        t_stop_ind = -get_index(3 * noSI_T / 4, dt)
        long_stim_start_ind = get_index(noSI_T, dt) - t_start_ind
        long_stim_stop_ind = get_index(noSI_T + longSI_T, dt) - t_start_ind
        short_stim_start_ind = get_index(noSI_T + longSI_T + interim_T, dt) - t_start_ind
        short_stim_stop_ind = get_index(noSI_T + longSI_T + interim_T + stim_duration, dt) - t_start_ind
        for file in files_filtered:
            data = pickle.load(open(os.path.join(data_folder, file), "rb"))['protocol_runs']['Protocol_LongShortSI']
            pnames = data['population_names']
            VNA = np.sum(np.stack([VNA_coeffs[pname] * data["fr_history"][:, pnames.index(pname)] for pname in VNA_coeffs.keys()], axis = 1), axis=1)
            VNA = VNA[t_start_ind:t_stop_ind]
            PNA = data["fr_history"][:, pnames.index("RampI")][t_start_ind:t_stop_ind]  
            v = file.split('_')[2].split('.pkl')[0]
            if float(v) > 10:
                v = int(float(v))
            else:
                v = round(float(v), 2)
            outpath = os.path.join(img_folder, f"responses_{prefix}_{param_name}_{v}.pdf")
            plot_responses(VNA, PNA, long_stim_start_ind, long_stim_stop_ind, short_stim_start_ind, short_stim_stop_ind, outpath=outpath)