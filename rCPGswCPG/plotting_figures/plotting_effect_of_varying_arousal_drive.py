from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from scipy.signal import savgol_filter as sg
from rCPGswCPG.utils.gen_utils import get_project_root, create_dir_if_not_exist


def plot_analytics(base_folder, xlabel, title, x_vline, PIR_bnds, save_to):
    data_file = os.path.join(base_folder, "data_table.pkl")
    img_folder = os.path.join(base_folder, "imgs", "analysis")
    create_dir_if_not_exist(img_folder)
    data_table = pickle.load(open(data_file, "rb+"))

    columns = data_table["columns"]
    vals = data_table["vals"]
    x = np.arange(vals.shape[0])*200/50
    print(columns)
    # plotting 'spont_swallows', 'N_sw', 'N_br', N_sw_shortSI
    fig = plt.figure(figsize=(8, 8))

    spont_sw = vals[:, columns.index("spont_swallows")]
    # plt.scatter(x, spont_sw)
    tmp = np.diff(spont_sw)
    if np.any(spont_sw == 1):
        starts = x[np.where(tmp == 1)[0] + 1]
        ends = x[np.where(tmp == -1)[0] - 1]
        spont_sw_start = x[0] if len(starts) == 0 else starts[0]
        spont_sw_end = x[-1] if len(ends) == 0 else ends[-1]
        if spont_sw_start != x[0]:
            plt.axvline(spont_sw_start, color ='b', linestyle='--')
        if spont_sw_end != x[-1]:
            plt.axvline(spont_sw_end, color ='b', linestyle='--')
        plt.axvspan(spont_sw_start, spont_sw_end, color=None, facecolor='b', alpha=0.05)
    

    PIR_start, PIR_end = PIR_bnds
    if PIR_start != -np.inf:
        plt.axvline(PIR_start, color ='r', linestyle='--')
    if PIR_end != np.inf:
        plt.axvline(PIR_end, color ='r', linestyle='--')
    if PIR_start == -np.inf:
        PIR_start = x[0]
    if PIR_end == np.inf:
        PIR_end = x[-1]
    plt.axvspan(PIR_start, PIR_end, color=None, facecolor='r', alpha=0.05)

    plt.axvline(x_vline, linestyle='-', color='k', alpha = 0.5, linewidth = 3)
    plt.scatter(x, vals[:, columns.index("N_sw")], color='g', linewidth=2, label="N swallows")
    plt.xlabel(xlabel, fontsize=27)
    plt.ylabel("Number of swallows", fontsize=24)
    plt.suptitle(title, fontsize=27)
    # plt.legend(fontsize=24, loc = 5)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(img_folder, save_to), bbox_inches="tight", transparent=True)
    plt.savefig(os.path.join(img_folder, save_to.split(".svg")[0] + '.pdf'), bbox_inches="tight", transparent=True)
    plt.show()
    plt.close()
    return None

if __name__ == '__main__':
    img_folder = os.path.join(get_project_root(), "img")

    exp_name = "Experiment_arousal"
    xlabel = r"arousal drive, %"
    title = "Varying arousal drive"
    x_vline = 100
    PIR_bnds = [-np.inf,100]
    save_to = os.path.join(img_folder, f"{exp_name}.svg")
    base_folder = os.path.join(get_project_root(), "data", "experiments", exp_name)
    plot_analytics(base_folder, xlabel, title, x_vline, PIR_bnds, save_to)