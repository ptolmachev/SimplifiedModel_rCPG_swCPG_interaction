#plot traces of a specific stack of files

from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from src.utils.gen_utils import get_project_root, create_dir_if_not_exist
from src.utils.utils import plot_data
data_folder = os.path.join(get_project_root(), "data",
                           "experiments", "PRC_experiments", "model_complex",
                           'PRC_short_KF_inh_stim', "num_run_short_KF_inh_stim_0.45_0.25")
img_folder = os.path.join(get_project_root(), "img",
                          "experiments", "PRC_experiments", "model_complex",
                          'PRC_short_KF_inh_stim', "num_run_short_KF_inh_stim_0.45_0.25")
create_dir_if_not_exist(img_folder)
files = os.listdir(data_folder)

for file in files:
    full_file_path = os.path.join(data_folder, file)
    data = pickle.load(open(full_file_path, "rb+"))
    pnames_to_plot = pnames = data["population_names"]
    t = data["t"]
    VNA_components = {"KF_phasic" : 0.75,  "Sw1" : 0.6, "RampI" : 0.9}
    fig, axes = plot_data(t, data['signals'], pnames, pnames_to_plot, VNA_components)
    img_file_name = file.split(".pkl")[0] + '.png'
    plt.savefig(os.path.join(img_folder, img_file_name))
    plt.show(block = True)
    plt.close()


