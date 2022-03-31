from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

def replace_name(name):
    if name == 'Sensory_relay':
        return "SR"
    elif name == 'KF_gate':
        return 'KFg'
    elif name == 'KF_phasic':
        return 'KFp'
    else:
        return name


def get_short_name(param_to_vary):
    if 'self.model.W' in param_to_vary:
        nrn_from = replace_name(param_to_vary.split("\"")[1])
        nrn_to = replace_name(param_to_vary.split("\"")[3])
        return f"W_{nrn_from}_to_{nrn_to}"
    elif 'populations' in param_to_vary:
        pop_name = replace_name(param_to_vary.split("\"")[1])
        param_name = param_to_vary.split(".")[-1]
        return f"{pop_name}_{param_name}"
    else:
        raise NameError(f"Name of the parameter couldnt be extraced: {param_to_vary}")


def get_val_of_param(full_param_name, model_params):
    pnames = model_params['pnames']
    if 'self.model.W' in full_param_name:
        nrn_from = full_param_name.split("\"")[1]
        nrn_to = full_param_name.split("\"")[3]
        return model_params["W"][pnames.index(nrn_from), pnames.index(nrn_to)]

    elif 'populations' in full_param_name:
        pop_name = full_param_name.split("\"")[1]
        param_name = full_param_name.split(".")[-1]
        if param_name == 'drive':
            return np.sum(model_params[f"drives_misc"], axis = 0)[pnames.index(pop_name)]
        else:
            return model_params[f"{param_name}"][pnames.index(pop_name)]
    else:
        raise NameError(f"Name of the parameter couldnt be extraced: {full_param_name}")

def plot_data(t, data, pnames, pnames_to_plot, VNA_components):
    N = len(pnames_to_plot)
    if VNA_components is None:
        M = N
    else:
        M = N+1
    fig, axes = plt.subplots(M, 1, figsize=(14, 7))
    for label in (pnames):
        if label in pnames_to_plot:
            i = pnames.index(label)
            j = pnames_to_plot.index(label)
            axes[j].plot(t, data[:, i], label=pnames_to_plot[j], color='k', linewidth=3)
            axes[j].legend(fontsize=15, loc=2)
            axes[j].set_ylim([0, 1.05])
            axes[j].set_yticks([])
    plt.suptitle("Simplified model dynamics", fontsize=25)
    if VNA_components is not None:
        VNA = np.zeros_like(data[:, pnames.index("Sw1")])
        for pop in list(VNA_components.keys()):
            VNA += VNA_components[pop] * data[:, pnames.index(pop)]
        axes[-1].plot(t, VNA, label="VNA", color='k', linewidth=3)
    axes[-1].set_xlabel("t", fontsize=15)
    axes[-1].legend(fontsize=15, loc=2)
    axes[-1].set_ylim([0, 1.05])
    axes[-1].set_yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes

