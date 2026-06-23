"""Sweep tau_v over {0.5, 1.0, 2.0, 5.0} for the complex model and save trace figures.

Mirrors run_model.py (same Hydra config + Experiment pipeline) but loops over
tau_v, overriding model_params.neuron_defaults.tau_v for each run, and writes one
trace figure per value to img/experiments/tau_v_sweep/tau_v_<value>.png.

Run from the repo root:  python rCPGswCPG/experiments/tau_v_sweep.py
"""
import os
import sys
import glob
import shutil

import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    # support direct script execution: python rCPGswCPG/experiments/tau_v_sweep.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rCPGswCPG.model_params.config_loader import model_params_from_cfg
from rCPGswCPG.Experiment import Experiment
from rCPGswCPG.utils.gen_utils import get_project_root

TAU_V_VALUES = [0.5, 1.0, 2.0, 5.0]


def main():
    """Run the complex model once per tau_v value and collect the trace figures.

    Returns:
        None. Writes one PNG per tau_v to <repo>/img/experiments/tau_v_sweep/.
    """
    root = str(get_project_root())
    config_dir = os.path.join(root, "configs")
    out_dir = os.path.join(root, "img", "experiments", "tau_v_sweep")
    os.makedirs(out_dir, exist_ok=True)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        for tau_v in TAU_V_VALUES:
            cfg = compose(
                config_name="config",
                overrides=[f"model_params.neuron_defaults.tau_v={tau_v}"],
            )
            model_params = model_params_from_cfg(cfg.model_params)
            base_folder = os.path.join(root, "data", "experiments", "tau_v_sweep", f"tau_v_{tau_v}")
            config_dict = {
                "model_params": model_params,
                "description": f"complex model, tau_v={tau_v}",
                "varied_params": None,
                "param_points": None,
                "protocol_dict": OmegaConf.to_container(cfg.experiment.protocol_dict, resolve=True),
            }
            exp = Experiment(config_dict, base_folder)
            exp.run(rerun=True)
            exp.plot_traces(param_names_tuple=None, show=False)

            # copy the trace figure produced by plot_traces into the sweep folder
            pngs = sorted(glob.glob(os.path.join(exp.img_folder, "*.png")))
            dst = os.path.join(out_dir, f"tau_v_{tau_v}.png")
            shutil.copyfile(pngs[0], dst)
            print(f"tau_v={tau_v}: saved {dst}")


if __name__ == "__main__":
    main()
