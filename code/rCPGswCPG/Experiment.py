import pickle
import numpy as np
import time
from tqdm.auto import tqdm
from rCPGswCPG.protocols.protocols import *
from rCPGswCPG.parameter_extraction.param_extraction_utils import analyse_noSI, analyse_longSI, analyse_shortSI
from rCPGswCPG.utils.gen_utils import *
import os
from rCPGswCPG.utils.utils import *
from copy import deepcopy
import os, time, pickle, numpy as np, ray
from tqdm import tqdm
from rCPGswCPG.Network import construct_model
from rCPGswCPG.construct_protocol import construct_protocol
import json
from rCPGswCPG.utils.gen_utils import array2str


class Experiment:
    '''
    Sets up model + protocols, runs sweeps, saves recordings, plots, analyzes.
    '''

    # --- tiny utils kept on the class to avoid external deps ---
    def __init__(self, config_dict, base_folder):
        self.base_folder  = base_folder
        self.data_folder  = os.path.join(base_folder, "runs")
        self.img_folder   = os.path.join(base_folder, "imgs")
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.img_folder, exist_ok=True)

        self.description   = config_dict["description"]
        self.model_params  = config_dict["model_params"]
        self.varied_params = config_dict["varied_params"]
        self.param_points  = config_dict["param_points"]
        self.protocol_dict = config_dict["protocol_dict"]
        config_dict["date"]= time.strftime("%Y%m%d-%H%M%S")

        # construct a model + protocols
        self.model = construct_model(self.model_params)
        names      = list(self.protocol_dict.keys())
        params     = list(self.protocol_dict.values())
        self.protocols = [construct_protocol(self.model, names[i], params[i]) for i in range(len(names))]

        # save meta
        with open(os.path.join(self.base_folder, "description.txt"), "w+") as f: f.write(self.description)
        pickle.dump(config_dict, open(os.path.join(self.base_folder, "config_file.pkl"), "wb+"))

    # robust setter that accepts: W["A","B"], W[A,B], W[2,5], or self.model.pnames.index("A") forms
    _name_re = re.compile(r'index\(\s*["\']([^"\']+)["\']\s*\)')
    _w_re    = re.compile(r'W\s*\[\s*(.+?)\s*,\s*(.+?)\s*\]')

    def _resolve_idx(self, token):
        m = self._name_re.search(token)
        if m: return self.model.pnames.index(m.group(1))
        t = token.strip().strip('"').strip("'")
        if t.lstrip('-').isdigit(): return int(t)
        return self.model.pnames.index(t)

    def set_param(self, name, value):
        if 'W' in name and '[' in name and ']' in name:
            m = self._w_re.search(name)
            if not m: 
                raise ValueError(f"Unrecognized W spec: {name}")
            i = self._resolve_idx(m.group(1))
            j = self._resolve_idx(m.group(2))
            self.model.W[i, j] = float(value)
            self.model.sync_params()
            vij = self.model.W[i, j]
            if not np.isclose(vij, float(value), rtol=0, atol=1e-12):
                print("WARN: W mismatch right after set:",
                      name, "expected", float(value), "got", vij)
            return None
        if name.endswith(".drive"):
            pop = name.split('"')[1] if '"' in name else name.split("'")[1]
            k = self.model.pnames.index(pop); self.model.populations[k].drive = float(value)
            self.model.sync_params(); return None
        if name.endswith(".tau_m"):
            pop = name.split('"')[1] if '"' in name else name.split("'")[1]
            k = self.model.pnames.index(pop); self.model.populations[k].tau_m = float(value)
            self.model.sync_params(); return None
        raise ValueError(f"Unsupported parameter spec: {name}")

    def set_params_batch(self, names, values):
        for n, v in zip(names, values):
            if 'W' in n and '[' in n and ']' in n:
                m = self._w_re.search(n)
                i = self._resolve_idx(m.group(1))
                j = self._resolve_idx(m.group(2))
                self.model.W[i, j] = float(v)
            elif n.endswith(".drive"):
                pop = n.split('"')[1] if '"' in n else n.split("'")[1]
                k = self.model.pnames.index(pop)
                self.model.populations[k].drive = float(v)
            elif n.endswith(".tau_m"):
                pop = n.split('"')[1] if '"' in n else n.split("'")[1]
                k = self.model.pnames.index(pop)
                self.model.populations[k].tau_m = float(v)
            else:
                raise ValueError(f"Unsupported parameter spec: {n}")
        self.model.W = np.asarray(self.model.W, dtype=np.float64)
        self.model.sync_params(); return None

    # --- serial run (unchanged, with clear_history per protocol) ---
    def run(self, rerun=False):
        save_dir = os.path.join(self.data_folder, "recordings")
        os.makedirs(save_dir, exist_ok=True)
        if os.listdir(save_dir) and not rerun:
            print(f"Warning: The directory {save_dir} is not empty"); return None
        if self.param_points is None or (hasattr(self.param_points, "__len__") and len(self.param_points) == 0):
            self.param_points = [None]
        for i, vals in tqdm(enumerate(self.param_points), total=len(self.param_points),
                            desc="Running experiments for parameter points"):
            if not self.varied_params is None and len(self.varied_params) > 0:
                self.set_params_batch(self.varied_params, vals)
                rec = {"param_point": vals, "protocol_runs": {}}
                fname = f"{str(i).zfill(3)}_recordings_{array2str(vals)}.pkl"
            else:
                rec = {"param_point": None, "protocol_runs": {}}
                fname = f"{str(i).zfill(3)}_recordings_default_params.pkl"
            
            for prot in self.protocols:
                if hasattr(self.model, "clear_history"):
                    self.model.clear_history()
                prot.run()
                rec["protocol_runs"][prot.name] = self.model.get_recordings()
            with open(os.path.join(save_dir, fname), "wb+") as f:
                pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL)
        return None


    def analyse_data(self):
        data_table = {"columns": ["Ti", "Ti_std", "Te", "Te_std", "Ttot", "Ttot_std", "spont_swallows",
                                  "N_sw", "N_br", "time_to_1st_sw", "time_to_2nd_sw", "N_sw_shortSI", "PIR"]}
        if self.varied_params is None or len(self.varied_params) == 0:
            print("No varied params; skipping analysis")
            return None
        for n in self.varied_params: data_table["columns"].append(n)
        data_table_vals = []
        for i, param_point in tqdm(enumerate(self.param_points)):
            num_str = str(i).zfill(3)
            param_str = array2str(param_point)
            filename = f"{num_str}_recordings_{param_str}.pkl"
            rec_file = os.path.join(self.data_folder, "recordings", filename)
            if not os.path.isfile(rec_file):
                print(f"Warning: Recording file {rec_file} not found")
                continue
            recordings = pickle.load(open(rec_file, "rb+"))
            protocol_names = list(recordings["protocol_runs"].keys())
            data_entry = []
            for pname in protocol_names:
                if pname == "Protocol_noSI":
                    r = recordings["protocol_runs"][pname]
                    idx = [self.model.pnames.index("Insp"), self.model.pnames.index("Exp"), self.model.pnames.index("Sw1")]
                    fr  = r["fr_history"][:, idx]
                    pnames = ["Insp", "Exp", "Sw1"]
                    data_entry.extend(analyse_noSI(self.model.dt, fr, pnames))
                elif pname == "Protocol_longSI":
                    r = recordings["protocol_runs"][pname]
                    idx = [self.model.pnames.index("Insp"), self.model.pnames.index("Sw1"), self.model.pnames.index("Sw2")]
                    fr  = r["fr_history"][:, idx]
                    pnames = ["Insp", "Sw1", "Sw2"]
                    data_entry.extend(analyse_longSI(self.model.dt, fr, pnames))
                elif pname == "Protocol_shortSI":
                    r = recordings["protocol_runs"][pname]
                    idx = [self.model.pnames.index("Sensory_relay"), self.model.pnames.index("Sw1")]
                    fr  = r["fr_history"][:, idx]
                    pnames = ["Sensory_relay", "Sw1"]
                    data_entry.extend(analyse_shortSI(self.model.dt, fr, pnames))
                elif pname == "Protocol_LongShortSI":
                    pass
                else:
                    raise NameError("No such protocol")
            data_entry.extend(param_point); data_table_vals.append(deepcopy(data_entry))
            data_table["vals"] = np.array(data_table_vals)
        pickle.dump(deepcopy(data_table), open(os.path.join(self.base_folder, "data_table.pkl"), "wb+"))
        return None

    def plot_traces(self, param_names_tuple, show=False):
        files = os.listdir(os.path.join(self.data_folder, "recordings"))
        for i, file in enumerate(files):
            rec_file = os.path.join(self.data_folder, "recordings", file)
            recordings = pickle.load(open(rec_file, "rb+"))
            protocol_names = list(recordings["protocol_runs"].keys())
            t = np.array([0])
            for j in range(len(protocol_names)): t = np.append(t, recordings["protocol_runs"][protocol_names[j]]['t'] + t[-1])
            t = t[1:]
            fr = np.vstack([recordings["protocol_runs"][protocol_names[i]]['fr_history'] for i in range(len(protocol_names))])
            param_point = recordings["param_point"]
            VNA_components = {"KF_phasic": 1.0, "Sw1": 0.6, "Insp": 0.65}
            fig, axes = plot_data(t, fr, self.model.pnames, self.model.pnames, VNA_components)

            if not (param_names_tuple is None) and (hasattr(self.param_points, "__len__") and len(self.param_points) == 0):
                    fig.suptitle(f"{param_names_tuple} = {array2str(param_point)}", fontsize=25)
            name = f"recordings_{array2str(param_point)}"
            plt.savefig(os.path.join(self.img_folder, f"{str(i).zfill(3)}_{name}.png"))
            if show: plt.show(block=True)
            plt.close(fig)
        return None

    def plot_analytics_1d(self):
        data_file = os.path.join(self.base_folder, "data_table.pkl")
        img_folder = os.path.join(self.base_folder, "imgs", "analysis")
        xlabel = get_short_name(self.varied_params[0])
        os.makedirs(img_folder, exist_ok=True)
        data_table = pickle.load(open(data_file, "rb+"))
        columns, vals = data_table["columns"], data_table["vals"]
        x = vals[:, -1]
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, vals[:, columns.index("N_sw")], color='r', linewidth=2, label="N_sw")
        plt.plot(x, vals[:, columns.index("N_br")], color='b', linewidth=2, label="N_br")
        plt.plot(x, vals[:, columns.index("N_sw_shortSI")], color='orange', linewidth=2, label="N_sw_shortSI")
        plt.scatter(x, vals[:, columns.index("spont_swallows")], label="Spont_swallows")
        plt.scatter(x, vals[:, columns.index("PIR")], color='magenta', label="PIR")
        plt.xlabel(xlabel, fontsize=24); plt.ylabel("T", fontsize=24)
        plt.suptitle("Swallowing behaviour on varied parameters", fontsize=24)
        plt.legend(fontsize=24); plt.grid(True)
        fig.savefig(os.path.join(img_folder, "swallows.png")); plt.close()

        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, vals[:, columns.index("time_to_1st_sw")], color='r', linewidth=2, label="time_to_1st_sw")
        plt.plot(x, vals[:, columns.index("time_to_2nd_sw")], color='b', linewidth=2, label="time_to_2nd_sw")
        plt.xlabel(xlabel, fontsize=24); plt.ylabel("T", fontsize=24)
        plt.suptitle("Swallowing behaviour on varied parameters", fontsize=24)
        plt.legend(fontsize=24); plt.grid(True)
        fig.savefig(os.path.join(img_folder, "time_to_1st_and_2nd_swallow.png")); plt.close()

        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, vals[:, columns.index("Ti")],   color='r', linewidth=2, label="Ti")
        plt.plot(x, vals[:, columns.index("Te")],   color='b', linewidth=2, label="Te")
        plt.plot(x, vals[:, columns.index("Ttot")], color='k', linewidth=2, label="Ttot")
        plt.xlabel(xlabel, fontsize=24); plt.ylabel("T", fontsize=24)
        plt.suptitle("Dependence of T on varied parameters", fontsize=24)
        plt.legend(fontsize=24); plt.grid(True)
        fig.savefig(os.path.join(img_folder, "Dependence_on_T.png")); plt.close()
        return None


