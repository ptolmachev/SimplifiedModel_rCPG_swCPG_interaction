import pickle
import numpy as np
import time
from tqdm.auto import tqdm
from src.construct_model import construct_model
from src.construct_protocol import construct_protocol
from src.exp_protocols.protocols import *
from src.parameter_extraction.param_extraction_utils import analyse_noSI, analyse_longSI, analyse_shortSI
from src.utils.gen_utils import *
import os
from matplotlib import pyplot as plt
from src.utils.utils import *
from copy import deepcopy

class Experiment():
    '''
    Class experiment, accepts the parameters of
    -the input model
    -the varied parameters, and their range
    -the experimental protocols
    methods:
    -initialises all the classes (model and protocols) and prepare the folders
    -runs the experiments and save the recordings into a folder
    -plots the traces
    -analyses the recordings and extracts the parameters out of them
    '''

    def __init__(self, config_dict, base_folder):
        self.base_folder = base_folder
        self.data_folder = os.path.join(base_folder, "runs")
        self.img_folder = os.path.join(base_folder, "imgs")
        create_dir_if_not_exist(self.data_folder)
        create_dir_if_not_exist(self.img_folder)

        self.description = config_dict["description"]
        self.model_params = config_dict["model_params"]
        self.varied_params = config_dict["varied_params"]
        self.param_points = config_dict["param_points"] # the list of points with concrete parameter-values as coords
        # a dictionary of protocols to run (key: name, val: parameters)
        self.protocol_dict = config_dict['protocol_dict']
        config_dict["date"] = time.strftime("%Y%m%d-%H%M%S")

        #construct a model
        self.model = construct_model(self.model_params)
        # set up the experimental protocols
        protocol_names = list(self.protocol_dict.keys())
        protocol_params = list(self.protocol_dict.values())
        self.protocols = [
            construct_protocol(self.model,protocol_names[i], protocol_params[i])
            for i in range(len(protocol_names))
        ]

        #save description and a config file
        text_file = open(os.path.join(self.base_folder, "description.txt"), "w+")
        text_file.write(self.description)
        text_file.close()
        pickle.dump(config_dict, open(os.path.join(self.base_folder, "config_file.pkl"), "wb+"))

    def run(self):
        save_to_folder = os.path.join(f"{self.data_folder}", "recordings")
        # check if the folder is empty
        create_dir_if_not_exist(save_to_folder)
        if len(os.listdir(save_to_folder)) != 0:
            raise FileExistsError(f"The directory {save_to_folder} is not empty")
        else:
            # assign the new values to the varied parameters
            for i, param_point in tqdm(enumerate(self.param_points)):
                for j in range(len(param_point)):
                    exec(f"{self.varied_params[j]} = {param_point[j]}")

                recordings = dict()
                recordings["param_point"] = param_point
                recordings["protocol_runs"] = dict()
                file_name = f"{str.zfill(str(i), 3)}_recordings_{array_to_formatted_string(param_point)}.pkl"

                # run experimental_protocols
                for protocol in self.protocols:
                    protocol.run()
                    recordings["protocol_runs"][protocol.name] = self.model.get_recordings()
                pickle.dump(recordings, open(os.path.join(f"{save_to_folder}", f"{file_name}"), "wb+"))
        return None

    def analyse_data(self):
        # create a dicitonary with the data
        data_table = dict()
        # the list of the parameters that currently could be extracted:
        data_table["columns"] = ["Ti", "Ti_std", "Te", "Te_std", "Ttot", "Ttot_std", "spont_swallows",
                                 "N_sw", "N_br", "time_to_1st_sw", "time_to_2nd_sw", "N_sw_shortSI", "PIR"]
        for j in range(len(self.varied_params)): # add the names of varied parameters to "columns"
            data_table["columns"].append(self.varied_params[j])
        data_table_vals = []
        # for each parameter-point in the parameter_points
        for i, param_point in tqdm(enumerate(self.param_points)):
            # load the recordings.
            recodings_file = os.path.join(self.data_folder,
                          "recordings",
                          f"{str.zfill(str(i), 3)}_recordings_{array_to_formatted_string(param_point)}.pkl")
            recordings = pickle.load(open(recodings_file, "rb+"))
            protocol_names = list(recordings["protocol_runs"].keys())
            # for each protocol in the recording do the relevant analysis:
            data_entry = []
            for protocol_name in (protocol_names):
                if protocol_name == "Protocol_noSI":
                    recordings_noSI = recordings["protocol_runs"][protocol_name]
                    indices = [self.model.pnames.index("Insp"),
                               self.model.pnames.index("Exp"),
                               self.model.pnames.index("Sw1")]
                    fr = recordings_noSI["fr_history"][:, indices]
                    # Ti, Ti_std, Te, Te_std, Ttot, Ttot_std
                    data_entry.extend(analyse_noSI(self.model.dt, fr))
                elif protocol_name == "Protocol_longSI":
                    recordings_longSI = recordings["protocol_runs"][protocol_name]
                    indices = [self.model.pnames.index("Insp"),
                               self.model.pnames.index("Sw1"),
                               self.model.pnames.index("Sw2")]
                    fr = recordings_longSI["fr_history"][:, indices]
                    data_entry.extend(analyse_longSI(self.model.dt, fr))
                elif protocol_name == "Protocol_shortSI":
                    recordings_shortSI = recordings["protocol_runs"][protocol_name]
                    indices = [self.model.pnames.index("Sensory_relay"),
                               self.model.pnames.index("Sw1")]
                    fr = recordings_shortSI["fr_history"][:, indices]
                    data_entry.extend(analyse_shortSI(self.model.dt, fr))
                else:
                    raise NameError("No such protocol")
            data_entry.extend(param_point)
            data_table_vals.append(deepcopy(data_entry))
            data_table["vals"] = np.array(data_table_vals)
        pickle.dump(deepcopy(data_table), open(os.path.join(self.base_folder,"data_table.pkl"), "wb+"))
        return None

    def plot_traces(self, param_names_tuple):
        files = os.listdir(os.path.join(self.data_folder, "recordings"))
        for i, file in enumerate(files):
            if i == 25:
                x = 1
            recordings_file = os.path.join(f"{self.data_folder}", "recordings", f"{file}")
            recordings = pickle.load(open(recordings_file, "rb+"))
            protocol_names = list(recordings["protocol_runs"].keys())

            # stacking t
            t = np.array([0])
            for j in range(len(protocol_names)):
                t = np.append(t, recordings["protocol_runs"][protocol_names[j]]['t'] + t[-1])
            t = t[1:]

            fr = np.vstack([recordings["protocol_runs"][protocol_names[i]]['fr_history'] for i in range(len(protocol_names))])
            param_point = recordings["param_point"]

            VNA_components = {"KF_phasic": 1.0, "Sw1": 0.6, "Insp": 0.65}
            fig, axes = plot_data(t, fr, self.model.pnames, self.model.pnames, VNA_components)
            fig.suptitle(f"{param_names_tuple} = {array_to_formatted_string(param_point)}", fontsize=25)
            name = f"recordings_{array_to_formatted_string(param_point)}"
            plt.savefig(os.path.join(f"{self.img_folder}", f"{str.zfill(str(i), 3)}_{name}.png"))
            plt.close(fig)
        return None

    def plot_analytics_1d(self):
        data_file = os.path.join(self.base_folder, "data_table.pkl")
        img_folder = os.path.join(self.base_folder, "imgs", "analysis")
        xlabel = get_short_name(self.varied_params[0])
        create_dir_if_not_exist(img_folder)
        data_table = pickle.load(open(data_file, "rb+"))

        # 1D plots
        columns = data_table["columns"]
        vals = data_table["vals"]
        x = vals[:, -1]
        print(columns)
        # plotting 'spont_swallows', 'N_sw', 'N_br', N_sw_shortSI
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, vals[:, columns.index("N_sw")], color='r', linewidth=2, label="N_sw")
        plt.plot(x, vals[:, columns.index("N_br")], color='b', linewidth=2, label="N_br")
        plt.plot(x, vals[:, columns.index("N_sw_shortSI")], color='orange', linewidth=2, label="N_sw_shortSI")
        plt.scatter(x, vals[:, columns.index("spont_swallows")], label="Spont_swallows")
        plt.scatter(x, vals[:, columns.index("PIR")], color = 'magenta', label="PIR")
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel("T", fontsize=24)
        plt.suptitle("Swallowing behaviour on varied parameters", fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        # plt.show()
        fig.savefig(os.path.join(img_folder, "swallows.png"))
        plt.close()

        # plotting Time to 1st swallow, time to 2nd swallow
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, vals[:, columns.index("time_to_1st_sw")], color='r', linewidth=2, label="time_to_1st_sw")
        plt.plot(x, vals[:, columns.index("time_to_2nd_sw")], color='b', linewidth=2, label="time_to_2nd_sw")
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel("T", fontsize=24)
        plt.suptitle("Swallowing behaviour on varied parameters", fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        # plt.show()
        fig.savefig(os.path.join(img_folder, "time_to_1st_and_2nd_swallow.png"))
        plt.close()
        #
        # plotting T
        fig = plt.figure(figsize=(10, 5))
        plt.plot(x, vals[:, columns.index("Ti")], color='r', linewidth=2, label="Ti")
        plt.plot(x, vals[:, columns.index("Te")], color='b', linewidth=2, label="Te")
        plt.plot(x, vals[:, columns.index("Ttot")], color='k', linewidth=2, label="Ttot")
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel("T", fontsize=24)
        plt.suptitle("Dependence of T on varied parameters", fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        # plt.show()
        fig.savefig(os.path.join(img_folder, "Dependence_on_T.png"))
        plt.close()


