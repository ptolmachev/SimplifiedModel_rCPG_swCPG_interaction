from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from src.utils.gen_utils import get_project_root, create_dir_if_not_exist

# given the name of the experiment, it plots the 'spont_swallows', 'N_sw', 'N_br', N_sw_shortSI'

exp_name = "Varying_W_Sw2_to_Sw1"
xlabel = "W[Sw2, Sw1]"
base_folder = os.path.join(f"{get_project_root()}", "data", "experiments", f"{exp_name}")
data_file = os.path.join(get_project_root(), "data", "experiments", f"{exp_name}", "data_table.pkl")
data_table = pickle.load(open(data_file, "rb+"))


columns = data_table["columns"]
vals = data_table["vals"]
x = vals[:, -1]
# plotting 'spont_swallows', 'N_sw', 'N_br', N_sw_shortSI
fig = plt.figure(figsize = (10, 5))
plt.plot(x, vals[:, columns.index("N_sw")], color='r', linewidth=2,  label = "N_sw")
# plt.plot(x, vals[:, columns.index("N_br")], color='b', linewidth=2, label = "N_br")
# plt.plot(x, vals[:, columns.index("N_sw_shortSI")], color='orange', linewidth=2, label = "N_sw_shortSI")
avg_sw = np.mean(vals[:, columns.index("N_sw")])
plt.scatter(x, avg_sw * vals[:, columns.index("spont_swallows")], label = "Spont_swallows")
plt.xlabel(xlabel, fontsize = 24)
plt.ylabel("Number of Swallows", fontsize = 24)
plt.suptitle("Swallowing behaviour on varied parameters", fontsize = 24)
plt.legend(fontsize = 24)
plt.grid(True)
plt.show(block = True)
plt.close()