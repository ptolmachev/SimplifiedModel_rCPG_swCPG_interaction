import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from src.utils.gen_utils import get_project_root, create_dir_if_not_exist

model_name = "full_model"
exp_name = "Varying_Sw2_drive"
xlabel = "Sw2_drive"
data_file = os.path.join(get_project_root(), "data", "experiments", f"{exp_name}", "data_table.pkl")
img_folder = os.path.join(get_project_root(), "data", "experiments", f"{exp_name}", "imgs", "analysis")
create_dir_if_not_exist(img_folder)
data_table = pickle.load(open(data_file, "rb+"))

# 1D plots
columns = data_table["columns"]
vals = data_table["vals"]
x = vals[:, -1]
print(columns)
# plotting 'spont_swallows', 'N_sw', 'N_br', N_sw_shortSI
fig = plt.figure(figsize = (10, 5))
plt.plot(x, vals[:, columns.index("N_sw")], color='r', linewidth=2,  label = "N_sw")
plt.plot(x, vals[:, columns.index("N_br")], color='b', linewidth=2, label = "N_br")
plt.plot(x, vals[:, columns.index("N_sw_shortSI")], color='orange', linewidth=2, label = "N_sw_shortSI")
plt.scatter(x, vals[:, columns.index("spont_swallows")], label = "Spont_swallows")
plt.xlabel(xlabel, fontsize = 24)
plt.ylabel("T", fontsize = 24)
plt.suptitle("Swallowing behaviour on varied parameters", fontsize = 24)
plt.legend(fontsize = 24)
plt.grid(True)
plt.show()
fig.savefig(os.path.join(img_folder, "swallows.png"))
plt.close()

# plotting Time to 1st swallow, time to 2nd swallow
fig = plt.figure(figsize = (10, 5))
plt.plot(x, vals[:, columns.index("time_to_1st_sw")], color='r', linewidth=2,  label = "time_to_1st_sw")
plt.plot(x, vals[:, columns.index("time_to_2nd_sw")], color='b', linewidth=2, label = "time_to_2nd_sw")
plt.xlabel(xlabel, fontsize = 24)
plt.ylabel("T", fontsize = 24)
plt.suptitle("Swallowing behaviour on varied parameters", fontsize = 24)
plt.legend(fontsize = 24)
plt.grid(True)
plt.show()
fig.savefig(os.path.join(img_folder, "time_to_1st_and_2nd_swallow.png"))
plt.close()
#
# plotting T
fig = plt.figure(figsize = (10, 5))
plt.plot(x, vals[:, columns.index("Ti")], color='r', linewidth=2, label = "Ti")
plt.plot(x, vals[:, columns.index("Te")], color='b', linewidth=2,  label = "Te")
plt.plot(x, vals[:, columns.index("Ttot")], color='k', linewidth=2, label = "Ttot")
plt.xlabel(xlabel, fontsize = 24)
plt.ylabel("T", fontsize = 24)
plt.suptitle("Dependence of T on varied parameters", fontsize = 24)
plt.legend(fontsize = 24)
plt.grid(True)
plt.show()
fig.savefig(os.path.join(img_folder, "Dependence_on_T.png"))
plt.close()
# 2d plots
#
# model_name = "full_model"
# exp_name = "Experiment_varying_connectivity_to_Insp"
# data_file = os.path.join(get_project_root(), "data", "experiments", f"{exp_name}", "data_table.pkl")
# data_table = pickle.load(open(data_file, "rb+"))
# columns = data_table["columns"]
# print(columns)
#
# vals = data_table["vals"]
# param_to_plot = "N_sw_shortSI"
# x = vals[:, -2].reshape(11, 11)
# y = vals[:, -1].reshape(11, 11)
# z = vals[:, columns.index(param_to_plot)].reshape(11, 11)
# plt.contourf(x, y, z, cmap='viridis')
# plt.xlabel("W[SI, Insp]", fontsize = 24)
# plt.ylabel("W[Sw1, Insp]", fontsize = 24)
# plt.suptitle(f"{param_to_plot} on params", fontsize = 24)
# plt.colorbar()
# plt.show()