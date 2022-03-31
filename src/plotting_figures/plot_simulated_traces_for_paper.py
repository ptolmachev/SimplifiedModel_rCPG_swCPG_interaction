from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from src.utils.gen_utils import get_project_root

#open a specific folder and load the data


exp_name = "Varying_W_Sw2_to_Sw1"
file_name = "024_recordings_-0.51.pkl" # for Varying_W_Sw2_to_Sw1 trace 2

# file_name = "010_recordings_0.3 _0.56_0.4 .pkl" # trace 1 arousal
# file_name = "040_recordings_0.08_0.14_0.1 .pkl" #trace 2 arousal

# file_name = "027_recordings_0.3.pkl" # for drive to Sw2 2nd trace
# file_name = "035_recordings_0.21.pkl" # for drive to Sw1


data_file = os.path.join(get_project_root(),
                         "data", "experiments", exp_name, "runs", "recordings", file_name)
img_folder = os.path.join(get_project_root(),"img")
data = pickle.load(open(data_file, "rb+"))
protocol_names = ["Protocol_noSI", "Protocol_longSI", "Protocol_shortSI"]
pnames = data["protocol_runs"]["Protocol_noSI"]["population_names"]
dt = data["protocol_runs"]["Protocol_noSI"]["dt"]
t = np.empty([])
Sw1 = np.empty([])
RampI = np.empty([])
KF_phasic = np.empty([])
Relay = np.empty([])
ind_first_short_SI = 0
ind_long_stim_end = 0
for protocol_name in protocol_names:
    l = len(data["protocol_runs"][protocol_name]["fr_history"][:, pnames.index("RampI")])
    RampI = np.append(RampI, data["protocol_runs"][protocol_name]["fr_history"][:, pnames.index("RampI")])
    KF_phasic = np.append(KF_phasic, data["protocol_runs"][protocol_name]["fr_history"][:, pnames.index("KF_phasic")])
    Sw1 = np.append(Sw1, data["protocol_runs"][protocol_name]["fr_history"][:, pnames.index("Sw1")])
    Relay = np.append(Relay, data["protocol_runs"][protocol_name]["fr_history"][:, pnames.index("Sensory_relay")])
    if protocol_name == "Protocol_shortSI":
        ind_first_short_SI += np.array(np.where(np.abs(np.diff(data["protocol_runs"][protocol_name]["fr_history"][:, pnames.index("Sensory_relay")]))>0.1)[0])[0]
    else:
        ind_first_short_SI += l
        ind_long_stim_end += l
ind_long_stim_end +=1
stop_ind = ind_first_short_SI
t = dt * np.arange(len(Relay))
VNA = 0.9*RampI + 0.75*KF_phasic + 0.6*Sw1
transient_inds = 87000
x = np.array(np.where(np.abs(np.diff(Relay[transient_inds:stop_ind])) > 0.1)[0])
x = np.append(x, [10])
x = x[np.where(np.abs(np.diff(x)) > 1)[0]]
fig = plt.figure(figsize=(10,2))
l = x.shape[0]
ind = 0

while ind < l:
    try:
        plt.axvline(x[ind],color="r")
        plt.axvline(x[ind+1],color="r")
        plt.axvspan(x[ind], x[ind+1], color="r", alpha=0.1)
        ind+=2
    except:
        break

# plot long stimulus effect
fig = plt.plot(figsize = (9, 2))
plt.plot(VNA[transient_inds:stop_ind], linewidth = 2, color="k")
plt.xticks([])
plt.yticks([])
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
plt.axis('off')
plt.savefig(os.path.join(img_folder, f"{exp_name}_{file_name}_long_stimulus.pdf"),bbox_inches='tight')
plt.show()
plt.close()


x = np.array(np.where(np.abs(np.diff(Relay[ind_long_stim_end:])) > 0.1)[0])
x = np.append(x, [10])
x = x[np.where(np.abs(np.diff(x)) > 1)[0]]
fig = plt.figure(figsize=(10,2))
l = x.shape[0]
ind = 0

while ind < l:
    try:
        plt.axvline(x[ind],color="r")
        plt.axvline(x[ind+1],color="r")
        plt.axvspan(x[ind], x[ind+1], color="r", alpha=0.1)
        ind+=2
    except:
        break

# plot short stimulus effect
fig = plt.plot(figsize = (9, 2))
plt.plot(VNA[ind_long_stim_end:], linewidth = 2, color="k")
plt.xticks([])
plt.yticks([])
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
plt.axis('off')
plt.savefig(os.path.join(img_folder, f"{exp_name}_{file_name}_short_stimulus.pdf"),bbox_inches='tight')
plt.show()
plt.close()

