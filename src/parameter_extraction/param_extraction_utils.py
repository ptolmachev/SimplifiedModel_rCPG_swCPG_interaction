import numpy as np
from matplotlib import pyplot as plt
import ruptures as rpt
from scipy import signal
from sklearn.cluster import KMeans, DBSCAN
from copy import copy as cp

def flip(zeros_and_ones):
    return -(zeros_and_ones - 0.5) + 0.5

def relabel(raw_labels, lookup_table):
    return [cp(lookup_table[int(raw_label)]) for raw_label in (raw_labels)]

def calc_streaks(time_series, label_to_count):
    count = False
    streaks = np.array([])
    len_current_streak = 0
    for i in range(len(time_series)):
        if (time_series[i] != label_to_count):
            if count == False:
                pass
            else:
                count = False
                streaks = np.append(streaks, len_current_streak)
                len_current_streak = 0
        else:
            if count == False:
                count = True
            else:
                pass
            len_current_streak += 1
    return streaks


def analyse_noSI(dt, fr):
    down_factor = 10

    decimated_fr = (signal.decimate(fr.T, q=down_factor)).T
    Sw = decimated_fr[:, 2]
    insp_exp =decimated_fr[:, :2]
    dt = dt * down_factor

    #swallows detection
    transients = int(250/dt) #250 ms
    swallows = signal.find_peaks(Sw[transients:], height = 0.1, width=5)[0]
    if len(swallows) != 0:
        spont_swallows = True
    else:
        spont_swallows = False

    # characterisation of breathing
    decider = KMeans(n_clusters=2)
    decider.fit(insp_exp)
    raw_labels = decider.labels_
    cluster_centres = decider.cluster_centers_
    if cluster_centres[0, 0] < cluster_centres[0, 1]:
        labels = relabel(raw_labels, ["Exp", "Insp"])
    else:
        labels = relabel(raw_labels, ["Insp", "Exp"])

    streaks_insp = calc_streaks(labels, label_to_count="Insp")
    streaks_exp = calc_streaks(labels, label_to_count="Exp")
    # what if there is one cluster?

    Ti = np.median(streaks_insp) * dt
    Ti_std = np.std(streaks_insp) * dt**2
    Te = np.median(streaks_exp) * dt
    Te_std = np.std(streaks_exp) * dt**2
    Ttot = Ti + Te
    Ttot_std = Ti_std + Te_std
    return Ti, Ti_std, Te, Te_std, Ttot, Ttot_std, spont_swallows

def analyse_longSI(dt, fr):
    down_factor = 10

    decimated_fr = (signal.decimate(fr.T, q=down_factor)).T
    Sw = decimated_fr[:, 1:]
    insp = decimated_fr[:, 0]

    dt = dt * down_factor
    decider = KMeans(n_clusters=2)
    decider.fit(Sw)
    raw_labels_sw = decider.labels_
    cluster_centres = decider.cluster_centers_
    if cluster_centres[0, 0] < cluster_centres[0, 1]:
        labels = relabel(raw_labels_sw, ["Sw2", "Sw1"])
    else:
        labels = relabel(raw_labels_sw, ["Sw1", "Sw2"])

    # number of swallows
    swallows = signal.find_peaks(Sw[:, 0], height=0.1, width=5)[0]
    N_sw = len(swallows)
    # get time to the first swallow and time to the second
    streaks_sw1 = calc_streaks(labels, label_to_count="Sw1")
    streaks_sw2 = calc_streaks(labels, label_to_count="Sw2")
    if labels[0] == 'Sw1':
        t1 = labels.index("Sw1")
        if 'Sw1' in labels[int(streaks_sw1[0]):]:
            t2 = labels[int(streaks_sw1[0]):].index("Sw1") + int(streaks_sw1[0])
        else:
            t2 = np.inf
    else:
        t1 = labels.index("Sw1")
        if 'Sw1' in labels[int(streaks_sw1[0] + streaks_sw2[0]):]:
            t2 = labels[int(streaks_sw1[0] + streaks_sw2[0]):].index("Sw1") + int(streaks_sw2[0] + streaks_sw1[0])
        else:
            t2 = np.inf
    #breakthroughts detection
    insp_breakthroughs = signal.find_peaks(insp, height = 0.1, width=5)[0]
    N_br = len(insp_breakthroughs)
    return N_sw, N_br, t1*dt, t2*dt

def analyse_shortSI(dt, fr):
    down_factor = 10

    decimated_fr = (signal.decimate(fr.T, q=down_factor)).T
    SI = decimated_fr[:, 0]
    Sw1 = decimated_fr[:, 1]
    swallow_peaks = signal.find_peaks(Sw1, height=0.1, width=5)[0]
    # analyse the presence of rebound swallows
    tmp = np.diff(SI)
    inds_stim_start = np.where(tmp > 0.3)[0]
    inds_stim_end = np.where(tmp < -0.3)[0]
    window = int((40/dt)) #40 ms
    Mean_sw_traces_after_stimulus = []
    for i in range(len(inds_stim_end)):
        Mean_sw_traces_after_stimulus.append(np.mean(Sw1[inds_stim_end[i]:inds_stim_end[i]+window]))

    m1 = np.mean(Mean_sw_traces_after_stimulus)
    # m2 = np.mean(Sw1[inds_stim_end[-1] + window:])
    PIR = True if m1 > 0.1 else False

    # plt.plot(Sw1)
    # plt.plot(SI)
    # plt.title(f"PIR = {PIR}, m1 = {m1}")
    # plt.show(block = True)

    N_sw = len(swallow_peaks)
    return N_sw, PIR
