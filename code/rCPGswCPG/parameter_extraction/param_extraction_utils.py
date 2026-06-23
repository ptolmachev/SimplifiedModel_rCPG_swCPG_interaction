import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans, DBSCAN
from copy import copy as cp

def flip(zeros_and_ones):
    return -(zeros_and_ones - 0.5) + 0.5

def relabel(raw_labels, lookup_table):
    return [cp(lookup_table[int(raw_label)]) for raw_label in (raw_labels)]

def label_segments(labels):
    segs = []
    if not labels: 
        return segs
    start = 0
    cur = labels[0]
    for i, lbl in enumerate(labels[1:], 1):
        if lbl != cur:
            segs.append((cur, start, i))  # end is exclusive
            cur, start = lbl, i
    segs.append((cur, start, len(labels)))
    return segs

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

def analyse_noSI(dt, fr, pnames):
    down_factor = 10

    decimated_fr = (signal.decimate(fr.T, q=down_factor)).T
    Sw = decimated_fr[:, pnames.index("Sw1")]
    insp_exp = np.hstack([decimated_fr[:, pnames.index("Insp")].reshape(-1, 1), 
                decimated_fr[:, pnames.index("Exp")].reshape(-1, 1)])
    dt = dt * down_factor

    #swallows detection
    transients = int(250/dt) #250 ms
    swallows = signal.find_peaks(Sw[transients:], height = 0.07, width=5)[0]
    
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

    segs = label_segments(labels)

    streaks_insp = calc_streaks(labels, label_to_count="Insp")
    streaks_exp = calc_streaks(labels, label_to_count="Exp")

    Ti = np.median(streaks_insp) * dt
    Ti_std = np.std(streaks_insp) * dt
    Te = np.median(streaks_exp) * dt
    Te_std = np.std(streaks_exp) * dt
    Ttot = Ti + Te
    Ttot_std = Ti_std + Te_std

    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(Sw, color='g', label="Sw1")
    # ax.plot(insp_exp[:, 0], color='b', label="Insp")
    # ax.plot(insp_exp[:, 1], color='r', label="Exp")
    # for seg in segs:
    #     if seg[0] == "Insp":
    #         ax.axvspan(seg[1], seg[2], color='b', alpha=0.1)
    #     else:
    #         ax.axvspan(seg[1], seg[2], color='r', alpha=0.1)
    # ax.set_xlabel("Time (a.u.)")
    # ax.set_ylabel("Firing rate (a.u.)")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_title(f"T_i={Ti:.2f}s, T_e={Te:.2f}s, T_tot={Ttot:.2f}s, spont_sw={spont_swallows}")
    # plt.legend(fontsize=12)
    # plt.grid(False)
    # plt.show()

    return Ti, Ti_std, Te, Te_std, Ttot, Ttot_std, spont_swallows

def analyse_longSI(dt, fr, pnames):
    down_factor = 10

    decimated_fr = (signal.decimate(fr.T, q=down_factor)).T
    Sw = np.hstack([decimated_fr[:, pnames.index("Sw1")].reshape(-1, 1), 
                   decimated_fr[:, pnames.index("Sw2")].reshape(-1, 1)])
    insp = decimated_fr[:, pnames.index("Insp")]

    dt = dt * down_factor
    decider = KMeans(n_clusters=2)
    decider.fit(Sw)
    raw_labels_sw = decider.labels_
    cluster_centres = decider.cluster_centers_
    if cluster_centres[0, 0] < cluster_centres[0, 1]:
        labels = relabel(raw_labels_sw, ["Sw2", "Sw1"])
    else:
        labels = relabel(raw_labels_sw, ["Sw1", "Sw2"])
    segs = label_segments(labels)
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
            try:
                t2 = labels[int(streaks_sw1[0] + streaks_sw2[0]):].index("Sw1") + int(streaks_sw2[0] + streaks_sw1[0])
            except:
                t2 = np.inf
        else:
            t2 = np.inf

    #breakthroughts detection
    insp_breakthroughs = signal.find_peaks(insp, height = 0.25, width=10)[0]
    N_br = len(insp_breakthroughs)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(Sw[:, 0], color='r', label="Sw1")
    # ax.plot(Sw[:, 1], color='g', label="Sw2")
    # ax.plot(insp, color='b', label="Insp")
    # for seg in segs:
    #     if seg[0] == "Sw1":
    #         ax.axvspan(seg[1], seg[2], color='r', alpha=0.1)
    #     else:
    #         ax.axvspan(seg[1], seg[2], color='g', alpha=0.1)
    # ax.set_xlabel("Time (a.u.)")
    # ax.set_ylabel("Firing rate (a.u.)")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_title(f"N_sw={N_sw}, insp_breakthroughs={N_br}")
    # plt.legend(fontsize=12)
    # plt.grid(False)
    # plt.show()

    return N_sw, N_br, t1*dt, t2*dt

def analyse_shortSI(dt, fr, pnames):
    down_factor = 10

    decimated_fr = (signal.decimate(fr.T, q=down_factor)).T
    SI = decimated_fr[:, pnames.index("Sensory_relay")]
    Sw1 = decimated_fr[:, pnames.index("Sw1")]
    swallow_peaks = signal.find_peaks(Sw1, height=0.1, width=5)[0]
    # analyse the presence of rebound swallows
    tmp = np.diff(SI)
    inds_stim_start = np.where(tmp > 0.3)[0]
    inds_stim_end = np.where(tmp < -0.3)[0]
    window = int((50/dt)) #50 ms
    Mean_sw_traces_after_stimulus = []

    for i in range(len(inds_stim_end)):
        Mean_sw_traces_after_stimulus.append(np.mean(Sw1[inds_stim_end[i]:inds_stim_end[i] + window]))

    m1 = np.mean(Mean_sw_traces_after_stimulus)
    PIR = True if m1 > 0.05 else False

    N_sw = len(swallow_peaks)

    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # ax.plot(Sw1, color='r', label="Sw1")
    # ax.plot(SI, color='b', label="Sensory relay")
    # ax.set_xlabel("Time (a.u.)")
    # ax.set_ylabel("Firing rate (a.u.)")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_title(f"PIR={PIR}, N_sw={N_sw}")
    # plt.legend(fontsize=12)
    # plt.grid(False)
    # plt.show()

    return N_sw, PIR
