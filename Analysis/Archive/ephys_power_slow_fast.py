# Investigate whether there are significant differences between the power during the fast and the slow condition

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import pandas as pd
import scipy
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set parameters
########################################################################################################################
med = "Off"
plot_individual = False
tmin = -0.2
tmax = -0.1
baseline = (-0.75, -0.5)
mode = "percent"
cmap = "jet"
fmin = 70
fmax = 90
n_cycles = 4
freqs = np.arange(fmin, fmax, 2)

# Read the list of the datasets
df = pd.read_excel(f'../../Data/Dataset_list.xlsx', sheet_name=med)

# Loop through the subjects
subject_list = list(df["ID Berlin_Neurophys"][1:21])
subject_list.remove("L003")  # NO neurophys data available

# Initialize an empty list to store the power spectrums for each subject
power_all_sub = np.zeros((len(subject_list), 2, 4))
for i_sub, sub in enumerate(subject_list):

    # Load the data
    ########################################################################################################################
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
           f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
           f'rawdata/'
    bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                        extensions=".vhdr",
                                        subjects=sub,
                                        acquisitions=["StimOnB", "StimOnBDopaPre"],
                                        sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                                  f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])

    # Load dataset
    raw = read_raw_bids(bids_path=bids_path[0])
    raw.load_data()
    sfreq = raw.info["sfreq"]

    # Filter out line noise
    #raw.notch_filter(50)

    # Drop bad and not interesting channels
    ########################################################################################################################
    raw.drop_channels(raw.info["bads"])
    raw.pick_channels(raw.copy().pick_types(ecog=True, dbs=True, eeg=True).info["ch_names"] +
                      ["STIMULATION", "STIM_CONDITION", "BLOCK"])
    ch_names = raw.info["ch_names"]

    # Add average LFP channels and get target ECOG channel
    ########################################################################################################################
    for loc in ["LFP_L", "LFP_R"]:
        target_chs = [ch for ch in ch_names if (loc in ch) and (not "01" in ch) and (not "08" in ch)]
        target_ch = f"av_{loc}"
        new_ch = raw.get_data(target_chs).mean(axis=0)
        new_ch = raw.get_data(target_chs[0]).squeeze()
        u.add_new_channel(raw, new_ch[np.newaxis, :], target_ch, type="dbs")
    ECOG_target = df.loc[df["ID Berlin_Neurophys"] == sub]["ECOG_target"].iloc[0]
    ch_names = raw.info["ch_names"]

    # Extract relevant epochs
    ########################################################################################################################
    # Extract events
    events = mne.events_from_annotations(raw)[0]

    # Cut into epochs aligned to the peak speed of each movement
    event_id = 10003
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2,
                        baseline=None, reject_by_annotation=False)

    # Loop over epochs and check how far the stimulation is from the window of interest
    times = epochs.times
    is_stim = []
    for i, stim_epoch in enumerate(epochs.get_data("STIMULATION").squeeze()):
        # Get the samples at which stimulation is applied in this block
        idx_stim = np.where(stim_epoch == 1)[0]
        # Check how close the stimulation sample is to thw window of interest
        if len(idx_stim) > 0:
            # Time during which the artifact influences the power in the target window due to the wavelets
            t_extra = (n_cycles-1)/fmin
            if np.any((times[idx_stim] < tmax+t_extra) & (times[idx_stim] > tmin-t_extra)):
                is_stim.append(i)

    # Remove bad epochs (stimulation during movement)
    epochs.load_data()
    epochs.drop(is_stim)

    # Extract the block number and condition for each epoch
    epochs_tmp = epochs.copy().crop(tmin=tmin, tmax=tmax)
    block = np.unique(epochs_tmp.get_data("BLOCK"), axis=-1).flatten()
    cond = np.unique(epochs_tmp.get_data("STIM_CONDITION"), axis=-1).flatten()

    # Compute power using morlet wavelets
    ########################################################################################################################
    # Get array of channels of interest
    target_idx = [ch_names.index(ch) for ch in ["av_LFP_L", "av_LFP_R", ECOG_target]] \
        if isinstance(ECOG_target, str) else [ch_names.index(ch) for ch in ["av_LFP_L", "av_LFP_R"]]
    power = mne.time_frequency.tfr_morlet(epochs, n_cycles=n_cycles, return_itc=False,
                                          picks=target_idx,
                                          freqs=freqs, average=False, use_fft=True)

    # Apply baseline correction
    #power.apply_baseline(baseline=baseline, mode=mode)

    # Average the power in the determined window
    power.crop(tmin=tmin, tmax=tmax)
    power_perc = np.percentile(power.data, 95, axis=[-1, -2])[:, :, np.newaxis, np.newaxis]
    power.average(dim="times").average(dim="freqs")
    #power._data = power_perc

    # Determine whether there is a significant difference between
    # the power in the stim slow and fast condition
    ########################################################################################################################

    # Compute for 4 channels (LFP Contra/Ipsilateral, ECOG Contra/Ipsilateral)
    hand = df.loc[df["ID Berlin_Neurophys"] == sub]["Hand"].iloc[0]
    LFP_ipsi = "av_LFP_R" if hand == "R" else "av_LFP_L"
    LFP_contra = "av_LFP_L" if hand == "R" else "av_LFP_R"
    ECOG_ipsi = ECOG_target if (isinstance(ECOG_target, str) and
                                ((hand == "R" and "R" in ECOG_target) or(hand == "L" and "L" in ECOG_target))) else ""
    ECOG_contra = ECOG_target if (isinstance(ECOG_target, str) and
                                  ((hand == "L" and "R" in ECOG_target) or (hand == "R" and "L" in ECOG_target))) else ""
    target_chs_all = [ECOG_contra, ECOG_ipsi, LFP_contra, LFP_ipsi]
    target_chs = [ch for ch in target_chs_all if ch != ""]

    # Loop over channels
    cond_names = ["Slow", "Fast"]
    color_slow = "#00863b"
    color_fast = "#3b0086"
    power_change_chs = []
    for i, ch in enumerate(target_chs):
        power_data = power.copy().pick(ch).data.squeeze()

        # Loop over the conditions
        power_change_conds = []
        for c in range(2):
            # Get the power values for all epochs in the first block
            cond_power = power_data[(cond == c) & ((block == 1) | (block == 3))]
            # Delete first 5 moves
            cond_power = cond_power[5:]
            # Normalize power to average power of next 5 moves
            #power_change = cond_power
            power_change = u.norm_perc(cond_power)
            # Compute the average power
            mean_power = power_change.mean(axis=-1)
            # Save change in power
            power_change_conds.append(power_change)

            # Save data for group level analysis
            power_all_sub[i_sub, c, target_chs_all.index(ch)] = mean_power

            # Save change in power for one channel
        power_change_chs.append(power_change_conds)

    # If needed, create a plot for the subject
    if plot_individual:
        plt.figure()
        pos_list = [[1.25, 1.75], [2.25, 2.75], [3.25, 3.75]]
        for i, ch in enumerate(target_chs):
            slow = power_change_chs[i][0]
            fast = power_change_chs[i][1]

            # Plot mean power (change)
            plt.bar(pos_list[i][0], slow.mean(), color=color_slow, label="Slow", width=0.5)
            plt.errorbar(pos_list[i][0], slow.mean(), yerr=slow.std(), c=color_slow)
            plt.bar(pos_list[i][1], fast.mean(), color=color_fast, label="Fast", width=0.5)
            plt.errorbar(pos_list[i][1], fast.mean(), yerr=fast.std(), c=color_fast)

        # Adjust plot
        plt.xticks(np.array(pos_list[:len(target_chs)]).mean(axis=-1), target_chs)
        plt.ylabel(f"average % change in power t={tmin}-{tmax}, f={fmin}-{fmax}")
        plt.title(sub)
        plt.subplots_adjust(bottom=0.2)
        #plt.legend()
        #plt.show()
        # Save
        plt.savefig(f"../../Figures/{sub}_slow_fast_{tmin}-{tmax},_{fmin}-{fmax}.png", format="png", bbox_inches="tight", transparent=True)
        #plt.close()


# Perform group level analysis
plt.figure()
pos_list = [[1.25, 1.75], [2.25, 2.75], [3.25, 3.75], [4.25, 4.75]]
for i in range(4):
    slow = power_all_sub[:, 0, i]
    fast = power_all_sub[:, 1, i]

    # Plot mean power (change)
    plt.bar(pos_list[i][0], slow.mean(), color=color_slow, label="Slow", width=0.5)
    plt.errorbar(pos_list[i][0], slow.mean(), yerr=slow.std(), c=color_slow)
    plt.bar(pos_list[i][1], fast.mean(), color=color_fast, label="Fast", width=0.5)
    plt.errorbar(pos_list[i][1], fast.mean(), yerr=fast.std(), c=color_fast)

    # Plot the individual points
    for j in range(len(slow)):
        plt.plot(pos_list[i][0], slow[j], marker='o', markersize=3, color=color_slow)
        plt.plot(pos_list[i][1], fast[j], marker='o', markersize=3, color=color_fast)
        # Add line connecting the points
        plt.plot([pos_list[i][0], pos_list[i][1]], [slow[j], fast[j]], color="black", linewidth=0.7, alpha=0.5)

    # Add statistics
    z, p = scipy.stats.ttest_rel(slow, fast)
    plt.text(pos_list[i][0], np.max(fast), f"p = {np.round(p, 3)}", fontsize=18)
    plt.yticks(fontsize=16)

# Adjust plot
plt.xticks(np.array(pos_list).mean(axis=-1), ["ECOG_contra", "ECOG_ipsi", "LFP_contra", "LFP_ipsi"])
plt.ylabel(f"average % change in power t={tmin}-{tmax}, f={fmin}-{fmax}")

plt.savefig(f"../../Figures/group_slow_fast_{tmin}-{tmax},_{fmin}-{fmax}.png", format="png", bbox_inches="tight",
            transparent=True)
plt.show()
