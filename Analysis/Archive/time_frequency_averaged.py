# Compute the time frequency plot for every subject and recording location
# Average over subjects

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
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Specify the medication group
med = "Off"

# Specify if individual or only group plots are needed
plot_individual = False

# Specify the root folder
root_folder = 'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/Tablet_task/'

# Set parameters for analysis
tmin = -0.5
tmax = 0.5
baseline = (-0.75, -0.5)
mode = "percent"
cmap = "jet"
freq_min = 5
freq_max = 120
frequencies = np.arange(freq_min, freq_max, 2)
target_names = ["STN Contralateral", "STN Ipsilateral", "ECOG Contralateral", "ECOG Ipsilateral"]

# Read the list of the datasets
df = pd.read_excel(f'{root_folder}Data/Dataset_list.xlsx', sheet_name=med)

# Loop through the subjects
subject_list = list(df["ID Berlin_Neurophys"][1:21])
subject_list.remove("L003")  # NO neurophys data available

power_all_sub = []
behav_all_sub = []
for sub in subject_list:

    # Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
           f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
           f'rawdata/'

    bids_paths = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                        extensions=".vhdr",
                                        subjects=sub,
                                        sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                                  f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])
    raw = read_raw_bids(bids_path=bids_paths[0])
    raw.load_data()
    sfreq = raw.info["sfreq"]

    # Drop bad channels
    raw.drop_channels(raw.info["bads"])
    ch_names = raw.info["ch_names"]

    # Filter out line noise
    raw.notch_filter(50)

    # Add average LFP channels
    for loc in ["LFP_L", "LFP_R"]:
        target_chs = [ch for ch in ch_names if (loc in ch) and (not "01" in ch) and (not "08" in ch)]
        target_ch = f"av_{loc}"
        new_ch = raw.get_data(target_chs).mean(axis=0)
        u.add_new_channel(raw, new_ch[np.newaxis, :], target_ch, type="dbs")
    # Select the ecog target channel
    ECOG_target = df.loc[df["ID Berlin_Neurophys"] == sub]["ECOG_target"].iloc[0]
    if "E" in sub:
        if "R" in ECOG_target:
            target_ECOG_R = ECOG_target
            target_ECOG_L = ""
        elif "L" in ECOG_target:
            target_ECOG_L = ECOG_target
            target_ECOG_R = ""
    else:
        target_ECOG_R = ""
        target_ECOG_L = ""

    # Extract events
    events = mne.events_from_annotations(raw)[0]

    # Annotate periods with stimulation
    sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
    n_stim = len(sample_stim)
    onset = (sample_stim / sfreq) - 0.1
    duration = np.repeat(0.5, n_stim)
    stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
    raw.set_annotations(stim_annot)

    # Compute time-frequency spectrum aligned to peak speed
    event_id = 10003

    # Cut into epochs
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2, baseline=None, reject_by_annotation=True)
    epochs.drop_bad()

    # Compute the tfr for LFP ipsi/contralateral and eocg ipsi/contralateral
    if sub == "EL012" or sub == "L013":  # Left
        target_chs = ["av_LFP_R", "av_LFP_L", target_ECOG_R, target_ECOG_L]
    else:  # right
        target_chs = ["av_LFP_L", "av_LFP_R", target_ECOG_L, target_ECOG_R]

    if plot_individual:
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 6))

    # Get speed curve for plotting
    behav_mean = np.mean(epochs.get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    behav_std = np.std(epochs.get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    # Scale such that it fits in the plot
    behav_mean_scaled = u.scale_min_max(behav_mean, freq_min + 2, freq_max - 2, behav_mean.min(axis=0),
                                        behav_mean.max(axis=0))
    behav_std_scaled = u.scale_min_max(behav_std, freq_min + 2, freq_max - 2, behav_mean.min(axis=0),
                                       behav_mean.max(axis=0))
    times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
    # Save behavior (mean speed)
    behav_all_sub.append(behav_mean_scaled)

    power_sub = []
    for i, target_ch in enumerate(target_chs):
         if target_ch:
            # Compute the tfr for the target channel
            power = mne.time_frequency.tfr_morlet(epochs, n_cycles=7,  picks=[raw.info["ch_names"].index(target_ch)],
                                              return_itc=False, freqs=frequencies, average=True, verbose=3, use_fft=True)
            # Apply baseline correction using the defined method
            power.apply_baseline(baseline=baseline, mode=mode)
            # Crop in window of interest
            power.crop(tmin=tmin, tmax=tmax)
            # Smooth the tfr
            power.data = uniform_filter1d(power.data, size=int(power.info["sfreq"]/1000*250), axis=-1)

            # Plot individual if needed
            if plot_individual:
                power.plot(baseline=None, axes=axes[i], vmin=-4, vmax=4, tmin=tmin, tmax=tmax, mode=mode, show=False, colorbar=True, cmap=cmap)
                axes[i].plot(times, behav_mean_scaled, color="black", linewidth=2, alpha=0.7)
                axes[i].fill_between(times, behav_mean_scaled - behav_std_scaled, behav_mean_scaled + behav_std_scaled, color="black", alpha=0.2)
                axes[i].set_title(target_names[i])
            # Save power
            power_sub.append(power.data)
            power_dim = power.data.shape

         else:
            power_sub.append(np.nan * np.ones(shape=power_dim))

    # Save individual plot
    if plot_individual:
        #plt.show()
        #plt.savefig(f"../../Figures/{sub}_tfr.svg", format="svg", bbox_inches="tight", transparent=True)
        plt.savefig(f"../../Figures/{sub}_tfr.png", format="png", bbox_inches="tight", transparent=True)
        plt.close()

    # Save power for all subjects
    power_all_sub.append(np.array(power_sub).squeeze())

# tmp, get the index of ipsi/contralateral ecog
idx_contra = [subject_list.index("EL012"), subject_list.index("EL008")]
idx_ipsi = [i for i, sub in enumerate(subject_list) if (i not in idx_contra) and ("E" in sub)]
idx = [np.arange(len(subject_list)), np.arange(len(subject_list)), idx_contra, idx_ipsi]

# Average power spectrum over subjects and plot
power_all_sub = np.array(power_all_sub)
power_mean = np.nanmedian(power_all_sub, axis=0)
behav_all_sub = np.array(behav_all_sub)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(5, 15))
for i, target in enumerate(target_names):
    power.data = power_mean[i, :, :][np.newaxis, :, :]
    power.plot(baseline=None, axes=axes[i], vmin=-4, vmax=2, tmin=tmin, tmax=tmax, mode=mode, show=False, colorbar=True,
               cmap=cmap)
    # Add averaged behavior
    behav_mean = behav_all_sub[idx[i]].mean(axis=0)
    behav_std = np.array(behav_all_sub[idx[i]]).std(axis=0)
    axes[i].plot(times, behav_mean, color="black", linewidth=2, alpha=0.7)
    axes[i].fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color="black",
                         alpha=0.2)
    if i < 3:
        axes[i].set_xticks([])
        axes[i].set_xlabel("")
    axes[i].yaxis.get_label().set_fontsize(20)
    axes[i].xaxis.get_label().set_fontsize(20)
    axes[i].yaxis.set_tick_params(labelsize=16)
    axes[i].xaxis.set_tick_params(labelsize=16)
    im = axes[i].images
    cbar = im[-1].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Power (z-score)', fontsize=18)

    #axes[i].set_title(target_names[i])
plt.subplots_adjust(left=0.2, hspace=0.15)

# Save
plt.savefig(f"../../Figures/group_tfr.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/group_tfr.png", format="png", bbox_inches="tight", transparent=True)
plt.show()
plt.close()