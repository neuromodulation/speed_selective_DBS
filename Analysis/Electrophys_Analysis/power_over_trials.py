# Plot the power of beta and gamma over time
# start of movement-peak of movement (without stimulation)

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import scipy
import matplotlib
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set the dataset
sub = "EL012"
med = "Off"

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects=sub,
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                              f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])

# Load dataset
raw = read_raw_bids(bids_path=bids_path[0])
raw.load_data()
raw.drop_channels(raw.info["bads"])
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]

if sub == "EL008":
    target_chan_name = "ECOG_L_05_SMC_AT"
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"

# Filter out line noise
#raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Annotate periods with stimulation
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.01
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
raw.set_annotations(stim_annot)

# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-0.3, tmax=0, baseline=None, reject_by_annotation=True)
epochs_samples = epochs.selection.copy()
epochs.drop_bad()
drop_sample = np.where(list(map(lambda x: x == ('bad stim',), epochs.drop_log)))[0]
drop_idx = np.where(epochs_samples == drop_sample)[0]

# Plot the power spectral density
#epochs.plot_psd(picks=[target_chan_name], fmax=200)

# Extract the peak speed for each epoch
#speed = epochs.get_data(["SPEED_MEAN"]).mean(axis=0).flatten()

# Compute the power in different frequency bands
#θ(4-8 Hz), α(8–12Hz), β(13–35Hz), low β(13–20Hz), high β(20–35Hz), all γ(60–200Hz), low γ(60–80Hz) (90–20Hz)
# Keep only target channel
epochs.load_data()
epochs = epochs.pick([target_chan_name, 'STIM_CONDITION'])
bands = [[4, 8], [8, 12], [13, 35], [13, 20], [20, 35], [60, 200], [60, 80], [90, 200]]
bands = [[15, 40], [60, 100]]
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]
box_width = 0.3
for i, band in enumerate(bands):

    # Compute power in frequency band
    psds, freqs = epochs.compute_psd(fmin=band[0], fmax=band[-1], method='multitaper').get_data(return_freqs=True)
    power = np.mean(psds, axis=-1).flatten()

    # Plot (2 subplots, one with the feature over time and one with the boxplot)
    f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(12.5, 3.5))

    power_perc_stim = []
    power_perc_recov = []

    # Calculate change in power from start
    for cond in range(2):

        # Get all trials for one condition
        cond_id = np.unique(epochs.get_data(["STIM_CONDITION"]), axis=-1).flatten()
        stim_idx = np.where(cond_id == cond)[0]
        power_tmp = power[stim_idx]

        # Remove outliers
        np.apply_along_axis(lambda m: u.fill_outliers_nan_ephys(m), axis=0, arr=power_tmp)

        # Cut away first 5
        power_tmp = power_tmp[5:]

        # Normalize in % to next 5 trials
        power_perc = ((power_tmp - power_tmp[:5].mean()) / power_tmp[:5].mean()) * 100
        #power_perc = power_tmp

        # Calculate average change for stimulation and recovery block
        drop_stim = np.sum(drop_idx < 96 or (drop_idx < 96*3 and drop_idx > 96*2))
        power_perc_stim.append(power_perc[5:96-drop_stim])
        drop_recov = np.sum(drop_idx > 96*3 or (drop_idx < 96*2 and drop_idx > 96))
        power_perc_recov.append(power_perc[-96-drop_recov:])

        # Plot the feature over time (compute mean over patients)
        x = np.arange(power_perc.shape[-1])
        ax1.plot(x, power_perc, label=labels[cond], color=colors[cond], linewidth=3, alpha=0.4)

    # Add line at y=0 and x=96
    #ax1.axhline(0, linewidth=1, color="black", linestyle="dashed")
    ax1.axvline(96, linewidth=1, color="black", linestyle="dashed")
    # Adjust plot
    ax1.set_ylabel(f"Power change in %", fontsize=15)
    ax1.set_xlabel("Movement number", fontsize=15)
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.spines[['right', 'top']].set_visible(False)
    y_limits = ax1.get_ylim()
    ax1.text(25, y_limits[1], "Stimulation", rotation=0, fontsize=14)
    ax1.text(118, y_limits[1], "Recovery", rotation=0, fontsize=14)

    power_perc_all = [power_perc_stim, power_perc_recov]

    # Plot box plot with statistics
    for block in range(2):

        bar_pos = [block - (box_width / 1.5), block + (box_width / 1.5)]

        # Loop over conditions
        bps = []
        for cond in range(2):
            x = power_perc_all[block][cond]
            # remove nans
            x = x[~np.isnan(x)]
            bp = ax2.boxplot(x=x,
                             positions=[bar_pos[cond]],
                             widths=box_width,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors_op[cond], color=colors_op[cond]),
                             capprops=dict(color=colors_op[cond]),
                             whiskerprops=dict(color=colors_op[cond]),
                             medianprops=dict(color=colors[cond], linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor=colors_op[cond], markersize=5,
                                             markeredgecolor='none')
                             )
            bps.append(bp)  # Save boxplot for creating the legen

        # Add statistics
        x1 = power_perc_all[block][0]
        x1 = x1[~np.isnan(x1)]
        x2 = power_perc_all[block][1]
        x2 = x2[~np.isnan(x2)]
        z, p = scipy.stats.ttest_ind(x1, x2)
        """res = scipy.stats.permutation_test(data=(feature_av[:, 0], feature_av[:, 1]),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=100000, permutation_type="samples")
        p = res.pvalue"""
        sig = "bold" if p < 0.05 else "regular"
        ax2.text(block - box_width, y_limits[-1], f"p = {np.round(p, 3)}", weight=sig,
                 fontsize=14)
        # ax2.text(block-box_width, 63, f"p = {np.round(p, 3)}", weight=sig, fontsize=13)

        # Add legend
        ax2.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['Slow', 'Fast'],
                   loc='lower center', bbox_to_anchor=(-0.1, 0.6),
                   prop={'size': 13})

    # Adjust subplot
    #ax2.axhline(0, linewidth=1, color="black", linestyle="dashed")
    ax2.set_xticks(ticks=[0, 1], labels=["Stimulation", "Recovery"], fontsize=14)
    ax2.set_yticks([])
    ax2.spines[['left']].set_visible(False)
    ax2.set_ylim([y_limits[0], y_limits[1]])
    u.despine()

    # Adjust plot
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.8, wspace=0.01)
    plt.suptitle(f"{band[0]}-{band[-1]} Hz")

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{band[0]}_{band[-1]}.svg",
                format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{band[0]}_{band[-1]}.png",
                format="png", bbox_inches="tight", transparent=True)

plt.show()
