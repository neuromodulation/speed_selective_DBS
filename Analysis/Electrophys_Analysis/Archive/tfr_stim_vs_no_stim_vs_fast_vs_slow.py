# Compare the tfr stimulation vs no stimulation vs fast vs slow

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr, ttest_ind
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')


# Set parameters
tmin = -1
tmax = 1
baseline = (None, None)
mode = "percent"
cmap = "jet"
freq_min = 15
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)

# Load the data
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_123_CAR.fif"
#path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
#target_chan_name = ch_names[-1]
target_chan_name = "ECOG_R_1_CAR"

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.5, tmax=1.5, baseline=None, reject_by_annotation=True)

# Get fast and slow stimulated movements
stim_idx = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.5))], axis=-1).squeeze()
slow_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 0
fast_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 1
fast_stim_idx = np.where(stim_idx & fast_block_idx)[0]
slow_stim_idx = np.where(stim_idx & slow_block_idx)[0]
not_stim_idx = np.where(~stim_idx)[0]

# Get the fast and slow NOT stimulated movements
idx_recovery = np.where([True if (x in [2, 4]) else False for x in epochs.get_data(["BLOCK"])[:,:, epochs.times == 0].squeeze().astype(int)])[0]
epochs_recovery = epochs.copy()[idx_recovery]
peak_speed = epochs_recovery.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
slow_idx = []
fast_idx = []
for i, ps in enumerate(peak_speed):
    if i > 1 and np.all(ps < peak_speed[i-2:i]):
        slow_idx.append(i)
    elif i > 1 and np.all(ps > peak_speed[i-2:i]):
        fast_idx.append(i)
idx_slow_sort = np.argsort(slow_idx)
idx_fast_sort = np.argsort(fast_idx)
slow_idx = idx_recovery[slow_idx[:int(len(slow_idx)/2)]]
fast_idx = idx_recovery[fast_idx[:int(len(fast_idx)/2)]]
not_stim_idx = np.hstack([fast_idx, slow_idx])

# Compare different pairs
comparisons = [[fast_stim_idx, slow_stim_idx], [fast_idx, slow_idx], [fast_stim_idx, fast_idx], [slow_stim_idx, slow_idx], [stim_idx, not_stim_idx]]
titles = ["Fast Stim vs Slow Stim", "Fast vs Slow", "Fast Stim vs Fast", "Slow Stim vs Slow", "Stim vs No Stim"]
colors = [["#3b0086", "#00863b"], ["#3b0086","#00863b"], ["grey", "black"], ["grey", "black"], ["grey", "black"]]
colors_op = [["#b099ce", "#b2dac4"], ["#b099ce", "#b2dac4"],  ["grey", "black"], ["grey", "black"], ["grey", "black"]]

# Compute tfr
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)
#tfr = tfr.apply_baseline(baseline=(None, None), mode="zscore")
# Crop tfr
tfr.crop(tmin=tmin, tmax=tmax)
# Smooth tfr
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)
# Decimate tfr
tfr.decimate(decim=40)
tfr_data = tfr.data.squeeze()

# Loop over comparisons and plot difference in tfr and significantly different regions
for i, comp in enumerate(comparisons):

    idx1 = comp[0]
    idx2 = comp[1]

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16.8, 6))

    # Plot difference in power
    tfr1 = tfr[idx1].average().apply_baseline(baseline=baseline, mode=mode)
    tfr2 = tfr[idx2].average().apply_baseline(baseline=baseline, mode=mode)
    tfr_diff = tfr1.copy()
    tfr_diff.data = tfr1.data - tfr2.data
    tfr_diff.plot(axes=axes[0], colorbar=True, cmap=cmap, show=False)
    # Add colorbar
    im = axes[0].images
    cbar = im[-1].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Difference in power in %', fontsize=14)

    # Compute difference between tfr of slow and fast movements
    n_epochs, n_freqs, n_times = tfr_data.shape
    res = np.zeros((2, n_freqs, n_times))
    for j, freq in enumerate(frequencies):
        print(f"Freq: {freq}")
        for k, time in enumerate(tfr.times):
            t, p = ttest_ind(a=tfr_data[idx1, j, k], b=tfr_data[idx2, j, k])
            res[0, j, k] = t
            res[1, j, k] = p

    # Plot statistics
    # Delete values that are not significant
    res[1, res[1, :, :] > 0.05] = 1
    im = axes[1].imshow(res[1, :, :], aspect="auto", origin="lower", cmap=cmap,
                        extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('P-value of pair-wise t-test (uncorrected)', fontsize=14)
    im.set_clim(0, 0.05)

    # Add average speed
    ax0_twin = axes[0].twinx()
    ax1_twin = axes[1].twinx()
    for j, idx in enumerate(comp):
        behav_mean = np.mean(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
        behav_std = np.std(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
        # Plot
        times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
        ax0_twin.plot(times, behav_mean, color=colors[i][j], linewidth=2, alpha=0.7)
        ax0_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=colors_op[i][j], alpha=0.5)
        ax1_twin.plot(times, behav_mean, color=colors[i][j], linewidth=2, alpha=0.7)
        ax1_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=colors_op[i][j], alpha=0.5)
        ax0_twin.get_yaxis().set_visible(False)
        ax1_twin.get_yaxis().set_visible(False)

    # Adjust plot
    for ax in axes[0:2]:
        ax.set_ylim(freq_min, freq_max - 1)
        ax.set_xlabel("Time(seconds)", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

    # Compare the peak speed values
    peak_speed1 = epochs[idx1].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
    peak_speed2 = epochs[idx2].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
    # Plot as box plots
    ax = axes[2]
    ax.boxplot([peak_speed1, peak_speed2])
    ax.set_ylabel("Peak speed", fontsize=14)
    ax.yaxis.set_tick_params(labelsize=12)
    # Add the p value from an independent t-test
    t, p = ttest_ind(peak_speed1, peak_speed2)
    ax.text(0.5, 0.9, f"p = {p:.4f}", fontsize=14, ha="center", va="center", transform=ax.transAxes)

    # Adjust figure
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(titles[i], fontsize=20)

    # Save figure
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{titles[i]}.svg",
                format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{titles[i]}.png",
                format="png", bbox_inches="tight", transparent=False)

    plt.show(block=True)
