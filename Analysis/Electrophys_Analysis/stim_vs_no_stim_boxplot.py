# Compare the average gamma (or beta) power during the stimulation

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import scipy
import sys
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
freq_min = 10
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)
fontsize = 7

# Load the data
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_123_CAR.fif"
#path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_name = "ECOG_R_2_CAR"

# Load index of similar movements
id = 6
similar_slow = np.load(f"../../../Data/Off/processed_data/Slow_similar.npy").astype(bool)
similar_fast = np.load(f"../../../Data/Off/processed_data/Fast_similar.npy").astype(bool)
fast_idx = np.where(np.hstack((similar_fast[id, 1, :, :].flatten(), similar_fast[6, 0, :, :].flatten())))[0]
slow_idx = np.where(np.hstack((similar_slow[id, 1, :, :].flatten(), similar_slow[6, 0, :, :].flatten())))[0]
slow_fast = [fast_idx, slow_idx]

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1, baseline=None, reject_by_annotation=True)

# Get fast and slow stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
slow = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 0
fast = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 1
fast_stim_idx = np.where(stim & fast)[0]
slow_stim_idx = np.where(stim & slow)[0]
stim_idx = np.where(stim)[0]
slow_fast_stim = [slow_stim_idx, fast_stim_idx]

# Get average stimulation onset
onset_stim = [np.where(epochs[i].get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))].squeeze())[0][0] for i in stim_idx]
times = epochs.times[(epochs.times < 0.5) & (epochs.times > -0.2)]
onset_times = times[onset_stim]
mean_onset = np.median(onset_times)

# Compute tfr
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)

# Apply baseline
tfr = tfr.apply_baseline(baseline=baseline, mode=mode)

# Loop over frequency ranges of interest
bands = [[13, 35], [60, 80]]
labels = ["Beta (13-35 Hz) Power", "Low Gamma (60-80 Hz) Power"]
tmin = mean_onset
tmax = mean_onset + 0.3
colors_fill = np.array([["#EDB7B7", "white"], ["#EDB7B7", "white"]])
colors = np.array([["#00863b", "#00863b"], ["#3b0086", "#3b0086"]])
for i, band in enumerate(bands):

    # One plot for each frequency band
    fig, ax = plt.subplots(1, 1, figsize=(0.8, 1.7))

    # Calculate the average power in the frequency band
    band_power = tfr.get_data(picks=target_chan_name, tmin=tmin, tmax=tmax, fmin=band[0], fmax=band[1]).mean(axis=(1,2,3))

    # Loop over conditions
    for j in range(2):

        power_stim = band_power[slow_fast_stim[j]]
        power_no_stim = band_power[slow_fast[j]]
        power_all = [power_stim, power_no_stim]

        box_width = 0.25
        bar_pos = [j - (box_width / 1.5), j + (box_width / 1.5)]

        # Save for comparison
        if j == 0:
            slow_power = power_stim
            slow_pos = bar_pos[0]
        else:
            fast_power = power_stim
            fast_pos = bar_pos[0]

        for l in range(2):
            bp = ax.boxplot(x=power_all[l],
                            positions=[bar_pos[l]],
                            # label=labels[j, l],
                            widths=box_width,
                            patch_artist=True,
                            boxprops=dict(linestyle='-.', facecolor=colors_fill[j, l], color=colors[j, l]),
                            capprops=dict(color=colors[j, l]),
                            whiskerprops=dict(color=colors[j, l]),
                            medianprops=dict(color="dimgray", linewidth=1),
                            flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=0,
                                            markeredgecolor='none')
                            )
            # Add the points
            jitter = np.random.uniform(-0.05, 0.05, size=len(power_all[l]))
            ax.scatter(np.repeat(bar_pos[l], len(power_all[l])) + jitter, power_all[l], s=0.5, c="dimgray",
                       zorder=2)

        # Add statistics
        z, p = scipy.stats.ttest_ind(power_stim, power_no_stim)
        """res = scipy.stats.permutation_test(data=(feature_av[:, 0], feature_av[:, 1]),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=100000, permutation_type="samples")
        p = res.pvalue"""
        text = u.get_sig_text(p)
        if i == 0:
            ymax = 0.8
        else:
            ymax = 2.5
        ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 0.1, ymax, ymax, ymax - 0.1], color="black",
                linewidth=0.8)
        ax.text(j, ymax + 0.1, text, fontsize=fontsize, horizontalalignment='center')

    # Add statistics between stimulated movements
    z, p = scipy.stats.ttest_ind(slow_power, fast_power)
    print(p)
    text = u.get_sig_text(p)
    ymax = ymax + 0.5
    ax.plot([slow_pos, slow_pos, fast_pos, fast_pos], [ymax - 0.1, ymax, ymax, ymax - 0.1], color="black",
            linewidth=0.8)
    ax.text(j/2, ymax + 0.1, text, fontsize=fontsize, horizontalalignment='center')

    # Adjust plot
    if i == 0:
        ax.set_ylim([-1, 1.5])
    ax.set_ylabel(labels[i], fontsize=fontsize)
    ax.set_xticks(ticks=[0, 1], labels=["Slow", "Fast"], fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize-2)
    u.despine()

    # Adjust plot
    #plt.subplots_adjust()

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{labels[i]}.pdf", format="pdf", bbox_inches="tight",
                transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{labels[i]}.png", format="png", bbox_inches="tight",
                transparent=False)

plt.show()

