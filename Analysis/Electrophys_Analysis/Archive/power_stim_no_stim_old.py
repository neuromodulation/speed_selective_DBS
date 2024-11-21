# Results Figure 4: Compare power between stimulated and not stimulated movements

import os
import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import scipy
import sys
from scipy.stats import pearsonr, spearmanr, ttest_ind
path = "C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/vigor_stim_task/Code"
sys.path.insert(1, path)
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Set parameters
baseline = (-0.5, -0.2)
mode = "percent"
cmap = "bwr"
freq_min = 15
freq_max = 150
frequencies = np.arange(freq_min, freq_max, 4)
fontsize = 6

# Load the data
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_123_CAR.fif"
raw = mne.io.read_raw_fif(path).load_data()
# Add re-references raw channels
ecog_names = ["ECOG_R_1_CAR_raw", "ECOG_R_2_CAR_raw", "ECOG_R_3_CAR_raw", "ECOG_R_4_CAR_raw", "ECOG_R_5_CAR_raw"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT", "ECOG_R_04_SMC_AT", "ECOG_R_05_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_names = [f"ECOG_R_{x}_CAR_raw" for x in range(1,6)]

# Load index of similar movements
id = 6
similar_slow = np.load(f"../../../Data/Off/processed_data/Slow_similar.npy").astype(bool)
similar_fast = np.load(f"../../../Data/Off/processed_data/Fast_similar.npy").astype(bool)
slow_idx = np.where(np.hstack((similar_slow[id, 1, :, :].flatten(), similar_slow[6, 0, :, :].flatten())))[0]
fast_idx = np.where(np.hstack((similar_fast[id, 1, :, :].flatten(), similar_fast[6, 0, :, :].flatten())))[0]
fast_slow = [fast_idx, slow_idx]

# Filter out line noise
#raw.notch_filter(50)
#raw.filter(None, 100)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1.5, baseline=None, reject_by_annotation=True)

# Get fast and slow stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
slow = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 0
fast = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 1
fast_stim_idx = np.where(stim & fast)[0]
slow_stim_idx = np.where(stim & slow)[0]
stim_idx = np.where(stim)[0]
fast_slow_stim = [fast_stim_idx, slow_stim_idx]

# Compare the peak speed values
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 4))
labels = ["Fast", "Slow"]
for i in range(2):
    peak_speed_stim = epochs[fast_slow_stim[i]].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
    peak_speed_no_stim = epochs[fast_slow[i]].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
    # Plot as box plots
    ax[i].boxplot([peak_speed_stim, peak_speed_no_stim])
    # Add the p value from an independent t-test
    t, p = ttest_ind(peak_speed_stim, peak_speed_no_stim)
    ax[i].set_title(f"p = {p:.4f} {labels[i]}")

# Get average stimulation onset
onset_stim = [np.where(epochs[i].get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))].squeeze())[0][0] for i in stim_idx]
times = epochs.times[(epochs.times < 0.5) & (epochs.times > -0.2)]
onset_times = times[onset_stim]
mean_onset = np.median(onset_times)

# Compute tfr
tfr = epochs.compute_tfr(method="morlet", freqs=frequencies, picks=target_chan_names, average=False, n_cycles=5)#frequencies/2)

# Apply baseline
tfr = tfr.apply_baseline(baseline=baseline, mode=mode)

# Loop over frequency ranges of interest
band = [20, 33]
label = "$\\beta$ (13-35 Hz) \nNormalized power [%]"
tmin = mean_onset+0.5
tmax = mean_onset+0.6
#tmin = mean_onset+0.10
#tmax = mean_onset+0.20
colors_fill = np.array([["#EDB7B7", "white"], ["#EDB7B7", "white"]])
colors = np.array([["#3b0086", "#3b0086"],["#00863b", "#00863b"]])

for target_chan_name in target_chan_names:

    # One plot for each frequency band
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))

    # Calculate the average power in the frequency band
    band_power = tfr.get_data(picks=target_chan_name, tmin=tmin, tmax=tmax, fmin=band[0], fmax=band[1]).mean(axis=(1,2,3))

    # Loop over conditions
    for j in range(2):

        power_stim = band_power[fast_slow_stim[j]] * 100
        power_no_stim = band_power[fast_slow[j]] * 100
        power_all = [power_stim, power_no_stim]

        box_width = 0.25
        bar_pos = [j - (box_width / 1.5), j + (box_width / 1.5)]

        # Save for comparison
        if j == 1:
            slow_power = power_stim
            slow_pos = bar_pos[0]
        else:
            fast_power = power_stim
            fast_pos = bar_pos[0]

        for l in range(2):
            # Add the points
            jitter = np.random.uniform(-0.05, 0.05, size=len(power_all[l]))
            ax.scatter(np.repeat(bar_pos[l], len(power_all[l])) + jitter, power_all[l], s=0.5, c="dimgray",marker='o',
                       zorder=2)
            bp = ax.boxplot(x=power_all[l],
                            positions=[bar_pos[l]],
                            # label=labels[j, l],
                            widths=box_width,
                            patch_artist=True,
                            boxprops=dict(linestyle='--', fill=None, color=colors[j, l]),
                            capprops=dict(color=colors[j, l]),
                            whiskerprops=dict(color=colors[j, l]),
                            medianprops=dict(color="dimgray", linewidth=1),
                            flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=0,
                                            markeredgecolor='none')
                            )


        # Add statistics
        z, p = scipy.stats.ttest_ind(power_stim, power_no_stim)
        res = scipy.stats.permutation_test(data=(power_stim, power_no_stim),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=100000, permutation_type="independent")
        p = res.pvalue
        print(p)
        text = u.get_sig_text(p)
        ymin, ymax = ax.get_ylim()
        ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 0.1, ymax, ymax, ymax - 0.1], color="black",linewidth=0.8)
        ax.text(j, ymax + 0.1, text, fontsize=fontsize, horizontalalignment='center')

    # Add statistics between stimulated movements
    """z, p = scipy.stats.ttest_ind(slow_power, fast_power)
    print(p)
    text = u.get_sig_text(p)
    ymax = ymax + 10
    ax.plot([slow_pos, slow_pos, fast_pos, fast_pos], [ymax - 0.1, ymax, ymax, ymax - 0.1], color="black",linewidth=0.8)
    ax.text(j/2, ymax + 0.1, text, fontsize=fontsize, horizontalalignment='center')"""

    # Adjust plot
    ax.set_ylabel(label, fontsize=fontsize)
    ax.set_xticks(ticks=[0, 1], labels=["Fast", "Slow"], fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize-1)
    ax.set_title(f"{band[0]}-{band[1]} Hz {np.round(tmin, 2)}-{np.round(tmax, 2)} sec", fontsize=fontsize)
    u.despine()

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}.pdf", format="pdf", bbox_inches="tight",
                transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}.png", format="png", bbox_inches="tight",
                transparent=False)

plt.show()

# Make sure that the artifact does not contaminate the ITI
tmin_plot = -0.5
tmax_plot = 1
epochs_data = epochs.get_data(tmin=tmin_plot, tmax=tmax_plot, picks=target_chan_name).squeeze()
conds = ["Fast", "Slow"]
for i, idx_stim in enumerate(fast_slow_stim):
    for idx in idx_stim:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 4.8), height_ratios=[2, 1])
        power = tfr[idx].get_data(target_chan_name, tmin=tmin_plot, tmax=tmax_plot).squeeze()
        im = ax[0].imshow(power, aspect="auto", origin="lower", cmap=cmap, extent=(tmin_plot, tmax_plot, np.min(frequencies), np.max(frequencies)), vmin=-2, vmax=2)
        ax[0].axvline(tmin, 0, 1, linewidth=1, color="black")
        ax[0].axvline(tmax, 0, 1, linewidth=1, color="black")
        ax[1].plot(epochs_data[idx].flatten())
        plt.subplots_adjust(hspace=0.3, left=0.15)
        plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_epoch_stim_{conds[i]}_{idx}.png",
                format="png", bbox_inches="tight", transparent=False)
        plt.close()

