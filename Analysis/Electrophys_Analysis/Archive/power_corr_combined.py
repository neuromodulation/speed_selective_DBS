# Compute the correaltion between the power and the feature over trials
# Power computed on the -400-100 ms before peak speed (without stimulation)

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
import seaborn as sb
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set the parameters
sub = "EL008"
med = "Off"
feature_name = "mean_speed"
n_norm = 5
n_cutoff = 5

# Load behavioral data
# Load matrix containing the feature values
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape
# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)
# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))
if sub == "EL008":
    feature = feature[3, :, :]
else:
    feature = feature[6, :, :]

# Load the electrophysiological data
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
sfreq = raw.info["sfreq"]


if sub == "EL008":
    target_chan_name = "ECOG_L_06_SMC_AT"
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"

# Filter out line noise
raw.notch_filter(50)

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
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-0.3, tmax=0, baseline=None, reject_by_annotation=False)
epochs_samples = list(epochs.selection.copy())
epochs.drop_bad()
drop_sample = np.where(list(map(lambda x: x == ('bad stim',), epochs.drop_log)))[0]
drop_idx = np.where(epochs_samples == drop_sample)[0]
drop_idx = np.array([epochs_samples.index(samp) for samp in drop_sample])

# Compute the correlation between power and feature in different frequency bands
#θ(4-8 Hz), α(8–12Hz), β(13–35Hz), low β(13–20Hz), high β(20–35Hz), all γ(60–200Hz), low γ(60–80Hz) (90–20Hz)
# Keep only target channel
epochs.load_data()
epochs = epochs.pick([target_chan_name, 'STIM_CONDITION'])
#bands = [[4, 8], [8, 12], [13, 35], [13, 20], [20, 35], [60, 200], [60, 80], [90, 200]]
#bands = [[20, 40], [13, 35], [15, 40]]
bands = [[15, 40], [60, 100]]
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

for i, band in enumerate(bands):

    # Plot (2 subplots, one with the feature over time and one with the boxplot)
    f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(12.5, 3.5))

    # Compute power in frequency band
    psds, freqs = epochs.compute_psd(fmin=band[0], fmax=band[-1], method='multitaper').get_data(return_freqs=True)
    power = np.mean(psds, axis=-1).flatten()

    # Power over trials plot
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
        # power_perc = power_tmp
        # Calculate average change for stimulation and recovery block
        drop_stim = np.sum(drop_idx < 96 or (drop_idx < 96 * 3 and drop_idx > 96 * 2))
        power_perc_stim.append(power_perc[5:96 - drop_stim])
        drop_recov = np.sum(drop_idx > 96 * 3 or (drop_idx < 96 * 2 and drop_idx > 96))
        power_perc_recov.append(power_perc[-96 - drop_recov:])
        # Plot the feature over time (compute mean over patients)
        x = np.arange(power_perc.shape[-1])
        ax1.plot(x, power_perc, label=labels[cond], color=colors[cond], linewidth=3, alpha=0.4)

    # Add line at y=0 and x=96
    # ax1.axhline(0, linewidth=1, color="black", linestyle="dashed")
    ax1.axvline(96, linewidth=1, color="black", linestyle="dashed")
    ax1.spines[['right', 'top']].set_visible(False)
    y_limits = plt.ylim()

    power_perc_all = [power_perc_stim, power_perc_recov]

    # Plot box plot with statistics
    for block in range(2):

        # Add statistics
        x1 = power_perc_all[block][0]
        x1 = x1[~np.isnan(x1)]
        x2 = power_perc_all[block][1]
        x2 = x2[~np.isnan(x2)]
        z, p = scipy.stats.ttest_ind(x1, x2)
        sig = "bold" if p < 0.05 else "regular"
        if block == 0:
            ax1.text(25, y_limits[1], f"{labels[block]}: {np.round(p,3)}", fontsize=15)
        else:
            ax1.text(110, y_limits[1], f"{labels[block]}:{np.round(p,3)}", fontsize=15)

    # Correlation plot
    feature_all = []
    power_all = []
    # Calculate change in power from start
    for cond in range(2):

        # Get all trials for one condition
        cond_id = np.unique(epochs.get_data(["STIM_CONDITION"]), axis=-1).flatten()
        cond_idx = np.where(cond_id == cond)[0]
        power_tmp = power[cond_idx]

        # Remove outliers
        np.apply_along_axis(lambda m: u.fill_outliers_nan_ephys(m), axis=0, arr=power_tmp)
        # Cut away first 5
        power_tmp = power_tmp[n_cutoff:]
        # Normalize in % to next 5 trials
        power_perc = ((power_tmp - power_tmp[:n_norm].mean()) / power_tmp[:n_norm].mean()) * 100
        #power_perc = power_tmp

        # Behavior: Remove the same trials, cutoff and normalize
        # Remove trials that were dropped (because of stimulation artifact)
        feature_cond = feature[cond, :]
        if np.max(cond_idx) > 96*3:
            if len(drop_idx[drop_idx > 96*2]) > 0:
                drop_idx_new = np.array([idx - 96*2 for idx in drop_idx if idx > 96*2])
                feature_cond = np.delete(feature_cond, drop_idx_new)
        else:
            if len(drop_idx[drop_idx < 96*2]) > 0:
                drop_idx_new = np.array([idx for idx in drop_idx if idx < 96*2])
                feature_cond = np.delete(feature_cond, drop_idx_new)
        feature_cond = feature_cond[n_cutoff:]
        feature_cond = u.norm_perc(feature_cond, n_norm=n_norm)
        feature_all.append(feature_cond)
        power_all.append(power_perc)

    # Calculate correlation
    feature_all = np.array(feature_all).flatten()
    power_all = np.array(power_all).flatten()
    not_nan = ~np.isnan(feature_all) & ~np.isnan(power_all) #& (np.abs(feature_all) < 150) & (np.abs(power_all) < 150)
    try:
        corr, p = pearsonr(power_all[not_nan], feature_all[not_nan])
        # corr, p = u.permutation_correlation(x, y, n_perm=100000, method='pearson')
        p = np.round(p, 3)
        if p < 0.05:
            label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
        else:
            label = f" R = {np.round(corr, 2)} p = {p}"
        sb.regplot(x=power_all[not_nan], y=feature_all[not_nan], label=label, ax=ax2, scatter_kws={"color": "dimgrey", "alpha": 0.2}, line_kws={"color": "indianred"})

        # Adjust plot
        ax2.legend(loc="upper right", fontsize=11)
        u.despine()
    except:
        print(f"{band[0]}-{band[-1]} only nan")

    # Adjust plot
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.9, wspace=0.1, hspace=0.3)
    plt.title(f"{band[0]}-{band[-1]} Hz Power", fontsize=13)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{band[0]}-{band[-1]}_{feature_name}.svg",
                format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{band[0]}-{band[-1]}_{feature_name}.png",
                format="png", bbox_inches="tight", transparent=True)

plt.show()
