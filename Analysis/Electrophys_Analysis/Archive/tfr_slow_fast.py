# Correlate peak speed and power values in the time frequency plot

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
import matplotlib
from scipy.ndimage import uniform_filter1d
from scipy import stats
from sklearn.preprocessing import minmax_scale
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

matplotlib.use('Qt5Agg')

# Set the dataset
sub = "EL012"
med = "Off"

# Set parameters
tmin = -1
tmax = 1
baseline = None
cmap = "jet"
freq_min = 15
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)

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
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]

if sub == "EL008":
    target_chan_name = "ECOG_L_05_SMC_AT"
    target_chan_name = "ECOG_L_06_SMC_AT"
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"
    target_chan_name2 = "ECOG_R_03_SMC_AT"

# Rereference to ecog channel
raw._data[ch_names.index(target_chan_name), :] = raw.get_data(target_chan_name) - raw.get_data(target_chan_name2)

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.5, tmax=1.5, baseline=None, reject_by_annotation=True)

# Get epochs during the recovery block
idx_recovery = [True if (x in [1, 3]) else False for x in epochs.get_data(["BLOCK"])[:,:, epochs.times == 0].squeeze()]
epochs = epochs[idx_recovery]

# Extract the peak speed for each epoch
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Get the fast and slow epochs during the recovery block
slow_idx = []
fast_idx = []
for i, ps in enumerate(peak_speed):
    if i > 1 and np.all(ps < peak_speed[i-2:i]):
        slow_idx.append(i)
    elif i > 1 and np.all(ps > peak_speed[i-2:i]):
        fast_idx.append(i)

# Compute the power over time
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)

# Crop tfr
tfr.crop(tmin=tmin, tmax=tmax)

# Smooth tfr
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)

# Decimate tfr
tfr.decimate(decim=40)
tfr_data = tfr.data.squeeze()

# Compute difference between tfr of slow and fast movements
n_epochs, n_freqs, n_times = tfr_data.shape
res = np.zeros((2, n_freqs, n_times))
for i, freq in enumerate(frequencies):
    print(f"Freq: {freq}")
    for j, time in enumerate(tfr.times):
        t, p = stats.ttest_ind(a=tfr_data[slow_idx, i, j], b=tfr_data[fast_idx, i, j])
        res[0, i, j] = t
        res[1, i, j] = p

# Plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16.8, 6))

# Plot difference in power
tfr_slow = tfr[slow_idx].average().apply_baseline(baseline=(None,None), mode="percent")
tfr_fast = tfr[fast_idx].average().apply_baseline(baseline=(None,None), mode="percent")
tfr_diff = tfr_slow.copy()
tfr_diff.data = tfr_fast.data - tfr_slow.data
tfr_diff.plot(axes=axes[0], colorbar=True, cmap=cmap, show=False)

# Add average speed
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
ax0_twin = axes[0].twinx()
ax1_twin = axes[1].twinx()
for i, idx in enumerate([slow_idx, fast_idx]):
    behav_mean = np.mean(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    behav_std = np.std(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    # Plot
    times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
    ax0_twin.plot(times, behav_mean, color=colors[i], linewidth=2, alpha=0.7)
    ax0_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=colors_op[i], alpha=0.5)
    ax1_twin.plot(times, behav_mean, color=colors[i], linewidth=2, alpha=0.7)
    ax1_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=colors_op[i], alpha=0.5)

# Adjust plot
axes[0].set_xlabel("Time(seconds)", fontsize=20)
axes[0].set_ylim(freq_min, freq_max-1)
axes[0].yaxis.get_label().set_fontsize(20)
axes[0].yaxis.set_tick_params(labelsize=16)
axes[0].xaxis.set_tick_params(labelsize=16)
im = axes[0].images
cbar = im[-1].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Power', fontsize=16)

# Plot R values
# Delete correlation values that are not significant
sig = res[1, :, :] > 0.05
corr_sig = res[1, :, :]
corr_sig[sig] = 0
im = axes[1].imshow(corr_sig, aspect="auto", origin="lower", cmap=cmap, extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
# Add colorbar
cbar = fig.colorbar(im, ax=axes[1])
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Correlation', fontsize=16)

# Adjust plot
axes[1].set_yticks([])
axes[1].set_ylim(freq_min, freq_max-1)
axes[1].set_xlabel("Time(seconds)", fontsize=20)
axes[1].xaxis.set_tick_params(labelsize=16)

# Adjust figure
plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0.32)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}.svg",
        format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}.png",
        format="png", bbox_inches="tight", transparent=True)
plt.show(block=True)


plt.show(block=True)