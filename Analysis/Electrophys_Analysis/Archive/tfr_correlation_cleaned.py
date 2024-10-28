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
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set the dataset
sub = "EL012"
med = "Off"
chan = "ECOG_bipolar"

# Set parameters
tmin = -1
tmax = 1
baseline = None
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
target_chan_name = "ECOG_R_2_CAR"

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.5, tmax=1.5, baseline=None, reject_by_annotation=True)


stim_idx = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.5))], axis=-1).squeeze()
slow_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 0
fast_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 1
fast_stim_idx = np.where(stim_idx & fast_block_idx)[0]
slow_stim_idx = np.where(stim_idx & slow_block_idx)[0]
epochs = epochs#[stim_idx]

# Extract the peak speed for each epoch
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute the power over time
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)

# Compute correlation for every time-frequency value (baseline corrected)
tfr.crop(tmin=tmin, tmax=tmax)

# Smooth the power
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)

# Decimate
tfr.decimate(decim=40)

tfr_data = tfr.data.squeeze()
n_epochs, n_freqs, n_times = tfr_data.shape
corr_p = np.zeros((2, n_freqs, n_times))
for i, freq in enumerate(frequencies):
    print(f"Freq: {freq}")
    for j, time in enumerate(tfr.times):
        corr, p = pearsonr(peak_speed, tfr_data[:, i, j])
        corr_p[0, i, j] = corr
        corr_p[1, i, j] = p

# Plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16.8, 6))

# Plot power
tfr_average = tfr.average()
tfr_average.plot(axes=axes[0], baseline=(None,None), mode="percent", show=False, colorbar=True, cmap=cmap)

# Add average speed
behav_mean = np.mean(epochs.get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
behav_std = np.std(epochs.get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
# Scale such that it fits in the plot
behav_mean_scaled = u.scale_min_max(behav_mean, freq_min+2, freq_max-2, behav_mean.min(axis=0), behav_mean.max(axis=0))
behav_std_scaled = u.scale_min_max(behav_std, freq_min+2, freq_max-2, behav_mean.min(axis=0), behav_mean.max(axis=0))
# Plot
times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
axes[0].plot(times, behav_mean_scaled, color="black", linewidth=2, alpha=0.7)
axes[0].fill_between(times, behav_mean_scaled - behav_std_scaled, behav_mean_scaled + behav_std_scaled, color="black", alpha=0.2)

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
sig = corr_p[1, :, :] > 0.05
corr_sig = corr_p[0, :, :]
corr_sig[sig] = 0
im = axes[1].imshow(corr_sig, aspect="auto", origin="lower", cmap=cmap, extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
# Add colorbar
cbar = fig.colorbar(im, ax=axes[1])
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Correlation', fontsize=16)

# Add average speed values
axes[1].plot(times, behav_mean_scaled, color="black", linewidth=2, alpha=0.7)
axes[1].fill_between(times, behav_mean_scaled - behav_std_scaled, behav_mean_scaled + behav_std_scaled, color="black", alpha=0.2)

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
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{chan}.svg",
        format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{chan}.png",
        format="png", bbox_inches="tight", transparent=False)
plt.show(block=True)
