# Cut the data into epochs before the peak speed (without stimulation)

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
sub = "EL008"
med = "Off"

# Set parameters
tmin = -0.3
tmax = 0
baseline = (-0.75, -0.5)
#baseline = (-1.5, 1.5)
mode = "zscore"
cmap = "jet"
fmin = 3
fmax = 120
frequencies = np.arange(fmin, fmax, 2)

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

# Filter out line noise
#raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# How much later does the stimulation start in relation to the peak speed?
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
sample_peak = events[np.where(events[:, 2] == 10003)[0], 0]
diffs = np.zeros((len(sample_stim)))
stim_before_peak = np.zeros((len(sample_peak)))
for i in range(len(sample_stim)):
    tmp_diff = sample_stim[i] - sample_peak
    idx_min = np.argmin(np.abs(tmp_diff))
    diffs[i] = tmp_diff[idx_min]
    stim_before_peak[idx_min] = True if diffs[i] < 0 else False
#plt.hist(diffs)

# Annotate the epochs where stimulation comes before the peak
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.1
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
raw.set_annotations(stim_annot)

# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2, baseline=None, reject_by_annotation=False)
#epochs.drop_bad()
#epochs.load_data()
#epochs.crop(tmin=tmin, tmax=tmax)

# Remove bad epochs (stimulation during movement or baseline period)
epochs.drop(np.where(stim_before_peak)[0])

# Inspect power spectrum of epochs
#epochs.plot_psd(fmin=fmin, fmax=fmax)

# Extract the peak speed for each epoch
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute the power over time
channel = "ECOG_L_05_SMC_AT"
power = mne.time_frequency.tfr_morlet(epochs, n_cycles=7,  picks=[raw.info["ch_names"].index(channel)],
                                      return_itc=False, freqs=frequencies, average=False, use_fft=True)
power.data = uniform_filter1d(power.data, size=int(power.info["sfreq"]/1000*250), axis=-1)
power_raw = power.copy()

# Compute correlation for every time-frequency value (baseline corrected)
#power.apply_baseline(baseline=baseline, mode=mode)
power.crop(tmin=tmin, tmax=tmax)
power_data = power.data.squeeze()
n_epochs, n_freqs, n_times = power_data.shape
corr_p = np.zeros((2, n_freqs, n_times))
for i, freq in enumerate(frequencies):
    for j, time in enumerate(power.times):
        corr, p = spearmanr(peak_speed, power_data[:, i, j])
        corr_p[0, i, j] = corr
        corr_p[1, i, j] = p

# Plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16.8, 6))

# Plot power
power = power_raw.average(method="median")
power.plot([channel], baseline=baseline, axes=axes[0], tmin=tmin, tmax=tmax, mode=mode, show=False, colorbar=True, cmap=cmap)

# Add average speed
behav_mean = np.mean(epochs.get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
behav_std = np.std(epochs.get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
# Scale such that it fits in the plot
behav_mean_scaled = u.scale_min_max(behav_mean, fmin+2, fmax-2, behav_mean.min(axis=0), behav_mean.max(axis=0))
behav_std_scaled = u.scale_min_max(behav_std, fmin+2, fmax-2, behav_mean.min(axis=0), behav_mean.max(axis=0))
# Plot
times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
axes[0].plot(times, behav_mean_scaled, color="black", linewidth=2, alpha=0.7)
axes[0].fill_between(times, behav_mean_scaled - behav_std_scaled, behav_mean_scaled + behav_std_scaled, color="black", alpha=0.2)

# Adjust plot
axes[0].set_xlabel("Time(seconds)", fontsize=20)
axes[0].set_ylim(fmin, fmax-1)
axes[0].yaxis.get_label().set_fontsize(20)
axes[0].yaxis.set_tick_params(labelsize=16)
axes[0].xaxis.set_tick_params(labelsize=16)
im = axes[0].images
cbar = im[-1].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Power (z-score)', fontsize=16)

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
axes[1].set_ylim(fmin, fmax-1)
axes[1].set_xlabel("Time(seconds)", fontsize=20)
axes[1].xaxis.set_tick_params(labelsize=16)

# Adjust figure
plt.subplots_adjust(left=0.2, bottom=0.2, hspace=0.32)

# Save figure
plt.savefig(f"../../Figures/{sub}_tfr_speed_corr_poster_{channel}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/{sub}_tfr_speed_corr_poster_{channel}.png", format="png", bbox_inches="tight", transparent=True)

plt.show(block=True)