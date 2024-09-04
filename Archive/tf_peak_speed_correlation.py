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
sfreq = raw.info["sfreq"]

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Annotate periods with stimulation
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.1
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
raw.set_annotations(stim_annot)


# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]
tmin = -0.7
tmax = 0.7

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.3, tmax=1.3, baseline=None, reject_by_annotation=True)
epochs.drop_bad()

# Extract the peak speed for each epoch
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute the power over time
channel = "ECOG_R_01_SMC_AT"
frequencies = np.arange(5, 150, 3)
power = mne.time_frequency.tfr_morlet(epochs, n_cycles=7,  picks=[raw.info["ch_names"].index(channel)],
                                      return_itc=False, freqs=frequencies, average=False, verbose=3, use_fft=True)
power_raw = power.copy()

# Compute correlation for every time-frequency value (baseline corrected)
power.crop(tmin=tmin, tmax=tmax)
power.apply_baseline(baseline=(tmin, tmax))
power_data = power.data.squeeze()
n_epochs, n_freqs, n_times = power_data.shape
corr_p = np.zeros((2, n_freqs, n_times))
for i, freq in enumerate(frequencies):
    for j, time in enumerate(power.times):
        corr, p = spearmanr(peak_speed, power_data[:, i, j])
        corr_p[0, i, j] = corr
        corr_p[1, i, j] = p

# Plot
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(8, 15))

# Plot power
power = power_raw.average()
power.plot([channel], baseline=(tmin, tmax), axes=axes[0], tmin=tmin, tmax=tmax, mode="zscore", show=False, colorbar=False, cmap="jet")
axes[0].set_xticks([])
axes[0].set_xlabel("")
axes[0].colorbar()

# Plot average speed
behav = np.mean(epochs.get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
axes[1].plot(times, behav, color="black")
time_ticks = axes[1].get_xticklabels()
time_ticks_pos = axes[1].get_xticks()
axes[1].set_xticks([])

# Plot R values
axes[2].imshow(corr_p[0, :, :], aspect="auto", vmin=-0.25, vmax=0.25, cmap='jet', origin="lower", extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
axes[2].set_xticks([])
axes[2].set_ylabel("R-value")
axes[2].set_yticklabels(freqs_ticks)


# Plot p values
axes[3].imshow(corr_p[1, :, :], aspect="auto", vmin=0, vmax=0.05, cmap='jet', origin="lower", extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
axes[1].colorbar()
axes[3].set_ylabel("P-value")
axes[3].set_xlabel("Time[Sec]")

# Save figure
plt.savefig(f"../../Figures/{sub}_tfr_speed_corr_{channel}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/{sub}_tfr_speed_corr_{channel}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()