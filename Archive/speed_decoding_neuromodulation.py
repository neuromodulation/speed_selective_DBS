# Explore py_neuromodulation

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
import py_neuromodulation as pn
import seaborn as sb
from py_neuromodulation import (
    nm_analysis,
    nm_define_nmchannels,
    nm_plots
)
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
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

# Crop
raw.crop(tmin=560)

# Set channels
ch_names = ["ECOG_R_01_SMC_AT", "SPEED_MEAN"]
ch_idx =[raw.info["ch_names"].index(channel) for channel in ch_names]
ch_types = ["ECOG", "BEH"]
nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names, ch_types=ch_types, reference=None, target_keywords="SPEED")

# Settings
settings = pn.nm_settings.get_default_settings()
settings = pn.nm_settings.reset_settings(settings)
settings["features"]["fft"] = True
settings["features"]["bursts"] = True
settings["features"]["sharpwave_analysis"] = True

# Compute features
stream = pn.Stream(
    settings=settings,
    nm_channels=nm_channels,
    verbose=True,
    sfreq=sfreq,
    line_noise=50
)
features = stream.run(raw.get_data(picks=ch_names))

# Analyze features
analyzer = nm_analysis.Feature_Reader(
    feature_dir=stream.PATH_OUT,
    feature_file=stream.PATH_OUT_folder_name
)

# Compute correlation
feature_names = list(features)[:7]
for i in range(len(feature_names)):
    plt.figure()
    plt.title(feature_names[i])
    x = features[feature_names[i]]
    y = features["SPEED_MEAN"]
    corr, p = spearmanr(x, y, nan_policy='omit')
    p = np.round(p, 3)
    sb.regplot(x=x, y=y, label=f"r = {np.round(corr, 2)} " + f"p={p}")
    plt.legend()
    plt.show()

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
tmin = -0.9
tmax = 0.9

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.5, tmax=1.5, baseline=None, reject_by_annotation=True)
epochs.drop_bad()

# Extract the peak speed for each
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute the power over time
channel = "ECOG_R_01_SMC_AT"
frequencies = np.arange(5, 150, 3)
power = mne.time_frequency.tfr_morlet(epochs, n_cycles=3,  picks=[raw.info["ch_names"].index(channel)],
                                      return_itc=False, freqs=frequencies, average=False, verbose=3, use_fft=True)
power.crop(tmin=tmin, tmax=tmax)
# Compute correlation for every entry
power_data = power.data.squeeze()
n_epochs, n_freqs, n_times = power_data.shape
corr_p = np.zeros((2, n_freqs, n_times))
for i, freq in enumerate(frequencies):
    for j, time in enumerate(power.times):
        corr, p = spearmanr(peak_speed, power_data[:, i, j])
        corr_p[0, i, j] = corr
        corr_p[1, i, j] = p

# Plot
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 15))

# Plot power
power = mne.time_frequency.tfr_morlet(epochs, n_cycles=3,  picks=[raw.info["ch_names"].index(channel)],
                                      return_itc=False, freqs=frequencies, average=True, verbose=3, use_fft=True)
power.plot([channel], baseline=(tmin, tmax), axes=axes[0], tmin=tmin, tmax=tmax, mode="zscore", show=False, colorbar=False)
axes[0].set_xticks([])
axes[0].set_xlabel("")
freqs_ticks = axes[0].get_yticklabels()

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
#axes[0].colorbar()

# Plot p values
axes[3].imshow(corr_p[1, :, :], aspect="auto", vmin=0, vmax=0.01, cmap='jet', origin="lower", extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
#axes[1].colorbar()
axes[3].set_ylabel("P-value")
axes[3].set_xlabel("Time[Sec]")

# Save figure
plt.savefig(f"../../Figures/{sub}_tfr_speed_corr.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/{sub}_tfr_speed_corr.png", format="png", bbox_inches="tight", transparent=True)

plt.show()