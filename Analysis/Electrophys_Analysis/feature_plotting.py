# Plot the features centered on the peak speed and the feature importance

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
import random
import pandas as pd
from sklearn import metrics, model_selection, linear_model
import py_neuromodulation as nm
from catboost import CatBoostRegressor, Pool
from xgboost import XGBClassifier
from scipy.stats import zscore
from scipy.signal import find_peaks
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
)
import pickle
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')
random.seed(420)

# Set analysis parameters
samp_freq = 37
seg_ms = 354

# Load the data
sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\{sub}_cleaned.fif"
raw = mne.io.read_raw_fif(path).load_data()

sfreq = raw.info["sfreq"]
target_chan_name = raw.info["ch_names"][-1]
events = mne.events_from_annotations(raw)[0]

# Add a channel which marks the peak speed
peaks_idx = events[np.where((events[:, 2] == 3)), 0].flatten()
peaks_idx_ext = np.array([np.arange(x-60, x+60) for x in peaks_idx]).flatten()
peaks = np.zeros(raw._data.shape[-1])
peaks[peaks_idx_ext] = 1
u.add_new_channel(raw, peaks[np.newaxis, :], "peaks", type="misc")

# Add CAR channels
ecog_names = ["ECOG_R_1_CAR", "ECOG_R_2_CAR", "ECOG_R_3_CAR"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")

# Set channels
ch_names = ["ECOG_R_3_CAR", "SPEED_MEAN", "peaks"]
ch_types = ["ecog", "BEH", "BEH"]
nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names, ch_types=ch_types, target_keywords=["SPEED_MEAN", "peaks"], reference=None)

# Settings
settings = nm_settings.get_default_settings()
settings = nm_settings.reset_settings(settings)
settings["features"]["fft"] = True
settings["features"]["return_raw"] = True
settings["sampling_rate_features_hz"] = samp_freq
settings["segment_length_features_ms"] = seg_ms
settings["fft_settings"]["windowlength_ms"] = seg_ms
del settings["frequency_ranges_hz"]["theta"]
settings["postprocessing"]["feature_normalization"] = True
settings["feature_normalization_settings"]["normalization_time_s"] = 1
settings["feature_normalization_settings"]["normalization_method"] = "zscore"

# Attach the blocks together
tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][97] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][191] / sfreq
data_1 = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][192+97] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][383] / sfreq
data_2 = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

data = data_1 # np.hstack((data_1, data_2))

# Start the stream
stream = nm.Stream(
            settings=settings,
            nm_channels=nm_channels,
            verbose=False,
            sfreq=sfreq,
            line_noise=50
        )

features = stream.run(data=data)

# Plot the features centered on the peak speed
data = np.array(features.T)[:-3, :]
peaks = np.array(features.peaks)
peaks_idx = np.where(peaks == 1)[0]
speed = np.array(features.SPEED_MEAN)

# Plot index for viual inspection
"""plt.plot(speed)
for idx in peaks_idx:
    plt.axvline(idx)"""

tmin = -int(1 * samp_freq)
tmax = int(2.5 * samp_freq)
data_all = []
speed_all = []
for idx in peaks_idx:
    if idx + tmin > 0 and idx + tmax < data.shape[1]:
        data_all.append(data[:, idx + tmin:idx + tmax])
        speed_all.append(speed[idx + tmin:idx + tmax])
feature_mean = np.array(data_all).mean(axis=0)
speed_mean = np.array(speed_all).mean(axis=0)

# Plot as imshow
fontsize=7
labels = np.array(features.keys()[:7])
label_names = ["Raw signal", "Alpha", "Low Beta", "High Beta", "Low Gamma", "High Gamma", "HFA"]

fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 7], figsize=(3, 1.5), constrained_layout=True)

ax1.plot(speed_mean)
ax1.set_yticks([])
ax1.set_xticks([])
ax1.spines[["right", "top", "bottom"]].set_visible(False)
ax1.set_ylabel("Speed", fontsize=fontsize, rotation=0, labelpad=20)

im = ax2.imshow(feature_mean, aspect="auto")
ax2.set_yticks(np.arange(7), label_names, fontsize=fontsize)
ax2.yaxis.set_tick_params(labelsize=fontsize)
ax2.set_xlabel("Time (s)", fontsize=fontsize)
times = np.arange(tmin, tmax)
time_idx = np.arange(0, len(times), 30)
label_times = np.round(times[time_idx] / samp_freq, 2)
ax2.set_xticks(time_idx, label_times, fontsize=fontsize)
ax2.spines[["right", "top"]].set_visible(False)

cbar = fig.colorbar(im, ax=ax2)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('Feature strength (z-score)', fontsize=fontsize)

plt.subplots_adjust(left=0.4, hspace=0.3, wspace=0.3)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
    f"../../../Figures/{dir_name}/{plot_name}.pdf",
    format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(
    f"../../../Figures/{dir_name}/{plot_name}.png",
    format="png", bbox_inches="tight", transparent=False)
plt.show()