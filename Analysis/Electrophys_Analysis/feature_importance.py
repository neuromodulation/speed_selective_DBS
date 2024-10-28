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

# Add CAR channels
ecog_names = ["ECOG_R_1_CAR", "ECOG_R_2_CAR", "ECOG_R_3_CAR"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")

ch_names = ["ECOG_R_1_CAR", "ECOG_R_2_CAR", "ECOG_R_3_CAR", "SPEED_MEAN"]
ch_types = ["ecog", "ecog", "ecog",  "BEH"]
nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names, ch_types=ch_types, target_keywords="SPEED_MEAN", reference=None)

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
tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][0] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][191] / sfreq
data_1 = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][192] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][383] / sfreq
data_2 = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

data = np.hstack((data_1, data_2))

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
data = np.array(features)[:, :-2]
speed = np.array(features.SPEED_MEAN)

# Fit a catboost model and plot the feature importance
model = CatBoostRegressor(iterations=50,
                          depth=5,
                          loss_function='RMSE', learning_rate=0.42)
n_stack = 10
X, y = u.append_previous_n_samples(X=data, y=speed, n=n_stack)
model.fit(X, y)
feature_importance = model.get_feature_importance()
feature_importance = np.reshape(feature_importance, (n_stack, data.shape[-1]))
mean_fi_chans = np.mean(feature_importance, axis=0)
n_chans = len(ch_names)-1
mean_fi = np.reshape(mean_fi_chans, (n_chans, int(data.shape[-1]/n_chans)))

# Plot as bar plot
fontsize = 7
label_names = ["Raw signal amplitude", "$\\alpha$ (8-12 Hz)", "Low $\\beta$ (13-20 Hz)", "High $\\beta$ (20-35 Hz)",
               "Low $\gamma$ (60-80 Hz)", "High $\gamma$ (90-200 Hz)", "HFA(200-400 Hz)"]
fig, ax = plt.subplots(figsize=(2, 1.3))
x = np.arange(mean_fi.shape[-1])
colors = ["#b6daea", "#7f98a3", "#48575d"]
for i in range(n_chans):
    ax.barh(x+(i*0.2), mean_fi[i][::-1], height=0.2, color=colors[i])
ax.set_xticks([])
ax.set_yticks(x, label_names[::-1], fontsize=fontsize)
ax.set_xlabel(f"Feauture importance", fontsize=fontsize)
plt.legend(["ECoG 1", "ECoG 2", "ECoG 3"], fontsize=fontsize, bbox_to_anchor=(1, 1))
u.despine()

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