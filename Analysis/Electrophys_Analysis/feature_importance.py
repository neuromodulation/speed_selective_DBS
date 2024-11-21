# Plot the features centered on the peak speed and the feature importance

import os
import numpy as np
import mne
import py_neuromodulation as nm
import pandas as pd
from catboost import CatBoostRegressor
import random
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "../../../Code")
import utils as u
random.seed(420)
import matplotlib
matplotlib.use('TkAgg')

# Load the data
path = "EL012_ECoG_CAR_LFP_BIP_small.fif"
raw = mne.io.read_raw_fif(path).load_data()
events = mne.events_from_annotations(raw)[0]
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
ecog_names = [name for name in ch_names if "ECOG" in name]
lfp_names = [name for name in ch_names if "LFP" in name]
ch_names = ecog_names + ["SPEED_MEAN"]
ch_types = ["ecog"] * len(ecog_names) + ["BEH"]

# Get the data
tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][int(96*1)] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][int(96 * 2) - 1] / sfreq
data = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

# Load the optimal parameters (for all ECoG channels)
filename = f"results_4/feature_model_optimization_ECoG_combined_2.xlsx"
df = pd.read_excel(filename, sheet_name=f"Fold 2")
samp_freq, seg_ms, n_stack, depth, learning_rate, _ = df.loc[df['all'].idxmax()]

channels = nm.utils.set_channels(
        ch_names=ch_names,
        ch_types=ch_types,
        reference=None,
        used_types=ch_types[:-1],
        target_keywords=["SPEED_MEAN"],
    )

# Compute performance on the test set using the optimal parameters
settings = nm.NMSettings.get_fast_compute()
settings.features.fft = True
settings.features.return_raw = True
settings.sampling_rate_features_hz = samp_freq
settings.segment_length_features_ms = seg_ms
settings.fft_settings.windowlength_ms = seg_ms
del settings.frequency_ranges_hz["theta"]
settings.postprocessing.feature_normalization = True
settings.feature_normalization_settings.normalization_time_s = 1
settings.feature_normalization_settings.normalization_method = "zscore"
settings.preprocessing = settings.preprocessing[:2]

# Compute features
stream = nm.Stream(
    settings=settings,
    channels=channels,
    verbose=False,
    sfreq=sfreq,
    line_noise=50
)
features = stream.run(data=data)

# Plot the features centered on the peak speed
data = np.array(features)[:, :-2]
speed = np.array(features.SPEED_MEAN)

# Fit a catboost model and plot the feature importance
n_stack = int(n_stack)
model = CatBoostRegressor(iterations=30,
                          depth=int(depth),
                          loss_function='RMSE', learning_rate=learning_rate)
X, y = u.append_previous_n_samples(X=data, y=speed, n=n_stack)
model.fit(X, y)
feature_importance = model.get_feature_importance()
feature_importance = np.reshape(feature_importance, (n_stack, data.shape[-1]))
mean_fi_chans = np.mean(feature_importance, axis=0)
n_chans = len(ch_names)-1
mean_fi = np.reshape(mean_fi_chans, (n_chans, int(data.shape[-1]/n_chans)))

# Plot
fontsize = 6
label_names = ["Raw signal amplitude", "$\\alpha$ (8-12 Hz)", "Low $\\beta$ (13-20 Hz)", "High $\\beta$ (20-35 Hz)",
               "Low $\gamma$ (60-80 Hz)", "High $\gamma$ (90-200 Hz)", "HFA(200-400 Hz)"]
fig, ax = plt.subplots(figsize=(2.3, 1.3))
im = ax.imshow(mean_fi.T, aspect="auto", cmap="Blues")
# Adjust
ax.set_xticks(np.arange(5), ["ECoG 1", "ECoG 2", "ECoG 3", "ECoG 4", "ECoG 5"], fontsize=fontsize, rotation=25)
ax.set_yticks(np.arange(7), label_names, fontsize=fontsize)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
cbar = fig.colorbar(im, ax=ax)
cbar.set_ticks([])
cbar.set_label('Feature importance', fontsize=fontsize)
u.despine(sides=["right", "bottom"])

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