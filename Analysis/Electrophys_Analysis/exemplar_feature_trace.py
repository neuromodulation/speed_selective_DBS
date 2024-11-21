# Plot the features centered on the peak speed and the feature importance

import os
import numpy as np
import mne
import py_neuromodulation as nm
import pandas as pd
import random
import matplotlib.pyplot as plt
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
ch_names = [ecog_names[0], "SPEED_MEAN"]
ch_types = ["ecog", "BEH"]

# Get the data
tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][int(96*0)] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][int(96 * 1) - 1] / sfreq
data = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

# Load the optimal parameters (for all ECoG channels)
filename = f"results_1/feature_model_optimization_ECoG_combined_0.xlsx"
df = pd.read_excel(filename, sheet_name=f"Fold 0")
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
settings.features.raw_hjorth = False
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
speed = np.array(features.SPEED_MEAN)
feature_matrix = np.array(features.T)[:-2, :]

# Plot the speed and features
fontsize = 6
labels = np.array(features.keys()[:7])
label_names = ["Raw signal amplitude", "$\\alpha$ (8-12 Hz)", "Low $\\beta$ (13-20 Hz)", "High $\\beta$ (20-35 Hz)", "Low $\gamma$ (60-80 Hz)", "High $\gamma$ (90-200 Hz)", "HFA(200-400 Hz)"]

fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 5], figsize=(3, 1.5), constrained_layout=True)
idx_start = 300#20
idx_end = 1365
ax1.plot(speed[idx_start:idx_end])
ax1.set_yticks([])
ax1.set_xticks([])
ax1.spines[["right", "top", "bottom"]].set_visible(False)
ax1.set_ylabel("Speed", fontsize=fontsize, rotation=0, labelpad=20)

# Plot the features
im = ax2.imshow(feature_matrix[:, idx_start:idx_end], aspect="auto")

# Adjust plot
ax2.set_yticks(np.arange(7), label_names, fontsize=fontsize)
ax2.yaxis.set_tick_params(labelsize=fontsize)
ax2.set_xlabel("Time (s)", fontsize=fontsize)
xticks = ax2.get_xticks()
tick_labels = np.round(xticks/samp_freq)
ax2.set_xticklabels(tick_labels, fontsize=fontsize)
ax2.spines[["right", "top"]].set_visible(False)
cbar = fig.colorbar(im, ax=ax2)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('Feature strength (z-score)', fontsize=fontsize)
plt.subplots_adjust(left=0.4, hspace=0.3, wspace=0.3)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
    f"../../../Figures/{dir_name}/features_move/{plot_name}.pdf",
    format="pdf", bbox_inches="tight", transparent=True,  dpi=300)
plt.savefig(
    f"../../../Figures/{dir_name}/features_move/{plot_name}.png",
    format="png", bbox_inches="tight", transparent=False)
plt.show()