# Decode speed using py_neuromodulation

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from mne_bids import read_raw_bids, find_matching_paths
import py_neuromodulation as nm
from py_neuromodulation import nm_analysis, nm_define_nmchannels, nm_plots, nm_settings
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Set analysis parameters
samp_freq = 37
seg_ms = 354

# Load the data
sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\{sub}_cleaned.fif"
#path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_123_CAR.fif"
raw = mne.io.read_raw_fif(path).load_data()

sfreq = raw.info["sfreq"]
target_chan_name = raw.info["ch_names"][-1]
events = mne.events_from_annotations(raw)[0]

# Add average LFP channel (level 1)
"""new_ch = raw.get_data(["ECOG_R_03_SMC_AT"]) - raw.get_data(["ECOG_R_02_SMC_AT"])
target_chan_name = "ECOG_3_2"
u.add_new_channel(raw, new_ch, target_chan_name, type="ecog")
new_ch = raw.get_data(["ECOG_R_03_SMC_AT"]) - raw.get_data(["ECOG_R_01_SMC_AT"])
target_chan_name = "ECOG_3_1"
u.add_new_channel(raw, new_ch, target_chan_name, type="ecog")"""

# Set channels
#ch_names = ["ECOG_R_1_CAR", "ECOG_R_2_CAR", "ECOG_R_3_CAR", "SPEED_MEAN"]
#ch_types = ["ecog", "ecog", "ecog", "BEH"]
ch_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT", "SPEED_MEAN"]
ch_types = ["ecog", "ecog", "ecog", "BEH"]
nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names, ch_types=ch_types, target_keywords="SPEED_MEAN")

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

# Plot for visual inspection (mean speed)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(data[1, :].flatten())
ax2.plot(data[0, :].flatten())

# Start the stream
stream = nm.Stream(
            settings=settings,
            nm_channels=nm_channels,
            verbose=False,
            sfreq=sfreq,
            line_noise=50
        )

features = stream.run(data=data, out_path_root="..\\..\\..\\Data\\Off\\processed_data\\", folder_name=f"{samp_freq}_{seg_ms}")

plt.show()