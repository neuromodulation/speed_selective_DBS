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

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects="EL012",
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMedOff01", f"EcogLfpMedOff01",
                                              f"LfpMedOff02", f"EcogLfpMedOff02", f"LfpMedOffDys01"])
raw = read_raw_bids(bids_path=bids_path[0])
raw.load_data()
new_ch = raw.get_data("ECOG_R_02_SMC_AT") - raw.get_data("ECOG_R_03_SMC_AT")
target_chan_name = "ECOG_bipolar"
u.add_new_channel(raw, new_ch, target_chan_name, type="ecog")

sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\{sub}_cleaned.fif"
raw = mne.io.read_raw_fif(path).load_data()

sfreq = raw.info["sfreq"]
target_chan_name = raw.info["ch_names"][-1]
events = mne.events_from_annotations(raw)[0]

# Set channels
ch_names = ["ECOG_bipolar", "SPEED_MEAN"]
ch_types = ["ecog", "BEH"]
nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names, ch_types=ch_types, reference=None, target_keywords="SPEED_MEAN")

# Settings
settings = nm_settings.get_default_settings()
settings = nm_settings.reset_settings(settings)
settings["features"]["fft"] = True
settings["sampling_rate_features_hz"] = 60
settings["segment_length_features_ms"] = 300
settings["fft_settings"]["windowlength_ms"] = 300
#settings["raw_normalization_settings"]["normalization_time_s"] = None
#settings["feature_normalization_settings"]["normalization_time_s"] = None

# Calculate the features

# Attach the blocks together
tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][0] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][192] / sfreq
data_1 = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][192] / sfreq
tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][384] / sfreq
data_2 = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)

data = np.vstack((data_1, data_2))

# Plot for visual inspection (mean speed)
fig, ax = plt.subplots(1, 2)
ax.plot(data[-1, :].flatten())

# Start the stream
stream = nm.Stream(
            settings=settings,
            nm_channels=nm_channels,
            verbose=False,
            sfreq=sfreq,
            line_noise=50
        )

features = stream.run(data=data, out_path_root="../../../Data/Off/processed_data\\", folder_name=f"{}")

plt.show()