# Plot the power over time

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import scipy
import matplotlib
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
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
raw.drop_channels(raw.info["bads"])
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]

if sub == "EL008":
    target_chan_name = "ECOG_L_05_SMC_AT"
    target_chan_name = "ECOG_L_06_SMC_AT"
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"
    target_chan_name2 = "ECOG_R_03_SMC_AT"

# Rereference to ecog channel
raw._data[ch_names.index(target_chan_name), :] = raw.get_data(target_chan_name) - raw.get_data(target_chan_name2)

# Filter out line noise
raw.notch_filter(50)

# Get the timepoint of the start of the last block (recovery)
idx_start = np.where(raw.get_data("BLOCK").flatten() == 4)[0][0]
idx_end = np.where(raw.get_data("BLOCK").flatten() == 3)[0][0]

# Crop dataset to the last block
raw.crop(tmin=idx_start / sfreq, tmax=idx_end / sfreq)

# Compute power over time
fmin_array = [15, 60]
fmax_array = [45, 90]
tmin = 1
tmax = 50

fig, ax = plt.subplots(2, 1, figsize=(15, 10))
for i, (fmin, fmax) in enumerate(zip(fmin_array, fmax_array)):

    # compute tfr
    tfr = raw.compute_tfr(method="multitaper", freqs=np.arange(fmin, fmax, 2), picks=[target_chan_name], tmin=tmin, tmax=tmax)

    # Plot tfr
    tfr.plot(axes=ax[i], cmap="jet", tmin=tmin+2, tmax=tmax-2, colorbar=False, show=False)

    # Add speed
    speed = raw.get_data(["SPEED"], tmin=tmin+2, tmax=tmax-2).flatten()
    times_array = raw.times
    times_array = times_array[np.where(np.logical_and(times_array >= tmin+2, times_array < tmax-2))]
    speed_scaled = u.scale_min_max(speed, fmin + 2, fmax - 2, speed.min(axis=0), speed.max(axis=0))
    ax[i].plot(times_array, speed_scaled, color="black", alpha=0.5)

    # Average the tfr over frequencies and plot the average
    power_average = tfr.get_data(tmin=tmin+2, tmax=tmax-2).squeeze().mean(axis=0)
    power_average_scaled = u.scale_min_max(power_average, fmin+2, fmax-1, power_average.min(), power_average.max())
    ax[i].plot(times_array, power_average_scaled[:len(times_array)], color="red", alpha=0.5)

    # Compute the correlation
    corr, p = pearsonr(speed, power_average[:len(speed)])
    ax[i].set_title(f"Corr: {corr:.3f}, p = {p:.3f}")

plt.subplots_adjust(hspace=0.3)
plt.show()
print("Test")

