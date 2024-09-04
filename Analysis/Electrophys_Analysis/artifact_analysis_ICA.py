# Inspect the artifact and check how to deal with it

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
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
sub = "EL008"
med = "Off"
fmin = 2
fmax = 90
epoch_idx = 2
fmin_tfr = 20
tmin = -0.2
tmax = 0.8
reref = False

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

# Select channel
if sub == "EL008":
    target_chan_name = "ECOG_L_05_SMC_AT"
    ref_chan = "ECOG_L_01_SMC_AT"
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"
    ref_chan = "ECOG_R_01_SMC_AT"

if reref:
    for chan in raw.copy().pick(picks=["ecog"]).ch_names:
        if chan != ref_chan:
            # Rereference to ecog channel
            raw._data[ch_names.index(chan), :] = raw.get_data(chan) - raw.get_data(ref_chan)
    # Delete ref channel
    raw.set_channel_types(mapping={ref_chan: 'eeg'})


# Extract events
events = mne.events_from_annotations(raw)[0]
event_id = 10004

# Create plot with time series and tfr for one epoch (raw, filtered, ICA)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Raw data
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)
data = epochs[epoch_idx].get_data([target_chan_name]).squeeze()
axes[0, 0].plot(epochs.times, data)
axes[0, 0].set_title(f"raw data")
axes[0, 0].axvline(0, color="red")
#axes[0, 0].text(0, 0, "stim onset")
# TFR
tfr = epochs.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 0], show=False, tmin=tmin+0.2, tmax=tmax-0.2, cmap="jet")

# Filter raw object
raw_filt = raw.copy().filter(fmin, fmax, method='fir')
epochs_filt = mne.Epochs(raw_filt, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, picks=["ecog"])
data_filt = epochs_filt[epoch_idx].get_data([target_chan_name]).squeeze()
axes[0, 1].plot(epochs_filt.times,data_filt)
axes[0, 1].set_title(f"Band-pass filter {fmin}-{fmax} Hz")
# TFR
tfr = epochs_filt.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 1], show=False, tmin=tmin+0.2, tmax=tmax-0.2, cmap="jet")

# Apply ICA
n_ecog = len(epochs_filt.ch_names)
ica = ICA(n_components=n_ecog, random_state=42)
ica.fit(epochs_filt)
#ica.plot_sources(epochs_filt)
ica.exclude = [0, 1]
epochs_ICA = epochs_filt.copy()
epochs_ICA.load_data()
ica.apply(epochs_ICA)
data_ICA = epochs_ICA[epoch_idx].get_data([target_chan_name]).squeeze()
axes[0, 2].plot(epochs_filt.times, data_ICA)
axes[0, 2].set_title(f"ICA of band-pass filtered data")
# TFR
tfr = epochs_ICA.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 2], show=False, tmin=tmin+0.2, tmax=tmax-0.2, cmap="jet")

plt.suptitle(f"Epoch {epoch_idx} channel {target_chan_name} ecog ref {reref}")
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Save plot
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{reref}.svg",
        format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{reref}.png",
        format="png", bbox_inches="tight", transparent=True)

# Save raw data for PARRM test
np.save("burst_stim.npy", data)
plt.show()

