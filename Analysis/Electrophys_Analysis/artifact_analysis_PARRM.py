# Inspect the artifact and check how to deal with it using PARRM

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import os
from pyparrm import PARRM
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
import matplotlib
matplotlib.use('QtAgg')

# Set the dataset
sub = "EL012"
med = "Off"
fmin = 2
fmax = 100
epoch_idx = 2
fmin_tfr = 30
tmin = -0.031
tmax = 0.271
tmin_tfr = tmin
tmax_tfr = tmax
reref = True

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
tfr = epochs.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 0], show=False, tmin=tmin_tfr, tmax=tmax_tfr, cmap="jet")

# PARRM filtered data
data = data[np.newaxis, :]
parrm = PARRM(data=data,sampling_freq=sfreq,artefact_freq=130,verbose=False)
parrm.find_period()
parrm.create_filter(period_half_width=0.04, filter_half_width=599)
filtered_data = parrm.filter_data()
epochs_PARRM = epochs.copy()
epochs_PARRM.load_data()
epochs_PARRM._data[epoch_idx, ch_names.index(target_chan_name) :] = filtered_data[0]
data_PARRM = epochs_PARRM[epoch_idx].get_data([target_chan_name]).squeeze()
axes[0, 1].plot(epochs_PARRM.times, data_PARRM)
axes[0, 1].set_title(f"PARRM filtered data")
tfr = epochs_PARRM.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 1], show=False, tmin=tmin_tfr, tmax=tmax_tfr, cmap="jet")

# Band-pass filtered data
epochs_filt = epochs_PARRM.filter(fmin, fmax)
data_filt = epochs_filt[epoch_idx].get_data([target_chan_name]).squeeze()
axes[0, 2].plot(epochs_filt.times,data_filt)
axes[0, 2].set_title(f"Band-pass filter {fmin}-{fmax} Hz")
# TFR
tfr = epochs_filt.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 2], show=False, tmin=tmin_tfr, tmax=tmax_tfr, cmap="jet")

plt.suptitle(f"Epoch {epoch_idx} channel {target_chan_name} ecog ref {reref}")
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Save plot
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{reref}.svg",
        format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{reref}.png",
        format="png", bbox_inches="tight", transparent=True)
plt.show(block=True)

