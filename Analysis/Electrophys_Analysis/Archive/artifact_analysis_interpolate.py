# Inspect whetehr the pulses can be interpolated

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
from scipy.stats import zscore
matplotlib.use('Qt5Agg')

# Set the dataset
sub = "EL012"
med = "Off"
fmin = 2
fmax = 90
epoch_idx = 0
fmin_tfr = 40
tmin = -0.5
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
events = mne.events_from_annotations(raw)[0]
event_id = 10004
#raw.filter(fmin, None)

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

# Plor raw channel
plt.figure()
plt.plot(raw.get_data(target_chan_name).flatten())
event_stim = []
for i in range(40):
    event_stim.append([int(1017668+(i*31)), 0, 1])
event_stim = np.array(event_stim)
mapping = {
        1: "stim pulse"}
annot_from_events = mne.annotations_from_events(events=event_stim, event_desc=mapping, sfreq=raw.info["sfreq"])
raw.set_annotations(annot_from_events)
#mne.preprocessing.fix_stim_artifact(raw, events=event_stim, event_id=1, tmin=-(4/sfreq), tmax=4/sfreq, mode="linear")
tmp = ["LFP_R_03_STN_MT", "LFP_L_06_STN_MT"]
plt.plot(raw.get_data(tmp).flatten())
plt.legend(tmp)
#plt.plot(raw.get_data(target_chan_name).flatten())
plt.show()

# Set events
mapping = {
        10004: "stim onset"}
annot_from_events = mne.annotations_from_events(events=events, event_desc=mapping, sfreq=raw.info["sfreq"])
raw.set_annotations(annot_from_events)

# Create plot with time series and tfr for one epoch (raw, filtered, ICA)
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Raw data
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)
data = epochs[epoch_idx].get_data([target_chan_name]).squeeze()
axes[0, 0].plot(epochs.times, data)
axes[0, 0].set_title(f"interpolated")
axes[0, 0].axvline(0, color="red")
# TFR
tfr = epochs.compute_tfr(method="multitaper", freqs=np.arange(fmin_tfr, fmax, 2), picks=[target_chan_name])
tfr[epoch_idx].plot(axes=axes[1, 0], show=False, tmin=0.2, tmax=tmax-0.2, cmap="jet")

# Save plot
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{reref}.svg",
        format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{reref}.png",
        format="png", bbox_inches="tight", transparent=True)
plt.show()

