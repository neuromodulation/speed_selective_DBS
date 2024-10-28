# Plot the spectral power

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
sub = "EL008"
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
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Annotate periods with stimulation
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.01
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
raw.set_annotations(stim_annot)

# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-0.3, tmax=0, baseline=None, reject_by_annotation=True)
epochs_samples = epochs.selection.copy()
epochs.drop_bad()
drop_sample = np.where(list(map(lambda x: x == ('bad stim',), epochs.drop_log)))[0]
drop_idx = np.where(epochs_samples == drop_sample)[0]

# Plot the power spectral density
epochs.load_data()
psd = epochs.compute_psd(method='welch', fmin=10, fmax=100, picks=[target_chan_name], n_fft=int(0.3*sfreq))

# Plot
plt.figure()
mean_psd = psd.get_data().mean(axis=0).flatten()
std_psd = psd.get_data().std(axis=0).flatten()
plt.plot(psd.freqs, mean_psd)
#plt.fill_between(psd.freqs, mean_psd - std_psd, mean_psd + std_psd, alpha=0.3)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
y_max, y_min = plt.ylim()

# Mark the bands
bands = [[13, 40], [60, 100]]
colors = ["#00863b", "#3b0086"]
for i, band in enumerate(bands):
    plt.fill_between(band, y_min, y_max, color=colors[i], alpha=0.3)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{band[0]}_{band[-1]}.svg",
            format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{band[0]}_{band[-1]}.png",
            format="png", bbox_inches="tight", transparent=True)

plt.show()
