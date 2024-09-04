# Inpsect and try to remove the stimulation artifact

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from sklearn.decomposition import FastICA
from scipy.stats import zscore
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
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
sfreq = raw.info["sfreq"]

# Drop bad channels
raw.drop_channels(raw.info["bads"])

# Add channel
new_chan = raw.get_data(["ECOG_R_01_SMC_AT"]) - np.mean(raw.get_data(picks="ecog"), axis=0)
u.add_new_channel(raw, new_chan, "ecog_1_car", type="ecog")

# Filter out line noise
raw.notch_filter(50)
raw.notch_filter(130)

# Play around with ICA
X = raw.get_data(picks=["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT"]).T
transformer = FastICA(n_components=2, random_state=0, whiten='unit-variance')
X_transformed = transformer.fit_transform(X)
plt.figure(figsize=(12, 12))
for i in range(len(X_transformed.T)):
    plt.subplot(6, 1, i+1)
    plt.plot(np.abs(zscore(X_transformed[:, i].flatten())))
    #plt.axhline(15, color="red")
plt.show()
# Transform back
X_trans_new = X_transformed.copy()
X_trans_new[:, 0] = np.zeros(X_trans_new[:, 0].shape)
# Project back
X_new = transformer.inverse_transform(X_trans_new)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Annotate periods with stimulation
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.1
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
#raw.set_annotations(stim_annot)
raw_no_stim = raw.copy()
raw_no_stim.set_annotations(stim_annot)

# Plot time-frequency spectrum for all events
event_list = [10002, 10003, 10001]
event_names = ["Movement start", "Peak speed", "Movement end"]
tmin = -0.7
tmax = 0.7
channel = "ecog_1_car"
fmin = 5
fmax=45
frequencies = np.arange(fmin, fmax, 1)

fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6, 15))
fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))
for i, event_id in enumerate(event_list):

    # Cut into epochs - With stimulation epochs
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin-0.5, tmax=tmax+0.5, baseline=None)

    # Compute the power spectrum
    psd = epochs.compute_psd(fmin=fmin, fmax=fmax, picks=channel)
    axes2[i].plot(psd.get_data().mean(axis=0).flatten(), color="red")

    # Cut into epochs - Without stimulation epochs
    epochs_no_stim = mne.Epochs(raw_no_stim, events=events, event_id=event_id, tmin=-1.2, tmax=1.2, baseline=None, reject_by_annotation=True)
    epochs_no_stim.drop_bad()

    # Compute the power spectrum
    psd = epochs_no_stim.compute_psd(fmin=fmin, fmax=fmax, picks=channel)
    axes2[i].plot(psd.get_data().mean(axis=0).flatten(), color="black")

    # Plot the speed
    behav = np.mean(epochs.get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
    axes[2*i+1].plot(times, behav, color="black")
    if i < 2:
        axes[2*i+1].set_xticks([])
    else:
        axes[2*i+1].set_xlabel("Time [Sec]")
    axes[2*i+1].set_ylabel("Mean \nSpeed")
    axes[2*i+1].set_yticks([])
    axes[2 * i+1].axvline(0, color="red", linewidth=2)

    # Plot the power over time
    power = mne.time_frequency.tfr_morlet(epochs_no_stim, n_cycles=7, picks=[raw.info["ch_names"].index(channel)],
                                          return_itc=False, freqs=frequencies, average=True, verbose=3, use_fft=True)
    power.plot(axes=axes[2*i], baseline=(tmin, tmax), tmin=tmin, tmax=tmax, mode="zscore", show=False, colorbar=False, cmap="jet")
    axes[2 * i].set_xticks([])
    axes[2 * i].set_xlabel("")
    axes[2 * i].axvline(0, color="red", linewidth=2)

# Save figure
plt.savefig(f"../../Figures/{sub}_tfr_{channel}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/{sub}_tfr_{channel}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()