# Plot time frequency plots centered on movement starte/end and peak speed

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
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

# Extract events
events = mne.events_from_annotations(raw)[0]

# Annotate periods with stimulation
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.1
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
raw.set_annotations(stim_annot)

# Plot time-frequency spectrum for all events
event_list = [10002, 10003, 10001]
event_names = ["Movement start", "Peak speed", "Movement end"]
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(6, 15))
tmin = -0.7
tmax = 0.7
channel = "LFP_L_05_STN_MT"
channel = "ecog_1_car"
frequencies = np.arange(8, 100, 1)
for i, event_id in enumerate(event_list):

    # Cut into epochs
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1, baseline=None, reject_by_annotation=True)
    epochs.drop_bad()

    # Plot the average power spectrum of the epochs
    #ecog_bipolar = raw.info["ch_names"][-5:]
    #epochs.plot_psd(fmin=3, fmax=100, average=False, picks=ecog_bipolar)

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
    power = mne.time_frequency.tfr_morlet(epochs, n_cycles=7, picks=[raw.info["ch_names"].index(channel)],
                                          return_itc=False, freqs=frequencies, average=True, verbose=3, use_fft=True)
    power.plot(axes=axes[2*i], baseline=(tmin, tmax), tmin=tmin, tmax=tmax, mode="zscore", show=False, colorbar=False, cmap="jet")
    axes[2 * i].set_xticks([])
    axes[2 * i].set_xlabel("")
    axes[2 * i].axvline(0, color="red", linewidth=2)

# Save figure
plt.savefig(f"../../Figures/{sub}_tfr_{channel}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/{sub}_tfr_{channel}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()