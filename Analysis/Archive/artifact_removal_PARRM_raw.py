# Remove the stimulation artifact using PARRM and save the cleaned data

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import zscore
from scipy import signal
from pyparrm import PARRM
from mne_bids import read_raw_bids, find_matching_paths
import matplotlib
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
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
sfreq = raw.info["sfreq"]

# Apply bipolar re-reference for ecog channel
raw.load_data()
new_ch = raw.get_data("ECOG_R_02_SMC_AT")- raw.get_data(["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]).mean(axis=0)
target_chan_name = "ECOG_R_2_CAR"
u.add_new_channel(raw, new_ch, target_chan_name, type="ecog")
ch_names = raw.info["ch_names"]

# Crop dataset
raw.crop(tmin=0, tmax=raw.times[-1]-2)

# Find exact time point of stimulation onset and offset
events = mne.events_from_annotations(raw)[0]
tmin = -0.2
tmax = 0.5
epochs = mne.Epochs(raw, events=events, event_id=10004, tmin=tmin, tmax=tmax, baseline=None, picks=target_chan_name)
epochs.load_data()
idx_on = np.zeros(len(epochs))
idx_off = np.zeros(len(epochs))
for i, (data_epoch, event) in enumerate(zip(epochs.get_data(), epochs.events)):
    data_epoch = data_epoch.flatten()
    # First sample of the epoch that is above 3 standard deviations (from both sides)
    idx_stim_on = np.where(np.abs(zscore(data_epoch)) > 3)[0][0] - 10
    idx_stim_off = len(data_epoch) - np.where(np.abs(zscore(np.flip(data_epoch))) > 3)[0][0] + 15
    """plt.plot(data_epoch)
    plt.axvline(idx_stim_on, color="red")
    plt.axvline(idx_stim_off, color="red")
    plt.show(block=True)"""
    # Save as events
    idx_on[i] = event[0] + idx_stim_on + tmin*sfreq
    idx_off[i] = event[0] + idx_stim_off + tmin * sfreq
# Add the edges
idx_off = np.insert(idx_off, 0, 0).astype(int)
idx_on = np.insert(idx_on, len(idx_on), raw.times[-1]*sfreq).astype(int)

# Define a high-pass filter to center the data around 0
fc = 3  # Cut-off frequency of the filter
w = fc / (sfreq / 2) # Normalize the frequency
b, a = signal.butter(1, w, 'high')

# Loop over the stimulation onset/offset to 0-center (high-pass filter) clean data patches
data = raw.get_data(picks=target_chan_name).flatten()
data_new = data.copy()
for off, on in zip(idx_off, idx_on):

    # Get the data until the next artifact
    data_clean = data[off:on]

    # Apply a filter to center at 0
    data_filt = signal.filtfilt(b, a, data_clean)

    # Reinsert the data
    data_new[off:on] = data_filt

    """    
    # Plot
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(data_clean)
    for i in range(1,6):
        b, a = signal.butter(i, w, 'high')
        data_filt = signal.filtfilt(b, a, data_clean)
        axes[1].plot(data_filt, label=i)
    plt.legend()
    plt.show(block=True)"""

# Loop over the stimulation onset/offset to remove the artifact
for i, (off, on) in enumerate(zip(idx_off[1:], idx_on[:-1])):

    print(f"Epoch {i+1} of {len(idx_on)}")

    # Get the data during the artifact
    data_artefact = data[on:off].copy()
    data_artefact = data_artefact[np.newaxis, :]
    n_samples = data_artefact.shape[-1]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 7))
    axes[0].plot(data_artefact.flatten())
    axes[0].set_title("Raw")

    # Apply PARRM to remove the artefact
    parrm = PARRM(data=data_artefact, sampling_freq=sfreq, artefact_freq=130, verbose=False)
    parrm.find_period()
    #parrm.explore_filter_params()
    parrm.create_filter(period_half_width=0.04, filter_half_width=int((n_samples - 1) / 2))
    data_artefact_clean = parrm.filter_data().flatten()
    axes[1].plot(data_artefact_clean)
    axes[1].set_title("PARRM cleaned")

    # Remove the beginning and end artifact by replacing it with data around it
    for j in range(2):

        # Get half of the data
        tmp = data_artefact_clean[int((n_samples/2) * j):int((n_samples/2) * (j + 1))]

        # Get the start and end of the artifact in half of the data
        if j == 0 and target_chan_name == "LFP_bipolar":
            thres = 3
        else:
            thres = 3
        idx_start = np.where(np.abs(zscore(tmp)) > thres)[0][0] - 3
        if idx_start < 0:
            idx_start = 0
        idx_end = len(tmp) - np.where(np.abs(zscore(np.flip(tmp))) > thres)[0][0] + 3
        if idx_end-idx_start > 100:
            idx_end = len(tmp) - np.where(np.abs(zscore(np.flip(tmp))) > 3.5)[0][0] + 3
        if idx_end - idx_start > 100:
            idx_end = len(tmp) - np.where(np.abs(zscore(np.flip(tmp))) > 5)[0][0] + 3

        # Replace regions with data before and after
        n_half_artifact = int(np.ceil((idx_end-idx_start)/2))
        print(n_half_artifact)
        if j == 0:
            if idx_start >= n_half_artifact:
                tmp[idx_start:idx_start+n_half_artifact] = tmp[idx_start-n_half_artifact:idx_start]
            else:
                n_missing = n_half_artifact - idx_start
                tmp[idx_start:idx_start+n_half_artifact] = np.hstack((data_new[on-n_missing:on], tmp[0:idx_start]))
            tmp[idx_end-n_half_artifact:idx_end] = tmp[idx_end:idx_end+n_half_artifact]
        else:
            if len(tmp) - idx_end >= n_half_artifact:
                tmp[idx_end-n_half_artifact:idx_end] = tmp[idx_end:idx_end+n_half_artifact]
            else:
                n_missing = n_half_artifact - (len(tmp) - idx_end)
                tmp[idx_end-n_half_artifact:idx_end] = np.hstack((tmp[idx_end:], data_new[off:off+n_missing]))
            tmp[idx_start:idx_start+n_half_artifact] = tmp[idx_start-n_half_artifact:idx_start]

        # Replace in data
        data_artefact_clean[int((n_samples/2) * j):int((n_samples/2) * (j + 1))] = tmp

        # Mark replaces regions and replacement regions in plot
        ymin,ymax = axes[1].get_ylim()
        axes[1].fill_between([(n_samples/2) * j + idx_start, (n_samples/2) * j + idx_end], [ymin, ymin], [ymax, ymax], color="red", alpha=0.3)
        axes[1].fill_between([(n_samples/2) * j + idx_end, (n_samples/2) * j + idx_end+n_half_artifact], [ymin, ymin], [ymax, ymax], color="green", alpha=0.3)
        axes[1].fill_between([(n_samples/2) * j + idx_start - n_half_artifact, (n_samples/2) * j + idx_start], [ymin, ymin], [ymax, ymax], color="green", alpha=0.3)

    # Reinsert the data
    data_new[on:off] = data_artefact_clean

    # Plot with removed edge artifacts
    axes[2].plot(data_artefact_clean)
    axes[2].set_title("Edges replaced")

    # Adjust plot
    plt.suptitle("Epoch " + str(i))
    plt.subplots_adjust(hspace=0.4)

    # Save plot
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/epoch_{i}_clean_{target_chan_name}.svg",
                format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/epoch_{i}_clean_{target_chan_name}.png",
                format="png", bbox_inches="tight", transparent=False)

    plt.show(block=True)

# Inspect full dataset
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0, 0].plot(data[:-1000], color="red", label="Raw data")
axes[0, 0].set_title("Raw data")
raw.plot_psd(axes=axes[1, 0], picks=[target_chan_name], show=False, fmin=5, fmax=200)

# Reinsert into raw dataset and plot psd
axes[0, 1].plot(data_new[:-1000], color="green")
axes[0, 1].set_title("Cleaned data")
raw_new = raw.copy()
raw_new._data[ch_names.index(target_chan_name), :] = data_new
raw_new.plot_psd(axes=axes[1, 1], picks=[target_chan_name], show=False, fmin=5, fmax=200)
plt.legend()
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.suptitle(target_chan_name)

plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}.svg",
            format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}.png",
            format="png", bbox_inches="tight", transparent=False)

# Save the cleaned data
save_path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_{target_chan_name}_cleaned.fif"
raw_new.save(save_path, overwrite=True)
plt.show(block=True)