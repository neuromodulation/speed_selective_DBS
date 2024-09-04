# Remove the stimulation artifact using PARRM and save the cleaned data

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import zscore
from pyparrm import PARRM
from mne_bids import read_raw_bids, find_matching_paths
import matplotlib
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
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
raw = read_raw_bids(bids_path=bids_path[0])
raw.drop_channels(raw.info["bads"])
raw.pick(picks=["ecog", "dbs"])
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]

# Apply bipolar re-reference for ecog channels
raw.load_data()
for i, chan in enumerate(raw.copy().pick(picks="ecog").info["ch_names"]):
    # Bipolar average
    #new_chan = raw.get_data(picks="ecog")[i, :] - np.mean(raw.get_data(picks="ecog"), axis=0)
    #av_bipolar_chan_name = f"{chan}_bipolar"
    #u.add_new_channel(raw, new_chan[np.newaxis, :], av_bipolar_chan_name, type="ecog")
    # Bipolar neighbouring
    if i == 0 or i<len(ch_names)-1:
        new_chan = raw.get_data(picks="ecog")[i, :] - raw.get_data(picks="ecog")[i+1, :]
        chan2 = ch_names[i+1]
    else:
        new_chan = raw.get_data(picks="ecog")[i, :] - raw.get_data(picks="ecog")[i-1, :]
        chan2 = ch_names[i - 1]
    bipolar_chan_name = f"{chan}_{chan2}_bipolar"
    u.add_new_channel(raw, new_chan[np.newaxis, :], bipolar_chan_name, type="ecog")
    """fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(raw.get_data(picks=[chan]).flatten()[int(254.3*sfreq):int(254.6*sfreq)], color="blue")
    ax[1].plot(raw.get_data(picks=[av_bipolar_chan_name]).flatten()[int(254.3*sfreq):int(254.6*sfreq)], color="orange")
    ax[2].plot(raw.get_data(picks=[bipolar_chan_name]).flatten()[int(254.3*sfreq):int(254.6*sfreq)], color="red")
    ax[0].set_title(chan)
    ax[1].set_title(av_bipolar_chan_name)
    ax[2].set_title(bipolar_chan_name)
    plt.subplots_adjust(wspace=0.5)
    plt.show(block=True)"""

# Add average LFP channels
target_chans1 = ['LFP_L_05_STN_BS', 'LFP_L_04_STN_BS', 'LFP_L_03_STN_BS']
target_chans2 = ['LFP_L_10_STN_BS', 'LFP_L_11_STN_BS', 'LFP_L_12_STN_BS']
new_ch = raw.get_data(target_chans1).mean(axis=0) - raw.get_data(target_chans2).mean(axis=0)
u.add_new_channel(raw, new_ch[np.newaxis, :], "LFP_L_av_bipolar", type="dbs")
target_chans1 = ['LFP_R_05_STN_BS', 'LFP_R_04_STN_BS', 'LFP_R_03_STN_BS']
target_chans2 = ['LFP_R_10_STN_BS', 'LFP_R_11_STN_BS', 'LFP_R_12_STN_BS']
new_ch = raw.get_data(target_chans1).mean(axis=0) - raw.get_data(target_chans2).mean(axis=0)
u.add_new_channel(raw, new_ch[np.newaxis, :], "LFP_R_av_bipolar", type="dbs")

# Add bipolar re-reference for lfp channels
"""left_chans = [ch for ch in ch_names if "LFP_L" in ch]
right_chans = [ch for ch in ch_names if "LFP_R" in ch]
for i, chan in enumerate(left_chans):
    # Bipolar average
    new_chan = raw.get_data(picks=[chan]) - np.mean(raw.get_data(picks=left_chans), axis=0)
    av_bipolar_chan_name = f"{chan}_bipolar"
    u.add_new_channel(raw, new_chan, av_bipolar_chan_name, type="dbs")
    # Bipolar neighbouring
    if i == 0 or i<len(left_chans)-1:
        new_chan = raw.get_data(picks=chan) - raw.get_data(picks=left_chans[i+1])
        chan2 = ch_names[i+1]
    else:
        new_chan = raw.get_data(picks=chan) - raw.get_data(picks=left_chans[i - 1])
        chan2 = ch_names[i - 1]
    bipolar_chan_name = f"{chan}_{chan2}_bipolar"
    u.add_new_channel(raw, new_chan, bipolar_chan_name, type="ecog")
for i, chan in enumerate(right_chans):
    # Bipolar average
    new_chan = raw.get_data(picks=[chan]) - np.mean(raw.get_data(picks=left_chans), axis=0)
    av_bipolar_chan_name = f"{chan}_bipolar"
    u.add_new_channel(raw, new_chan, av_bipolar_chan_name, type="dbs")
    # Bipolar neighbouring
    if i == 0 or i<len(right_chans)-1:
        new_chan = raw.get_data(picks=chan) - raw.get_data(picks=right_chans[i+1])
        chan2 = ch_names[i+1]
    else:
        new_chan = raw.get_data(picks=chan) - raw.get_data(picks=right_chans[i - 1])
        chan2 = ch_names[i - 1]
    bipolar_chan_name = f"{chan}_{chan2}_bipolar"
    u.add_new_channel(raw, new_chan, bipolar_chan_name, type="ecog")"""
ch_names = raw.info["ch_names"]

# Find exact time point of stimulation onset (using the time stamps of the behavioral data)
# Extract events
events = mne.events_from_annotations(raw)[0]
event_id = 10004
tmin = -0.2
tmax = 0.5
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None)
epochs.load_data()
events_stim = []
# Loop trough epochs and get the sample shortly before aritfact start
for data_epoch, event in zip(epochs.get_data(), epochs.events):
    tmp = data_epoch.mean(axis=0).flatten()
    idx_stim_on = np.where(np.abs(zscore(tmp)) > 3)[0][0] - 20
    idx_stim_off = idx_stim_on + 1220
    """plt.plot(data_epoch.T)
    plt.axvline(idx_stim_on, color="red")
    plt.axvline(idx_stim_off, color="red")
    plt.show(block=True)"""
    # Save as events
    idx_event = event[0] + idx_stim_on + tmin*sfreq
    events_stim.append([idx_event, 0, 1])

"""plt.figure()
plt.plot(raw.get_data()[0,:].flatten())
plt.axvline(np.array(events_stim)[0, 0])
plt.show(block=True)"""

# add exact stim onset events to raw and gut into epochs based on those
events_stim = np.array(events_stim).astype(int)
mapping = {1: "stim onset"}
annot_from_events = mne.annotations_from_events(events=events_stim, event_desc=mapping, sfreq=sfreq)
raw.set_annotations(annot_from_events)
#raw.plot(block=True)

# Loop trough epoch and channel and remove artifact
tmax_PARRM = 0.305
tmin_PARRM = 0
fmin = 40
fmax = 100
#raw.pick(picks=[ch_names[-1]])
raw.filter(0.05, None, filter_length=500000)
raw_filt = raw.copy()
raw_filt.load_data()
epochs = mne.Epochs(raw.copy(), events=events_stim, event_id=1, tmin=tmin_PARRM, tmax=tmax_PARRM, baseline=None)
epochs_filt = epochs.copy()
epochs_filt.load_data()
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
for i, (epoch, epoch_idx) in enumerate(zip(epochs, epochs.events[:, 0])):
    for j, epoch_chan in enumerate(epoch):

        # Apply PARRM
        epoch_chan = epoch_chan[np.newaxis, :]
        parrm = PARRM(data=epoch_chan, sampling_freq=sfreq, artefact_freq=130, verbose=False)
        parrm.find_period()
        parrm.create_filter(period_half_width=0.04, filter_half_width=int((epoch_chan.shape[-1]-1)/2))
        filtered_epoch_chan = parrm.filter_data()

        # Replace in epoch and raw
        epochs_filt._data[i, j, :] = filtered_epoch_chan[0]
        raw_filt._data[j, int(epoch_idx+tmin_PARRM*sfreq)
                          :int(epoch_idx+tmax_PARRM*sfreq)+1] = filtered_epoch_chan[0]

        # Plot
        chan = epochs.ch_names[j]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].plot(epochs.times, epoch_chan[0])
        tfr = epochs.compute_tfr(method="multitaper", freqs=np.arange(fmin, fmax, 2), picks=[chan])
        tfr[i].plot(axes=axes[1, 0], show=False, tmin=0.1, tmax=tmax_PARRM, cmap="jet")
        axes[0, 1].plot(epochs.times, filtered_epoch_chan[0])
        tfr = epochs_filt.compute_tfr(method="multitaper", freqs=np.arange(fmin, fmax, 2), picks=[chan])
        tfr[i].plot(axes=axes[1, 1], show=False, tmin=0.1, tmax=tmax_PARRM, cmap="jet")
        plt.suptitle(f"Channel {chan} epoch {i}")
        plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.15)
        # Save figure
        plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_epoch_{i}_chan_{chan}.png",
                    format="png", bbox_inches="tight")
        plt.close()

        # Inspect raw object
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(raw.get_data(picks=[ch_names[j]]).flatten())
        ax.plot(raw_filt.get_data(picks=[ch_names[j]]).flatten())

        plt.show(block=True)


# Save filtered raw object
save_path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\{sub}.fif"
raw_filt.save(save_path, overwrite=True)



