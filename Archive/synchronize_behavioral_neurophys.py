# Add synchronized behavioral data to brain vision file
# Save to backup and updated folder

import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat, savemat
from scipy.stats import zscore
import numpy as np
import pandas as pd
import os
import json
import mne_bids
import matplotlib
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths, write_raw_bids
from tkinter import filedialog as fd
matplotlib.use('Qt5Agg')
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# Define the subject and the medication state
sub = "EL007"
med = "On"

# Get the behavioral matlab data from the source folder
#file_path_mat = fd.askopenfilename(initialdir=f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/sourcedata/sub-{sub}/')

# Load the MATLAB data
#behav_data = loadmat(file_path_mat)

# Extract the behavioral data stored in a matrix
#behav_data = behav_data["struct"][0][0][1]

# Determine the condition based on the filename
#slow_first = 1 if file_path_mat.index("Slow") < file_path_mat.index("Fast") else 0
#slow_first = 1

# Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/rawdata/'
file_path_raw = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"], extensions=".vhdr", descriptions="neurophys", subjects=sub, sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01", f"LfpMed{med}02", f"EcogLfpMed{med}02"])

# If there is more than one recording, ask
file_path_raw = file_path_raw[0]

# Load
raw_data = read_raw_bids(bids_path=file_path_raw)

# Save to backup folder
backup_root = "C:\\Users\\ICN\\Documents\\VigorStim\\Neurophys\\Backup"
bids_backup_path = file_path_raw.copy().update(root=backup_root)
mne_bids.write_raw_bids(raw_data, bids_path=bids_backup_path, allow_preload=True, format="BrainVision", verbose=False, overwrite=False)

# Downsample the neuro data to 500 Hz
new_sfreq = 500
raw_data.resample(new_sfreq)
raw_data_copy = raw_data.copy()

# Get the times of the samples
time_array_neuro = raw_data.times.flatten()

# Determine stimulation onset based on LFP channels
# Filter the data
raw_data_tmp = raw_data.copy().filter(l_freq=2, h_freq=200)

# Throw away the bad channels
raw_data_tmp.drop_channels(raw_data_tmp.info["bads"])

# Get remaining LFP channels
raw_dbs = raw_data_tmp.pick_types(dbs=True)

# Average to determine onset of stimulation (remove edge artifact)
mean_signal = np.mean(raw_dbs._data[:, :-1000], axis=0)

# Plot for visual inspection
plt.figure()
plt.plot(mean_signal)

# Find the first sample above a threshold
idx_onset_neuro = np.where(np.abs(zscore(mean_signal)) > 2.5)[0][0]

# Plot for visual inspection
plt.axvline(idx_onset_neuro, color="red")

# Find the first sample with stimulation in the behavioral data
behav_data_stim = behav_data[:, 10].flatten()
idx_onset_behav = np.where(behav_data_stim == 1)[0][0]

# Get time in sec at which stimulation onsets occur
time_onset_neuro = time_array_neuro[idx_onset_neuro]
time_array_behav = behav_data[:, 2].flatten()
time_onset_behav = time_array_behav[idx_onset_behav]

# Substract the time difference from the neuro data (neuro recording alsways starts first)
diff_time = time_onset_neuro - time_onset_behav
time_array_neuro = time_array_neuro - diff_time

# Get indexes of stimulation onset in behav data and visually check the alignment
plt.figure()
idx_stim = np.where(np.diff(behav_data_stim) == 1)[0]
plt.plot(time_array_neuro[:len(mean_signal)], mean_signal)
for idx in idx_stim:
    plt.axvline(time_array_behav[idx], color="red")

# For every sample in the neuro data find the closest sample in the behav data
n_cols = np.size(behav_data, 1)
behav_data_long = np.zeros((len(time_array_neuro), n_cols))
for i, time_samp in enumerate(time_array_neuro):
    # If neuro sample is before or after onset of behav recording, save zeros
    if time_samp < 0 or time_samp > np.max(time_array_behav):
        behav_data_long[i, :] = np.zeros(16)
    else:
        # Get the sample that is closest in time
        idx_samp_behav = np.argmin(np.abs(time_array_behav - time_samp))
        behav_data_long[i,:] = behav_data[idx_samp_behav,:]


# Add a channel containing the condition
cond = [0 if i in [1, 2] and slow_first or i in [3, 4] and not slow_first else 1 for i in behav_data_long[:, 7]]
behav_data_long = np.hstack((behav_data_long, np.array(cond)[:, np.newaxis]))

# Select channels that should be saved
behav_data_long = behav_data_long[:, [0,1,3,4,7,8,9,10,11,12,16]]

# Add behavioral channels to the raw mne object
ch_names = ["PEN_X", "PEN_Y", "SPEED_MEAN", "SPEED", "BLOCK", "TRIAL", "TARGET", "STIMULATION", "TARGET_X", "TARGET_Y", "STIM_CONDITION"]
info = mne.create_info(ch_names, raw_data.info['sfreq'], ["misc"]*len(ch_names))
behav_raw = mne.io.RawArray(behav_data_long.T, info)
raw_data.add_channels([behav_raw], force_update_info=True)

# Final plot for visual inspection
plt.figure()
plt.plot(raw_data.get_data(["STIMULATION"]).T)
plt.plot(raw_data.get_data(["TARGET"]).T)
plt.plot(u.norm_0_1(mean_signal))
plt.show()

# Extract events (Movement Onset, Offset, Peak speed and stimulation onset)
speed = raw_data.get_data(["SPEED_MEAN"])
stim = raw_data.get_data(["STIMULATION"])
blocks = raw_data.get_data(["BLOCK"])
trials = raw_data.get_data(["TRIAL"])
n_trials = int(np.max(trials))
n_blocks = int(np.max(blocks))
onset_idx = []
offset_idx = []
peak_speed_idx = []
stim_idx = []
plt.figure()
plt.plot(raw_data.get_data(["SPEED"]).T)
plt.plot(raw_data.get_data(["SPEED_MEAN"]).T)
plt.plot(u.norm_0_1(mean_signal))
for i_block in range(1, n_blocks + 1):
    for i_trial in range(1, n_trials + 1):
        mask = np.where(np.logical_and(blocks == i_block, trials == i_trial))[1]
        if not np.any(mask):
            print("STOOP")
        try:
            onset_idx.append(np.where(np.logical_and.reduce([blocks == i_block, trials == i_trial, speed > 250]))[1][0])
            plt.axvline(onset_idx[-1])
            offset_idx.append(np.where(np.logical_and.reduce([blocks == i_block, trials == i_trial, speed > 500]))[1][-1])
            plt.axvline(offset_idx[-1])
            mask = np.where(np.logical_and(blocks == i_block, trials == i_trial))[1]
            peak_speed_idx.append(mask[np.argmax(speed[:, mask])])
            plt.axvline(peak_speed_idx[-1])
            if np.any(stim[:, mask]):
                stim_idx.append(np.where(np.logical_and.reduce([blocks == i_block, trials == i_trial, stim == 1]))[1][0])
                plt.axvline(stim_idx[-1])
        except:
            print(f"Trial missing {i_trial} in block {i_block}")
            pass

# Add events to raw data object
events = np.vstack((np.hstack((onset_idx, peak_speed_idx, offset_idx, stim_idx)),
                    np.zeros(len(onset_idx)+len(offset_idx)+len(peak_speed_idx)+len(stim_idx)),
                   np.hstack((np.ones(len(onset_idx)), np.ones(len(peak_speed_idx))*2, np.ones(len(offset_idx))*3, np.ones(len(stim_idx))*4)))).T
mapping = {
    1: "Movement Start",
    2: "Peak Speed",
    3: "Movement End",
    4: "Stimulation onset"}
annot_from_events = mne.annotations_from_events(events=events, event_desc=mapping, sfreq=raw_data.info["sfreq"])
raw_data.set_annotations(annot_from_events)

# Last check
print(np.all(raw_data_copy.get_data()[:20,:] == raw_data.get_data()[:20,:]))

# Save to updated folder
update_root = "C:\\Users\\ICN\\Documents\\VigorStim\\Neurophys\\Updated"
bids_update_path = file_path_raw.copy().update(root=update_root)
mne_bids.write_raw_bids(raw_data, bids_path=bids_update_path, allow_preload=True, format="BrainVision", verbose=False, overwrite=True)

# Update JSON file
f = open(str(file_path_raw.fpath)[:-4]+"json", )
meta_data = json.load(f)

# Save for backup
json_object = json.dumps(meta_data, indent=4)
with open(str(bids_backup_path.fpath)[:-4]+"json", 'w', encoding='utf8') as json_file:
    json_file.write(json_object)

# Update information
meta_data["SamplingFrequency"] = 500
comment = input("Comments:")
meta_data["Comments"] = comment
# Save in the Updated folder
json_object = json.dumps(meta_data, indent=4)
with open(str(bids_update_path.fpath)[:-4]+"json", 'w', encoding='utf8') as json_file:
    json_file.write(json_object)

print(f"Successfully synchronized behavioral and neurophysiological data")

plt.show()
plt.close()
