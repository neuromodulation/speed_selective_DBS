# Add synchronized behavioral data to brain vision file
# 1. Save to backup
# 2. Add synchronized behavioral channels
# 3. Add events
# 4. Save to local update folder
# 5. Update JSON file
# 6. Load and test updated file
# 7. Replace in target location

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
import shutil
matplotlib.use('Qt5Agg')
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import mplcursors

# Define the subject and the medication state
sub = "EL012"
med = "Off"
run = "1"

# GET PATHS_______________________________________________________________________________________________
# Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
root = "C:\\Users\\ICN\\Documents\\VigorStim\\Neurophys\\Root"
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects=sub,
                                    runs=run,
                                    descriptions="neurophys",
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                              f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])

# If there is more than one recording, interrupt
if len(bids_path) > 1:
    input("More than one dataset exist for the specified BIDS path, please specify")
    bids_path = bids_path[0]
else:
    bids_path = bids_path[0]

# Get the path of the corresponding behavioral data saved in matlab
file_path_mat = fd.askopenfilename(initialdir=f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'
                                              f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/'
                                              f'sourcedata/sub-{sub}/')
#file_path_mat = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/sourcedata/sub-{sub}/ses-EcogLfpMedOff02/541IR68_MedOff02_VigorStimL_StimOn_1 - 20220510T103645/sub-10-MedOff-task-VigorStim-L-Fast-Slow-StimOn-run-01-behavioral.mat'
# Get name of file
for file in os.listdir("/".join(file_path_mat.split("/")[:-1])):
    if ".mat" in file:
        file_name_mat = file

# LOAD DATA__________________________________________________________________________________________
# Load neurophys data
raw_data = read_raw_bids(bids_path=bids_path)
raw_data.load_data()

# Load the behavioral data
behav_data = loadmat(file_path_mat, struct_as_record=True)
# Extract the behavioral data stored in a matrix
behav_data = behav_data["struct"][0][0][1]
# Determine the condition based on the filename
slow_first = 1 if file_name_mat.index("Slow") < file_name_mat.index("Fast") else 0


# BACKUP________________________________________________________________________________________________
# Save neurophys data to backup folder, no overwrite possible
backup_root = "C:\\Users\\ICN\\Documents\\VigorStim\\Neurophys\\Backup"
bids_backup_path = bids_path.copy().update(root=backup_root)
if not os.path.exists(bids_backup_path):
    mne_bids.write_raw_bids(raw_data, bids_path=bids_backup_path,
                            allow_preload=True, format="BrainVision",
                            verbose=False, overwrite=False)


# SYNCHRONIZE____________________________________________________________________________________________
raw_data_copy = raw_data.copy()


raw_data_tmp = raw_data.copy()
sfreq = raw_data.info["sfreq"]

# Get the times of the samples
time_array_neuro = raw_data.times.flatten()

# Determine stimulation onset based on LFP channels
# Throw away the bad channels
raw_data_tmp.drop_channels(raw_data_tmp.info["bads"])

# Get the first channel (LFP)
signal = raw_data_tmp._data[0, :]
signal_filt = raw_data_tmp.copy().filter(l_freq=1, h_freq=200).get_data()[0, :-1000]

# Find the first sample above a threshold
idx_onset_neuro_zscore = int(np.where(np.abs(zscore(signal_filt)) > 2.5)[0][0])

# Plot for visual inspection
fig = plt.figure()
scatter = plt.scatter(raw_data_tmp.times[idx_onset_neuro_zscore-1000:idx_onset_neuro_zscore+1000], signal[idx_onset_neuro_zscore-1000:idx_onset_neuro_zscore+1000])
selected_points = []
def on_click(event):
    if event.button == 1:  # Check for left mouse button click
        if scatter.contains(event)[0]:
            ind = scatter.contains(event)[1]["ind"][0]
            selected_points.append(ind)
            print(f"Selected point: {ind}")


fig.canvas.mpl_connect('button_press_event', on_click)
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Point {sel.target.index}'))
plt.show()

# Plot for visual inspection
idx_onset_neuro = selected_points[0] + idx_onset_neuro_zscore - 1000
# Substract 10 milliesonds that it takes to trigger the stimulation
idx_onset_neuro = int(idx_onset_neuro - (0.01*sfreq))
plt.figure()
plt.plot(signal)
plt.axvline(idx_onset_neuro, color="red")
plt.show()

# Find the first sample with stimulation in the behavioral data
behav_data_stim = behav_data[:, 10].flatten()
idx_onset_behav = np.where(behav_data_stim == 1)[0][0] - 1

# Get time in sec at which stimulation onsets occur
time_onset_neuro = time_array_neuro[idx_onset_neuro]
time_array_behav = behav_data[:, 2].flatten()
time_onset_behav = time_array_behav[idx_onset_behav]

# Substract the time difference from the neuro data (neuro recording always starts first)
diff_time = time_onset_neuro - time_onset_behav
time_array_neuro = time_array_neuro - diff_time

# Get indexes of stimulation onset in behav data and visually check the alignment
plt.figure()
idx_stim = np.where(np.diff(behav_data_stim) == 1)[0]
plt.plot(time_array_neuro[:len(signal)], signal)
for idx in idx_stim:
    plt.axvline(time_array_behav[idx], color="red")
plt.show()

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
ch_names_new = ["PEN_X", "PEN_Y", "SPEED_MEAN", "SPEED", "BLOCK", "TRIAL", "TARGET",
            "STIMULATION", "TARGET_X", "TARGET_Y", "STIM_CONDITION"]
info = mne.create_info(ch_names_new, raw_data.info['sfreq'], ["misc"]*len(ch_names_new))
behav_raw = mne.io.RawArray(behav_data_long.T, info)
if ch_names_new[0] in raw_data.info["ch_names"]:
    raw_data = raw_data.pick(raw_data.info["ch_names"][:-len(ch_names_new)])
raw_data.add_channels([behav_raw], force_update_info=True)

# Final plot for visual inspection
plt.figure()
plt.plot(raw_data.get_data(["STIMULATION"]).T)
plt.plot(raw_data.get_data(["TARGET"]).T)
plt.plot(u.norm_0_1(signal))
plt.show()


# ADD EVENTS______________________________________________________________________________________________
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
plt.plot(u.norm_0_1(signal))
for i_block in range(1, n_blocks + 1):
    for i_trial in range(1, n_trials + 1):
        mask = np.where(np.logical_and(blocks == i_block, trials == i_trial))[1]
        if not np.any(mask):
            print("STOOP")
        try:
            onset_idx.append(np.where(
                np.logical_and.reduce([blocks == i_block, trials == i_trial, speed > 300]))[1][0])
            plt.axvline(onset_idx[-1])
            offset_idx.append(np.where(
                np.logical_and.reduce([blocks == i_block, trials == i_trial, speed > 500]))[1][-1])
            plt.axvline(offset_idx[-1])
            mask = np.where(np.logical_and(blocks == i_block, trials == i_trial))[1]
            peak_speed_idx.append(mask[np.argmax(speed[:, mask])])
            plt.axvline(peak_speed_idx[-1])
            if np.any(stim[:, mask]):
                stim_idx.append(np.where(
                    np.logical_and.reduce([blocks == i_block, trials == i_trial, stim == 1]))[1][0])
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

# Plot the stimulation artifacts
artifacts = np.array([signal[int(x-sfreq*0.1):int(x+sfreq*0.5)] for x in stim_idx])
plt.figure()
plt.plot(artifacts.T)
plt.axvline(sfreq*0.1, linewidth=3)

# Last check
if not np.all(raw_data_copy.get_data()[:20,:] == raw_data.get_data()[:20,:]):
    x = input("Data is not matching, please check")

plt.show()

# SAVE TO LOCAL UPDATE FOLDER_______________________________________________________________________________
update_root = "C:\\Users\\ICN\\Documents\\VigorStim\\Neurophys\\Updated"
bids_update_path = bids_path.copy().update(root=update_root, description=None)
print("Writing out new file")
mne_bids.write_raw_bids(raw_data, bids_path=bids_update_path, allow_preload=True, format="BrainVision", verbose=False, overwrite=True)


# UPDATE JSON FILE__________________________________________________________________________________________
f = open(str(bids_path.fpath)[:-4]+"json", )
meta_data = json.load(f)

# save to backup folder
json_object = json.dumps(meta_data, indent=4)
with open(str(bids_backup_path.fpath)[:-4]+"json", 'w', encoding='utf8') as json_file:
    json_file.write(json_object)

# Update information
comment = input("Comments:")
meta_data["Comments"] = comment
meta_data["Condition"] = f"Condition 0=Slow Stim, 1=Fast Stim"
# Update task description
meta_data["TaskDescription"] = "Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. " \
                   "Start and stop events are visually cued on screen with a rest duration of 350 ms. 4 blocks with 96 movements each. " \
                   "In blocks 1 an 3 subthalamic deep brain stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. " \
                   "The order of slow/fast blocks is alternated between participants. Performed with the dominant hand."


# SAVE JSON TO UPDATE FOLDER_________________________________________________________________________________
json_object = json.dumps(meta_data, indent=4)
with open(str(bids_update_path.fpath)[:-4]+"json", 'w', encoding='utf8') as json_file:
    json_file.write(json_object)


# CHECK NEW BIDS DATA IN THE UPDATE FOLDER_________________________________________________________________
# Load updated data
raw_data_updated = read_raw_bids(bids_path=bids_update_path)
# Compare datasets
data_1 = raw_data_copy.get_data(picks=["dbs"])
data_2 = raw_data_updated.get_data(picks=["dbs"])
plt.plot(data_1.T, color="black")
plt.plot(data_2.T, color="red")
plt.show()

# Replace dataset
replace = input("Replace?[Y]")
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects=sub,
                                    runs=run,
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                              f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])
if len(bids_path) > 1:
    input("More than one dataset exist for the specified BIDS path, please specify")
    bids_path = bids_path[0]
else:
    bids_path = bids_path[0]

if replace == "Y":
    # Replace brainvision file
    mne_bids.copyfiles.copyfile_brainvision(bids_update_path.fpath, bids_path.fpath)
    # Replace JSON file
    shutil.copyfile(str(bids_update_path.fpath)[:-4]+"json", str(bids_path.fpath)[:-4]+"json")
    # Replace TSV file
    shutil.copyfile(str(bids_update_path.fpath)[:-9] + "channels.tsv", str(bids_path.fpath)[:-9] + "channels.tsv")
    # Add events file
    shutil.copyfile(str(bids_update_path.fpath)[:-9] + "events.tsv", str(bids_path.fpath)[:-9] + "events.tsv")

# FINAL CHECK______________________PLOT__________________________________
raw_data_check = read_raw_bids(bids_path=bids_path)
raw_data_check.plot(block=True)

print(f"Successfully synchronized behavioral and neurophysiological data")
