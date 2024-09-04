# Add synchronized behavioral data to brain vision file

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
from mne_bids import BIDSPath, read_raw_bids
matplotlib.use('Qt5Agg')
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# Define the dataset of interest
med = "Off"
ID = "EL006"

# BEHAVIOR__________________________________________________________________________________________________________
# Get the behavioral matlab data from the source folder
source_path = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation ' \
              f'- BIDS_01_Berlin_Neurophys/sourcedata/sub-{ID}/'
for root, dirs, files in os.walk(source_path):
    for file in files:
        x = os.path.exists(os.path.join(root, file))
        if not x:
            print(file)
        if "Vigor" in file and med in file and (file.endswith(".mat")):
            c = "C:\\Users\\ICN\\Downloads\\" + file
            v = "C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/sourcedata/sub-EL006/ses-EphysMedOff02/523ON64_MedOff_StimOn_VigorStimR_01 - 20210831T104103/"
            os.path.exists(c)
            file_path_mat = os.path.join(root, file)
            break


# create a root window
root = tk.Tk()
root.withdraw()

# open the file dialog box
file_path = filedialog.askopenfile()

# print the selected file path
print(file_path)

# Load the MATLAB data
behav_data = loadmat(file_path_mat)
# Extract the behavioral data stored in a matrix
behav_data = behav_data["struct"][0][0][1]
# Determine the condition based on the filename
slow_first = 1 if file_path_mat.index("Slow") < file_path_mat.index("Fast") else 0

# ELECTROPHYSIOLOGY__________________________________________________________________________________________________________
# Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
raw_path = f"C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation " \
              f"- BIDS_01_Berlin_Neurophys\\rawdata\\"
bids_path = BIDSPath(root=raw_path, subject=ID, run="01")
# Incorporate medication status
# Make invariant to R,L,B after Vigorstim
# Change that also when saving
# Clean up the dataset 
raw_data = mne.io.read_raw_brainvision(file_path_bv, preload=True)
raw_data.plot()



# Downsample the neuro data to 500 Hz
new_sfreq = 500
raw_data.resample(new_sfreq)

# Get the times of the samples
time_array_neuro = raw_data.times.flatten()

# Determine stimulation onset based on LFP channels

# Filter the data
raw_data_filt = raw_data.copy().filter(l_freq=2, h_freq=200)
# Average first 10 channels (cut out the last 100 samples because of end artifact)
data_mean = np.mean(raw_data_filt._data[:10,:-100], axis=0)

# Plot for visual inspection
plt.figure()
plt.plot(data_mean)

# Find the first sample above a threshold
idx_onset_neuro = np.where(np.abs(zscore(data_mean)) > 2.5)[0][0]

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
plt.plot(time_array_neuro[:len(data_mean)], data_mean)
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
info = mne.create_info(ch_names, raw_data.info['sfreq'], ["bio"]*len(ch_names))
behav_raw = mne.io.RawArray(behav_data_long.T, info)
raw_data.add_channels([behav_raw], force_update_info=True)

# Final plot for visual inspection
plt.figure()
plt.plot(raw_data.get_data(["STIMULATION"]).T)
plt.plot(raw_data.get_data(["TARGET"]).T)
plt.plot(u.norm_0_1(data_mean))



# Save new brain vision file
filename_new = path_subject + "neurophys_behavior.vhdr"
#mne.export.export_raw(fname=filename_new, raw=raw_data, fmt="brainvision", overwrite=True)

print(f"Successfully synchronized behavioral and neurophysiological data")

plt.show()
plt.close()