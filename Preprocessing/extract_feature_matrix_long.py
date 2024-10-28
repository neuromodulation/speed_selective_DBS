# Extract feature array for each trial
# Store as matrix with 5 dimensions (patient, condition, block, trial, sample) for later analysis

import numpy as np
import matplotlib
import os
from scipy.io import loadmat, savemat
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

med = "Off"
root = f"../../Data/{med}/raw_data/"
files_list = []
for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.mat'):
            files_list.append(os.path.join(root, file))

# Set the feature name
feature_name = "speed_during_stim"
n_samples = 18

feature_all = []
# Loop over all files in folder
for file in files_list:

    # Load behavioral data
    data = loadmat(file)
    data = data["struct"][0][0][1]

    # Determine the condition based on the filename
    slow_first = 1 if file.index("Slow") < file.index("Fast") else 0

    # Extract a feature array of a defined length for each trial
    n_trials = 96
    feature = np.zeros((2, 2, n_trials, n_samples+1))

    for i_block in range(1, 5):
        block_type = 0 if i_block in [1, 3] else 1
        cond = 0 if i_block in [1, 2] and slow_first or i_block in [3, 4] and not slow_first else 1

        for i_trial in range(1, n_trials+1):
            mask = np.where(np.logical_and(data[:,7] == i_block, data[:,8] == i_trial))
            data_mask = np.squeeze(data[mask, :])

            # Speed around peak (centered around peak)
            if feature_name == "speed_around_peak":
                n_samples_half = int(n_samples/2)
                peak_idx = mask[0][np.argmax(data_mask[:, 3])]
                if len(data) > peak_idx + n_samples_half + 1 and peak_idx > n_samples_half:
                    feature[cond, block_type, i_trial - 1, :] = data[peak_idx-n_samples_half:peak_idx+n_samples_half+1, 4]
                else:
                    n_samples_available = np.min([peak_idx, len(data) - peak_idx])
                    feature[cond, block_type, i_trial - 1, :] = np.zeros(n_samples_half*2+1) * np.nan
                    #feature[cond, block_type, i_trial-1, n_samples_half-n_samples_available:n_samples_half+n_samples_available-1] = data[peak_idx-n_samples_available:peak_idx+n_samples_available+1, 3]

            # Speed 3 samples after peak comparable to stim
            if feature_name == "speed_after_peak":
                peak_idx = mask[0][np.argmax(data_mask[:, 3])]
                if len(data) > peak_idx + 3 + n_samples:
                    feature[cond, block_type, i_trial - 1, :] = data[peak_idx + 3:peak_idx + n_samples + 4, 3]
                else:
                    feature[cond, block_type, i_trial - 1, :] = np.zeros(n_samples + 1) * np.nan
                    # feature[cond, block_type, i_trial-1, n_samples_half-n_samples_available:n_samples_half+n_samples_available-1] = data[peak_idx-n_samples_available:peak_idx+n_samples_available+1, 3]

            # Speed during stimulation (aligned to stimulation onset)
            if feature_name == "speed_during_stim":
                stim_idx = np.where(data_mask[:, 10] == 1)[0]
                if len(stim_idx) > 0 and len(data) > (stim_idx[0] + n_samples+1):
                    stim_idx = mask[0][stim_idx[0]]
                    feature[cond, block_type, i_trial - 1, :] = data[stim_idx:stim_idx+n_samples+1, 3]
                else:
                    feature[cond, block_type, i_trial - 1, :] = np.zeros(n_samples+1) * np.nan

    # Save the feature values for all datasest
    feature_all.append(feature)

feature_all = np.array(feature_all)

# Save matrix
np.save(f"../../Data/{med}/processed_data/{feature_name}.npy", feature_all)

# Save matrix as mat
savemat(f"../../Data/{med}/processed_data/{feature_name}.mat", {"feature" : feature_all})
