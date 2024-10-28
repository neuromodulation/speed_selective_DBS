# Extract feature from behavioral data saved as a dataframe

import numpy as np
import matplotlib
import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

# Set the edication condition
med = "On"

# Set the feature to be extracted
feature_name = "peak_speed"

# Read the list of the datasets
df = pd.read_excel(f'../../Data/Dataset_list.xlsx', sheet_name=med)

# Set the root folder
root_data = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'

# Loop over subjects
feature_all = []
# Extract the peak for each trial
n_trials = 96
n_patients = len(df["ID Berlin_Neurophys"])
feature_all = np.zeros((n_patients, 2, 2, n_trials))

for i, sub in enumerate(df["ID Berlin_Neurophys"]):

    # Load behavioral data
    for root, dirs, files in os.walk(root_data + "\sub-" + sub):
        for file in files:
            if "beh" in os.path.join(root, file) and file.endswith("tsv") and f"Med{med}" in file:
                filepath = os.path.join(root, file)
                break
    print(filepath)
    data = pd.read_csv(filepath, delimiter="\t")

    for i_block in range(1, 5):
        for i_trial in range(1, n_trials+1):
            data_trial = data[(data["BLOCK"] == i_block) & (data["TRIAL"] == i_trial)]
            cond = 0 if np.all(data_trial["Condition"] == "Slow_Stim") else 1
            block_type = 0 if np.all(data_trial["Block_Specification"] == "Stimulation_Block") else 1

            # Peak speed
            if feature_name == "peak_speed":
                feature_all[i, cond, block_type, i_trial - 1] = np.max(data_trial["SPEED_MEAN"])
                #feature[cond, block_type, i_trial - 1] = np.percentile(data[mask, 3], 95)

            """
            elif feature_name == "peak_acc":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_trial[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    onset_idx = np.where(data_trial[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                    feature[cond, block_type, i_trial - 1] = np.percentile(np.diff(data_mask[onset_idx:idx_peak_speed, 3]), 95)
                    #feature[cond, block_type, i_trial - 1] = np.max(np.diff(data_mask[onset_idx:idx_peak_speed, 3]))
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "peak_dec":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    #feature[cond, block_type, i_trial - 1] = np.min(np.diff(data_mask[idx_peak_speed:offset_idx, 3]))
                    feature[cond, block_type, i_trial - 1] = np.percentile(np.diff(data_mask[idx_peak_speed:offset_idx, 3]), 5)
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "median_dec":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    #feature[cond, block_type, i_trial - 1] = np.median(np.diff(data_mask[idx_peak_speed:offset_idx, 3]))
                    feature[cond, block_type, i_trial - 1] = np.percentile(np.diff(data_mask[idx_peak_speed:offset_idx, 3]), 10)
                except:
                    feature[cond, block_type, i_trial - 1] = None
                plt.plot(data_mask[:,3])
                plt.plot(data_mask[:, 4])
                plt.plot(np.diff(data_mask[:, 3]))
                plt.axvline(onset_idx)
                plt.axvline(idx_peak_speed)
                plt.show()

            elif feature_name == "stim":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    feature[cond, block_type, i_trial - 1] = 1

            elif feature_name == "stim_time":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    stim_time = data_mask[idx_stim[0], 2] - data_mask[0, 2]
                else:
                    stim_time = None
                feature[cond, block_type, i_trial - 1] = stim_time

            elif feature_name == "speed_peak_time":
                peak_idx = np.argmax(data_mask[:, 3])
                feature[cond, block_type, i_trial - 1] = data_mask[peak_idx, 2] - data_mask[0, 2]

            elif feature_name == "move_onset_time":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset (closest sample to peak below threshold)
                move_thres = 300
                onset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                feature[cond, block_type, i_trial - 1] = data_mask[onset_idx, 2] - data_mask[0, 2]

            elif feature_name == "move_offset_time":
                idx_peak_speed = np.argmax(data_mask[:, 3])
                move_thres = 300
                try:
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = data_mask[offset_idx, 2] - data_mask[0, 2]
                except:
                    feature[cond, block_type, i_trial - 1] = data_mask[-1, 2] - data_mask[0, 2]

            elif feature_name == "move_dur":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    onset_idx = np.where(data_mask[:, 3] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = data_mask[offset_idx, 2] - data_mask[onset_idx, 2]
                except:
                    feature[cond, block_type, i_trial - 1] = data_mask[-1, 2] - data_mask[onset_idx, 2]

            elif feature_name == "move_dur_start":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres_2 = 500
                move_thres_1 = 100
                try:
                    onset_idx = np.where(data_mask[:, 3] < move_thres_1)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres_1)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 3] < move_thres_2)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres_2)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = data_mask[offset_idx, 2] - data_mask[onset_idx, 2]
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "mean_speed":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    onset_idx = np.where(data_mask[:, 3] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:offset_idx, 3])
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "median_acc":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                onset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                try:
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.median(np.diff(data_mask[onset_idx:offset_idx, 3]))
                except:
                    feature[cond, block_type, i_trial - 1] = np.median(np.diff(data_mask[onset_idx:, 3]))
            # Get index of all slow movements (peak speed slower than the last two movements)

            elif feature_name == "fast" and i_trial > 3:
                # Get peak speed from current trials and two previous trials
                peak_speed = np.max(data[mask, 3])
                peak_speed_prev_1 = np.max(data[np.where(np.logical_and(data[:, 7] == i_block, data[:, 8] == i_trial - 1)), 3])
                peak_speed_prev_2 = np.max(data[np.where(np.logical_and(data[:, 7] == i_block, data[:, 8] == i_trial - 2)), 3])
                if peak_speed > peak_speed_prev_1 and peak_speed > peak_speed_prev_2:
                    feature[cond, block_type, i_trial - 1] = 1

            elif feature_name == "slow" and i_trial > 3:
                # Get peak speed from current trials and two previous trials
                peak_speed = np.max(data[mask, 3])
                peak_speed_prev_1 = np.max(data[np.where(np.logical_and(data[:, 7] == i_block, data[:, 8] == i_trial - 1)), 3])
                peak_speed_prev_2 = np.max(data[np.where(np.logical_and(data[:, 7] == i_block, data[:, 8] == i_trial - 2)), 3])
                if peak_speed < peak_speed_prev_1 and peak_speed < peak_speed_prev_2:
                    feature[cond, block_type, i_trial - 1] = 1

            elif feature_name == "time_to_peak":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset (closest sample to peak below threshold)
                move_thres = 300
                onset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                feature[cond, block_type, i_trial - 1] = data_mask[idx_peak_speed, 2] - data_mask[onset_idx, 2]

            elif feature_name == "motor_range":
                feature[cond, block_type, i_trial - 1] = np.percentile(data_mask[:, 3], 95) - np.percentile(data_mask[:, 3], 5)

            elif feature_name == "time_to_offset":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement offset (closest sample to peak below threshold)
                move_thres = 100
                try:
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][-1]]
                    feature[cond, block_type, i_trial - 1] = data_mask[offset_idx, 2] - data_mask[idx_peak_speed, 2]
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "time_peak_target":
                idx_target = np.where(data_mask[:, 9] == 1)[0]
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                if len(idx_target) > 0:
                    feature[cond, block_type, i_trial - 1] = data_mask[idx_target[0], 2] - data_mask[idx_peak_speed, 2]
                else:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "trial_side":
                feature[cond, block_type, i_trial - 1] = 1 if np.unique(data_mask[:, 11]) > 1000 else 0"""

# Save matrix
np.save(f"../../Data/{med}/processed_data/{feature_name}_test.npy", feature_all)

# Test
plt.figure()
x = np.load(f"../../Data/{med}/processed_data/{feature_name}_test.npy")
y = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
for i in range(len(x)):
    plt.subplot(5, 5, i+1)
    plt.plot(x[i, :, :, :].flatten())
    plt.plot(y[i, :, :, :].flatten())
    plt.title(df["ID Berlin_Neurophys"][i])
    plt.ylim([1000, 6000])
plt.show()

print("gg")

