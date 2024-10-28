# Extract feature from behavioral matlab data
# Store as matrix with 4 dimensions (patient, condition, block, trial) for later analysis

import numpy as np
import matplotlib
import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

med = "Off"
root = f"../../Data/{med}/raw_data/"
files_list = []
for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.mat'):
            files_list.append(os.path.join(root, file))

# Set the feature to be extracted
feature_name = "first_time_sample"

feature_all = []
# Loop over all files in folder
for file in files_list:

    count = 0

    # Load behavioral data
    data = loadmat(file)
    data = data["struct"][0][0][1]

    # Determine the condition based on the filename
    slow_first = 1 if file.index("Slow") < file.index("Fast") else 0

    # Extract the feature for each trial
    n_trials = 96
    feature = np.zeros((2, 2, n_trials))

    for i_block in range(1, 5):
        # 0=Stimulation, 1=Recovery
        block_type = 0 if i_block in [1, 3] else 1
        # 0=Slow, 1=Fast
        cond = 0 if i_block in [1, 2] and slow_first or i_block in [3, 4] and not slow_first else 1

        for i_trial in range(1, n_trials+1):
            # Get the data from one trial
            mask = np.where(np.logical_and(data[:,7] == i_block, data[:,8] == i_trial))
            data_mask = np.squeeze(data[mask, :])

            if feature_name == "peak_speed":
                feature[cond, block_type, i_trial - 1] = np.max(data[mask, 3])

            # CHOSEN OUTCOME MEASURE____________________________________________________________________________________
            elif feature_name == "mean_speed":
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 800.77
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                                np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]] + 1
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                                    np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]] + 1
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:offset_idx, 3])
                except:
                    count += 1
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:, 3])

                """plt.figure()
                plt.plot(data_mask[:, 3])
                plt.plot(data_mask[:, 4])
                plt.axvline(onset_idx)
                plt.axvline(offset_idx)
                plt.axhline(move_thres, color="red", linestyle= "--")
                plt.show()"""

            elif feature_name == "peak_speed_next":
                if i_trial < 96:
                    mask = np.where(np.logical_and(data[:, 7] == i_block, data[:, 8] == i_trial+1))
                    data_mask = np.squeeze(data[mask, :])
                    feature[cond, block_type, i_trial - 1] = np.max(data[mask, 3])
                else:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "first_time_sample":
                feature[cond, block_type, i_trial - 1] = data_mask[0, 2]
                print(data_mask[0, 2])

            elif feature_name == "peak_speed_original_block_order":
                cond = 0 if i_block in [1, 2] else 1
                feature[cond, block_type, i_trial - 1] = np.max(data[mask, 3])

            elif feature_name == "peak_acc":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    onset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                    feature[cond, block_type, i_trial - 1] = np.max(np.diff(data_mask[onset_idx:idx_peak_speed, 3]))
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "peak_dec":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 500
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                                np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]] + 1
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                                    np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]] + 1
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.min(np.diff(data_mask[idx_peak_speed:offset_idx, 3]))
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "mean_change":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                try:
                    onset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(np.abs(np.diff(data_mask[onset_idx:offset_idx, 3])))
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "mean_dec":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 800.77
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                                np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]] + 1
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                                    np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]] + 1
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(np.diff(data_mask[idx_peak_speed:offset_idx, 3]))
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
                    feature[cond, block_type, i_trial - 1] = np.percentile(np.diff(data_mask[idx_peak_speed:offset_idx, 3]), 50)
                except:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "stim":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    feature[cond, block_type, i_trial - 1] = 1

            elif feature_name == "stim_1":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    if i_trial < 96:
                        feature[cond, block_type, i_trial] = 1
            elif feature_name == "stim_2":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    if i_trial < 96-1:
                        feature[cond, block_type, i_trial+1] = 1
            elif feature_name == "stim_3":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    if i_trial < 96-2:
                        feature[cond, block_type, i_trial+2] = 1

            elif feature_name == "stim_time":
                idx_stim = np.where(data_mask[:, 10] == 1)[0]
                if len(idx_stim) > 0:
                    stim_time = data_mask[idx_stim[0], 2] - data_mask[0, 2]
                else:
                    stim_time = None
                feature[cond, block_type, i_trial - 1] = stim_time

            elif feature_name == "peak_speed_time":
                peak_idx = np.argmax(data_mask[:, 3])
                feature[cond, block_type, i_trial - 1] = data_mask[peak_idx, 2] - data_mask[0, 2]

            elif feature_name == "move_onset_time":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 3])
                # Get idx of movement onset (closest sample to peak below threshold)
                move_thres = 500
                onset_idx = np.where(data_mask[:, 3] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 3] < move_thres)) > 0)[1][-1]]
                if data_mask[onset_idx, 2] - data_mask[0, 2] < 2:
                    feature[cond, block_type, i_trial - 1] = data_mask[onset_idx, 2] - data_mask[0, 2]
                else:
                    feature[cond, block_type, i_trial - 1] = None

            elif feature_name == "move_offset_time":
                idx_peak_speed = np.argmax(data_mask[:, 3])
                move_thres = 500
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

            elif feature_name == "mean_speed_new":
                move_thres = 800
                onset_idx = np.where(data_mask[:, 4] > move_thres)[0][0]
                try:
                    onset_idx = np.where(data_mask[:, 4] > move_thres)[0][0]
                    offset_idx = len(data_mask) - np.where(np.flip(data_mask[:, 4]) > move_thres)[0][0] - 1
                    if offset_idx != onset_idx:
                        feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:offset_idx, 4])
                    else:
                        feature[cond, block_type, i_trial - 1] = None
                except:
                    count += 1
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:, 4])

            elif feature_name == "mean_speed_300":
                idx_peak_speed = np.argmax(data_mask[:, 4])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:offset_idx, 3])
                except:
                    count += 1
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:, 3])
                """plt.figure()
                plt.plot(data_mask[:, 3])
                plt.plot(data_mask[:, 4])
                plt.axvline(onset_idx)
                plt.axvline(offset_idx)
                plt.show()"""

            elif feature_name == "mean_speed_original_block_order":
                cond = 0 if i_block in [1, 2] else 1
                idx_peak_speed = np.argmax(data_mask[:, 4])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 800
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:offset_idx, 3])
                except:
                    count += 1
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:, 3])

            elif feature_name == "median_speed_raw":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 4])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 600
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.median(data_mask[onset_idx:offset_idx, 4])
                except:
                    count += 1
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:, 4])
                """plt.figure()
                plt.plot(data_mask[:, 3])
                plt.plot(data_mask[:, 4])
                plt.axvline(onset_idx)
                plt.axvline(offset_idx)
                plt.show()"""

            elif feature_name == "mean_speed_raw":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 4])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 300
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                    np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                try:
                    onset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                    offset_idx = np.where(data_mask[:, 4] < move_thres)[0][
                        np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) < 0)[1][0]]
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:offset_idx, 4])
                except:
                    count += 1
                    feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:, 4])

            elif feature_name == "median_speed":
                # Get index of peak speed
                idx_peak_speed = np.argmax(data_mask[:, 4])
                # Get idx of movement onset and offset (closest sample to peak below threshold)
                move_thres = 200
                onset_idx = np.where(data_mask[:, 4] < move_thres)[0][np.where((idx_peak_speed - np.where(data_mask[:, 4] < move_thres)) > 0)[1][-1]]
                on_target = np.where(data_mask[:, 9] == 1)[0][0]
                feature[cond, block_type, i_trial - 1] = np.mean(data_mask[onset_idx:on_target, 4])

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
                # Account for handedness
                right = "R" in file.split("-")
                if right:
                    feature[cond, block_type, i_trial - 1] = 1 if np.unique(data_mask[:, 11]) > 1000 else 0
                else:
                    feature[cond, block_type, i_trial - 1] = 1 if np.unique(data_mask[:, 11]) < 1000 else 0

            elif feature_name == "ITI_mean_speed":
                mask = np.where((data[:, 7] == i_block) & (data[:, 8] == i_trial) & (data[:, 9] == 1))
                data_mask = np.squeeze(data[mask, :])
                feature[cond, block_type, i_trial - 1] = np.mean(data_mask[data_mask[:, 2] > (data_mask[-1, 2] - 0.2), 3])

            elif feature_name == "ITI_peak_speed":
                mask = np.where((data[:, 7] == i_block) & (data[:, 8] == i_trial) & (data[:, 9] == 1))
                data_mask = np.squeeze(data[mask, :])
                feature[cond, block_type, i_trial - 1] = np.max(
                    data_mask[data_mask[:, 2] > (data_mask[-1, 2] - 0.2), 3])

    # Save the feature values for all datasest
    feature_all.append(feature)

    print(count)

feature_all = np.array(feature_all)

# Save matrix
np.save(f"../../Data/{med}/processed_data/{feature_name}.npy", feature_all)

# Save matrix as mat
savemat(f"../../Data/{med}/processed_data/{feature_name}.mat", {"feature" : feature_all})