# Test the synchronized datasets
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

# Specify the medication group
med = "Off"

# Read the list of the datasets
df = pd.read_excel(f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'
       f'Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/'
       f'Tablet_task/Data/Dataset_list.xlsx', sheet_name=med)

# Loop through subjects
for i, sub in enumerate(df["ID Berlin_Neurophys"][:4]):

    # Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
           f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
           f'rawdata/'
    bids_paths = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                        extensions=".vhdr",
                                        subjects=sub,
                                        sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                            f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])
    if len(bids_paths) > 0:
        bids_path = bids_paths[0]
        # Read the raw data
        raw_data = read_raw_bids(bids_path=bids_path)

        # Load the behavioral matlab data
        if i < 9:
             root = f"../../Data/{med}/raw_data/0{i+1}/"
        else:
            root = f"../../Data/{med}/raw_data/{i+1}/"
        for file in os.listdir(root):
            if ".mat" in file:
                data_mat = loadmat(root+file)
                data_mat = data_mat["struct"][0][0][1]
                break

        # Load the behavioral data stored as tsv in BIDS
        root = str(bids_path.root) + "\\sub-" + \
                   str(bids_path.subject) + "\\ses-" + \
                   str(bids_path.session) + "\\beh\\"
        for file in os.listdir(root):
            if ".tsv" in file:
                data_beh = pd.read_csv(root+file, delimiter="\t")

        # Compare matlab and beh data
        data_beh = np.array(data_beh)[:, :-2]
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(data_mat[:, 3].flatten(), color="red")
        plt.plot(data_beh[:, 3].flatten(), color="black")

        # Get peak speed from all trials and compare
        speed = raw_data.get_data(["SPEED_MEAN"]).flatten()
        blocks = raw_data.get_data(["BLOCK"])
        trials = raw_data.get_data(["TRIAL"])
        onset_idx = []
        offset_idx = []
        peak_speed = np.zeros((3, 4, 96))
        for i_block in range(1, 5):
            for i_trial in range(1, 97):

                # Raw data
                mask = np.where(np.logical_and(blocks == i_block, trials == i_trial))[1]
                if len(mask) > 0:
                    peak_speed[0, i_block-1, i_trial-1] = np.max(speed[mask])
                else:
                    peak_speed[0, i_block - 1, i_trial - 1] = None
                    print(f"Trial {i_trial} in block {i_block} not present")

                # Matlab
                mask = np.where(np.logical_and(data_mat[:, 7] == i_block, data_mat[:, 8] == i_trial))[0]
                peak_speed[1, i_block - 1, i_trial - 1] = np.max(data_mat[mask, 3])

                # Beh
                mask = np.where(np.logical_and(data_beh[:, 7] == i_block, data_beh[:, 8] == i_trial))[0]
                peak_speed[2, i_block - 1, i_trial - 1] = np.max(data_beh[mask, 3])

        # Compare
        plt.subplot(1, 2, 2)
        plt.suptitle(sub)
        plt.plot(peak_speed[0, :, :].flatten(), color="red")
        plt.plot(peak_speed[1, :, :].flatten(), color="black")
        plt.plot(peak_speed[2, :, :].flatten(), color="green")
plt.show()

print("lets see")

