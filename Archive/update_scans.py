# TMP file to update scans files with behavioral vigor stim data

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
subs = ["L011"]
med = "On"

for sub in subs:

    # Get the BIDS path for the corresponding folder
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/rawdata/'
    file_path_raw = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"], extensions=".vhdr", descriptions="neurophys", subjects=sub, sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01", f"LfpMed{med}02", f"EcogLfpMed{med}02"])
    file_path_raw = file_path_raw[1]

    # Define name of behavioral data
    beh_name = "beh/" + file_path_raw.basename[:-24] + "beh.tsv"

    # Load tcv scans file
    scans_path = str(file_path_raw.root) + "\\sub-" + str(file_path_raw.subject) + "\\ses-" + str(file_path_raw.session) + f"\\sub-{sub}" + "_ses-" + str(file_path_raw.session) + "_scans.tsv"
    df = pd.read_table(scans_path)

    # Copy and add line with behavioral data
    new_recording = df.iloc[-1].copy()
    new_recording["filename"] = beh_name
    df.loc[len(df)] = new_recording

    # Save
    df.to_csv(scans_path, index_label=False, sep="\t", index=False)

    print("DONE")