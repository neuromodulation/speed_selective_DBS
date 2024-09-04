# Write data collected in matlab form to BIDS
# Store in neurophys folder

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
subs = ["EL007"]
med = "On"

for sub in subs:

    # Get the BIDS path for the corresponding folder
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/rawdata/'
    file_path_raw = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"], extensions=".vhdr", descriptions="neurophys", subjects=sub, sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01", f"LfpMed{med}02", f"EcogLfpMed{med}02"])
    file_path_raw = file_path_raw[1]

    # Load the JSON file from the corresponding neurophys recording
    f = open(str(file_path_raw.fpath)[:-4]+"json", )
    meta_data = json.load(f)

    # Add the condition
    meta_data["Condition"] = f"Condition 0=Slow Stim, 1=Fast Stim"

    # Update task description
    task_description = "Performance of diagonal forearm movements with a cursor on a screen using a digitizing tablet. " \
                       "Start and stop events are visually cued on screen with a rest duration of 350 ms. 4 blocks with 96 movements each. " \
                       "In blocks 1 an 3 subthalamic deep brain stimulation is applied for 300 ms if a movement is slower/faster than the previous two movements. " \
                       "The order of slow/fast blocks is alternated between participants. Performed with the dominant hand."
    meta_data["TaskDescription"] = task_description
    # Save updated json file
    json_object = json.dumps(meta_data, indent=4)
    with open(str(file_path_raw.fpath)[:-4]+"json", 'w', encoding='utf8') as json_file:
        json_file.write(json_object)

    print("DONE")