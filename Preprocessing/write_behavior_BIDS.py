# Write data collected in matlab form to BIDS
# 1. Load correct data
# 2. Add informative channels
# 3. Save as tsv
# 4. Update JSON and save
# 5. Update session scans file

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
sub = "EL011"
med = "On"
run = "1"

# LOAD THE DATA__________________________________________________________________________________
# Get the behavioral matlab data from the source folder
file_path_mat = fd.askopenfilename(initialdir=f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'
                                              f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/'
                                              f'sourcedata/sub-{sub}/')
# Get name of file
for file in os.listdir("/".join(file_path_mat.split("/")[:-1])):
    if ".mat" in file:
        file_name_mat = file

# Load the MATLAB data
behav_data = loadmat(file_path_mat)

# Extract the behavioral data stored in a matrix
behav_data = behav_data["struct"][0][0][1]

print(np.unique(behav_data[:, 11]))

# Get the BIDS path for the corresponding folder
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"], extensions=".vhdr",
                                subjects=sub, runs=run,
                                acquisitions="StimOnB",
                                sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                          f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])

# If there is more than one recording, interrupt
if len(bids_path) > 1:
    input("More than one dataset exist for the specified BIDS path, please specify")
    bids_path = bids_path[0]
else:
    bids_path = bids_path[0]


# CREATE NEW FOLDER AND SAVE___________________________________________________________________________
# Create the behavioral data folder
beh_path = str(bids_path.root) + "\\sub-" + \
           str(bids_path.subject) + "\\ses-" + \
           str(bids_path.session) + "\\beh"
os.mkdir(beh_path)

# Convert array to data frame
df = pd.DataFrame(behav_data,
                  columns=["PEN_X", "PEN_Y", "TIME", "SPEED_MEAN", "SPEED",
                           "SPEED_X", "SPEED_Y", "BLOCK", "TRIAL", "TARGET",
                           "STIMULATION", "TARGET_X", "TARGET_Y", "TIME_H", "TIME_M", "TIME_S"])

# Get order of conditions
slow_first = 1 if file_name_mat.index("Slow") < file_name_mat.index("Fast") else 0

# Add column with condition
cond = ["Slow_Stim" if i in [1, 2] and slow_first or i in [3, 4] and not slow_first else "Fast_Stim" for i in behav_data[:, 7]]
df["Condition"] = cond

# Add column with block specification
block_spec = []
for i in behav_data[:, 7]:
    if i in [1, 3]:
        block_spec.append("Stimulation_Block")
    if i in [2, 4]:
        block_spec.append("Recovery_Block")
    if i == 0:
        block_spec.append("Test_Block")
df["Block_Specification"] = block_spec

# Save the data
data_path = beh_path + "\\" + bids_path.basename[:-24] + "beh.tsv"
df.to_csv(data_path, index_label=False, sep="\t")

# UPDATE AND SAVE JSON FILE_______________________________________________________________________
# Load the JSON file from the corresponding neurophys recording
f = open(str(bids_path.fpath)[:-4]+"json", )
meta_data = json.load(f)

# Create json file for behavior based on electrophysiological json file
meta_data_beh = {}
meta_data_beh["ElectricalStimulationParameters"] = meta_data["ElectricalStimulationParameters"]
meta_data_beh["InstitutionAddress"] = meta_data["InstitutionAddress"]
meta_data_beh["InstitutionName"] = meta_data["InstitutionName"]
meta_data_beh["Instructions"] = meta_data["Instructions"]
meta_data_beh["Hardware"] = "Wacom Cintiq 16"
meta_data_beh["Software"] = "Psychtoolbox Matlab"
meta_data_beh["TaskDescription"] = meta_data["TaskDescription"]
meta_data_beh["TaskName"] = meta_data["TaskName"]
meta_data_beh["Comments"] = "Stimulation lasted 300 ms only (not longer as indicated by the STIMULATION column)"

# Save updated json file
json_object = json.dumps(meta_data_beh, indent=4)
with open(data_path[:-3]+"json", 'w', encoding='utf8') as json_file:
    json_file.write(json_object)

# ADD BEHAVIOR TO SCANS FILE________________________________________________________________________________
    # Define name of behavioral data
    beh_name = "beh/" + bids_path.basename[:-24] + "beh.tsv"

    # Load tcv scans file
    scans_path = str(bids_path.root) + "\\sub-" + \
                 str(bids_path.subject) + "\\ses-" + \
                 str(bids_path.session) + f"\\sub-{sub}" + \
                 "_ses-" + str(bids_path.session) + "_scans.tsv"
    df = pd.read_table(scans_path)

    # Copy and add line with behavioral data
    for i in range(len(df)):
        if "Vigor" in df.iloc[i]["filename"]:
            new_recording = df.iloc[i].copy()
    new_recording["filename"] = beh_name
    df.loc[len(df)] = new_recording

    # Save
    df.to_csv(scans_path, index_label=False, sep="\t", index=False)

print("DONE")