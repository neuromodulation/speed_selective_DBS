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
sub = "L011"
med = "On"

# Get the behavioral matlab data from the source folder
file_dir_mat = fd.askdirectory(initialdir=f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/sourcedata/sub-{sub}/')
for file in os.listdir(file_dir_mat):
    if ".mat" in file:
        file_name = file
file_path_mat = fd.askopenfilename(initialdir=f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/sourcedata/sub-{sub}/')

# Load the MATLAB data
behav_data = loadmat(file_path_mat)

# Extract the behavioral data stored in a matrix
behav_data = behav_data["struct"][0][0][1]

print(np.unique(behav_data[:, 11]))

# Get the BIDS path for the corresponding folder
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/rawdata/'
file_path_raw = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"], extensions=".vhdr", descriptions="neurophys", subjects=sub, sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01", f"LfpMed{med}02", f"EcogLfpMed{med}02"])
file_path_raw = file_path_raw[0]

# Create the behavioral data folder
beh_path = str(file_path_raw.root) + "\\sub-" + str(file_path_raw.subject) + "\\ses-" + str(file_path_raw.session) + "\\beh"
os.mkdir(beh_path)

# Convert array to data frame
df = pd.DataFrame(behav_data,
                  columns=["PEN_X", "PEN_Y", "TIME", "SPEED_MEAN", "SPEED", "SPEED_X", "SPEED_Y", "BLOCK", "TRIAL", "TARGET", "STIMULATION", "TARGET_X", "TARGET_Y", "TIME_H", "TIME_M", "TIME_S"])

# Get order of conditions
slow_first = 1 if file_name.index("Slow") < file_name.index("Fast") else 0

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
data_path = beh_path + "\\" + file_path_raw.basename[:-24] + "beh.tsv"
df.to_csv(data_path, index_label=False, sep="\t")

# Load the JSON file from the corresponding neurophys recording
f = open(str(file_path_raw.fpath)[:-4]+"json", )
meta_data = json.load(f)

# Save updated json file
json_object = json.dumps(meta_data, indent=4)
with open(str(file_path_raw.fpath)[:-4]+"json", 'w', encoding='utf8') as json_file:
    json_file.write(json_object)

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

# Add to scans file


print("DONE")