# Check synchronized data and replace in target location

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

# Define the subject and the medication state
sub = "L014"
med = "Off"
run = "1"

# Quality check
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/rawdata/'
path_origin = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                  extensions=".vhdr", descriptions="neurophys",
                                  runs="1",
                                  subjects=sub, sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01", f"LfpMed{med}02", f"EcogLfpMed{med}02"])
raw_data_check = read_raw_bids(bids_path=path_origin[0])
raw_data_check.plot()

print(f"Successfully replaced the data, congrats")

