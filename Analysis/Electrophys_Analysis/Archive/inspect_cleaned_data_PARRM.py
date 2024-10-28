# Inspect cleaned raw object

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import zscore
from pyparrm import PARRM
from mne_bids import read_raw_bids, find_matching_paths
import matplotlib
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
matplotlib.use('Qt5Agg')

# Set the dataset
sub = "EL008"
med = "Off"

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects=sub,
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                              f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])
raw = read_raw_bids(bids_path=bids_path[0])
raw.drop_channels(raw.info["bads"])
raw.pick(picks=["ecog"])
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]

# Apply bipolar re-reference TODO
raw.load_data()
for i, chan in enumerate(ch_names):
    # Bipolar average
    new_chan = raw.get_data(picks="ecog")[i, :] - np.mean(raw.get_data(picks="ecog"), axis=0)
    u.add_new_channel(raw, new_chan[np.newaxis, :], f"{chan}_bipolar", type="ecog")
    # Bipolar neighbouring
    if i == 0 and i<len(ch_names)-1:
        new_chan = raw.get_data(picks="ecog")[i, :] - raw.get_data(picks="ecog")[i+1, :]
        chan2 = ch_names[i+1]
    else:
        new_chan = raw.get_data(picks="ecog")[i, :] - raw.get_data(picks="ecog")[i-1, :]
        chan2 = ch_names[i - 1]
    u.add_new_channel(raw, new_chan[np.newaxis, :], f"{chan}_{chan2}_bipolar", type="ecog")
ch_names = raw.info["ch_names"]

# Load cleaned data
clean_path = f"C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor" \
             f"\\vigor_stim_task\\Data\\Off\\Neurophys\\Artifact_removal\\{sub}.fif"
raw_clean = mne.io.read_raw_fif(clean_path)

# Plot for inspection
fig, ax = plt.subplots(1, 1, figsize=(10,10))
#ax.plot(raw.get_data(picks=[ch_names[-1]]).flatten())
ax.plot(raw_clean.get_data(picks=[ch_names[-1]]).flatten())
plt.show(block=True)
