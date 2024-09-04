import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

for i in range(1, 25):
    try:
        # Load the data
        if i < 10:
            sub = f"EL00{i}"
        else:
            sub = f"EL0{i}"
        med = "On"
        root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
               f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
               f'rawdata/'
        bids_path = find_matching_paths(root, tasks=["BlockRotationR", "BlockRotationL"],
                                            extensions=".vhdr",
                                            subjects=sub,
                                            sessions=[f"LfpMed{med}01", f"LfpMed{med}02", f"EcogLfpMed{med}01", f"EcogLfpMed{med}02"])

        # Load dataset
        raw = read_raw_bids(bids_path=bids_path[0])
        raw.load_data()
        try:
            rotameter = raw.get_data(picks="ANALOG_R_ROTA_CH")
        except:
            rotameter = raw.get_data(picks="ANALOG_L_ROTA_CH")
        plt.figure()
        plt.title(i)
        plt.plot(rotameter.flatten())
    except:
        pass
plt.show()
