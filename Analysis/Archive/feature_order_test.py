# Script for investigating the order effect

import numpy as np
import matplotlib.pyplot as plt
import mne
import gc
import scipy.stats
from statsmodels.stats.diagnostic import lilliefors
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
from alive_progress import alive_bar
import time
from statannot import add_stat_annotation
import seaborn as sb
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# Set analysis parameters
feature_name = "peak_speed" # out of ["peak_acc", "mean_speed", "move_dur", "peak_speed", "stim_time", "peak_speed_time", "move_onset_time", "move_offset_time"]
med = "Off"

# Read the list of the datasets
df = pd.read_excel(f'../../Data/Dataset_list.xlsx', sheet_name=med)

# extract order
order = df["Block order"]
slow_first = order == "Slow"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature_matrix.shape

# Choose only the stimulation period
feature_matrix = feature_matrix[:, :, 0, :]

# Reshape matrix such that blocks from one condition are concatenated
feature_matrix = np.reshape(feature_matrix, (n_datasets, 2, n_trials))

# Delete the first 5 movements
feature_matrix = feature_matrix[:, :, 5:]

# Normalize to average of first 5 movements
feature_matrix = u.norm_perc(feature_matrix)

# Compute the effect in the first half of the stimulation period (difference fast-slow)
median_effect = np.nanmedian(feature_matrix[:, 1, :90], axis=1) - np.nanmedian(
    feature_matrix[:, 0, :90], axis=1)
idx_slow_fast = np.array(np.where(slow_first == 1)).flatten()[1:]
idx_fast_slow = np.array(np.where(slow_first == 0)).flatten()
y = np.concatenate((median_effect[idx_slow_fast], median_effect[idx_fast_slow]))
# Compare the difference between Slow/Fast and Fast/Slow

# Visualize
plt.figure()
my_pal = {"Fast/Slow": "green", "Slow/Fast": "grey"}
my_pal_trans = {"Fast/Slow": "lightgreen",  "Slow/Fast": "lightgrey"}
x = np.concatenate((np.repeat("Slow/Fast", len(idx_slow_fast)), np.repeat("Fast/Slow", len(idx_fast_slow))))
box = sb.boxplot(x=x, y=y, showfliers=False, palette=my_pal_trans)
sb.stripplot(x=x, y=y, ax=box, palette=my_pal)

# Add statistics
add_stat_annotation(box, x=x, y=y,
                    box_pairs=[("Slow/Fast", "Fast/Slow")],
                    test='t-test_welch', text_format='simple', loc='inside', verbose=2)

# Add labels
feature_name_space = feature_name.replace("_", " ")
plt.ylabel(f"Difference Fast-Slow of change in {feature_name_space} [%]", fontsize=12)
plt.xticks(fontsize=14)

# Save figure
plt.savefig(f"../../Plots/order_{feature_name}_normalize_{normalize}.svg", format="svg", bbox_inches="tight")

plt.show()

# Between subjects design: Compare blocks in which slow and fast were the first block
feature_matrix_slow = np.nanmedian(feature_matrix[idx_slow_fast, 0, :], axis=1)
feature_matrix_fast = np.nanmedian(feature_matrix[idx_fast_slow, 1, :], axis=1)
plt.plot(feature_matrix_slow.flatten())
plt.plot(feature_matrix_fast.flatten())
plt.axhline(0)
print("End")