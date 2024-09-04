# Analyse which threshold to use for movement detection (in order to calculate average movement speed)

import numpy as np
import matplotlib
import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

n_samps = 18
med = "Off"
root = f"../../Data/{med}/raw_data/"
files_list = []
for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.mat'):
            files_list.append(os.path.join(root, file))

data_all = []
# Loop over all files in folder

plt.figure()
for file in files_list:

    count = 0

    # Load behavioral data
    data = loadmat(file)
    data = data["struct"][0][0][1]

    # Extract the feature for each trial
    n_trials = 96
    feature = np.zeros((2, 2, n_trials))

    for i_block in range(1, 5):

        for i_trial in range(1, n_trials+1):

            # Extract the raw speed while on the target
            mask = np.where((data[:, 7] == i_block) & (data[:, 8] == i_trial) & (data[:, 9] == 1))
            data_mask = np.squeeze(data[mask, 4])

            # Extend to one array
            data_all.extend(data_mask[-n_samps:])

data_all = np.array(data_all)
# PLot histogram
plt.figure(figsize=(10, 5))
plt.hist(data_all, bins=100, color="dimgrey")
# Compute zscore
threshold = 3
thres_max = np.mean(data_all) + threshold * np.std(data_all)
# Plot thresholds in histogram
plt.axvline(thres_max, color="red")
# Add text with values
plt.text(thres_max+100, 100, f"{thres_max:.2f}", color="red")
plt.title(f"Last {n_samps*16} ms on target (from 350 ms), z-score {threshold}")

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../Figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../../Figures/{plot_name}.png", format="png", transparent=True, bbox_inches="tight")

plt.show()
