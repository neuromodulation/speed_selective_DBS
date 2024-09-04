# Plot feature of stimulated trials

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import percentileofscore
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
def statistic(x, y):
    return np.mean(x) - np.mean(y)

# Set analysis parameters
feature_name = "peak_speed"
med = "OFF"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
feature_matrix = feature_matrix[:, :, 0, :]
n_dataset, _, n_trials = feature_matrix.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m, threshold=3), axis=2, arr=feature_matrix)

# Load matrix indicating which trial was stimulated
stim = np.load(f"../../Data/{med}/processed_data/stim.npy")
stim = stim[:, :, 0, :]

# Delete the first 5 movements
feature_matrix = feature_matrix[:, :, 5:]
stim = stim[:, :, 5:]

# Plot feature "slow" and "fast" stimulated trials
plt.figure(figsize=(10, 5))
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
for i in range(n_dataset):
    for cond in range(2):
        percentiles = [percentileofscore(feature_matrix[i, cond, :], x, nan_policy='omit')
                        for x in feature_matrix[i, cond, :][stim[i, cond, :] == 1]]
        # Plot as thin bar
        plt.boxplot(percentiles, positions=[i+(0.3*cond)], patch_artist=True,
           boxprops=dict(facecolor=colors_op[cond], color=colors_op[cond]),
                    capprops=dict(color=colors_op[cond]),
                    whiskerprops=dict(color=colors_op[cond]),
                    medianprops=dict(color=colors[cond], linewidth=3),
                    flierprops=dict(marker='o', markerfacecolor=colors_op[cond], markersize=5, markeredgecolor='none')
                    )

# Adjust plot
plt.axhline(66, color="black", linewidth=1)
#plt.text(1, 67, "66 %")
plt.axhline(34, color="black", linewidth=1)
#plt.text(1, 35, "34 %")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(np.arange(n_dataset)+0.15, labels=np.arange(n_dataset)+1)
plt.ylabel(f"Percentile of {feature_name} \nof stimulated movements", fontsize=16)
plt.xlabel("Patient Number", fontsize=16)
u.despine()

# Save
plt.savefig(f"../../Figures/stim_{feature_name}_{med}_detail.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/stim_{feature_name}_{med}_detail.png", format="png", bbox_inches="tight", transparent=True)

plt.show()