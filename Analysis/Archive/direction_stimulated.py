# Analyze if movements to the left/right were more/less often stimulated

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
med = "OFF"  # "on", "off", "all"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
fast = np.load(f"../../Data/{med}/processed_data/fast.npy")
slow = np.load(f"../../Data/{med}/processed_data/slow.npy")
stim = np.load(f"../../Data/{med}/processed_data/stim.npy")

# Select the dataset of interest
feature_matrix = feature_matrix[:, :, :, :]
stim = stim[:, :, :, :]
fast = fast[:, :, :, :]
slow = slow[:, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Delete outliers
np.apply_along_axis(lambda m: u.fill_outliers_nan(m, threshold=3), axis=3, arr=feature_matrix)
#np.apply_along_axis(lambda m: utils.fill_outliers_nan(m, threshold=3), axis=3, arr=feature_matrix)
# Compute mean/median feature "slow" and "fast" stimulated trials
feature_cond = np.zeros((n_dataset, 2))
feature_slow = []
feature_fast = []
for i in range(n_dataset):
    for cond in range(2):
        percentiles = [percentileofscore(feature_matrix[i, cond, :], x, nan_policy='omit')
                        for x in feature_matrix[i, cond, :][stim[i, cond, :] == 1]]
        feature_cond[i, cond] = np.nanmedian(percentiles)
        percentiles = np.array(percentiles)[~ np.isnan(percentiles)]
        if cond == 0:
            feature_slow.extend(percentiles)
        else:
            feature_fast.extend(percentiles)

# Plot the percentile of the stimulated trials
plt.figure(figsize=(5, 5))

color_slow = "#00863b"
color_fast = "#3b0086"

# Plot as violin
parts = plt.violinplot([np.array(feature_slow), np.array(feature_fast)], [1.25, 1.75],
                       showmeans=False, showextrema=False, showmedians=False)

# Set the color
parts['bodies'][0].set_facecolor(color_slow)
parts['bodies'][0].set_alpha(0.3)
parts['bodies'][1].set_facecolor(color_fast)
parts['bodies'][1].set_alpha(0.3)

# Plot the individual points
for dat in feature_cond:
    plt.plot(1.25, dat[0], marker='o', markersize=4, color=color_slow)
    plt.plot(1.75, dat[1], marker='o', markersize=4, color=color_fast)
    # Add line connecting the points
    plt.plot([1.25, 1.75], dat, color="grey", linewidth=0.7, alpha=0.5)

# Add statistics
z, p = scipy.stats.wilcoxon(x=feature_cond[:, 0], y=feature_cond[:, 1])
sig = "bold" if p < 0.05 else "regular"
plt.text(1.25, 107, f"p = {np.round(p, 5)}", weight=sig, fontsize=14)
plt.plot([1.25, 1.75], [105, 105], color="black", linewidth=1.7, alpha=1)
plt.plot([1.25, 1.25], [103, 105], color="black", linewidth=1.7, alpha=1)
plt.plot([1.75, 1.75], [103, 105], color="black", linewidth=1.7, alpha=1)
plt.yticks(fontsize=14)

plt.ylabel(f"Percentile of {feature_name} \nof stimulated movements", fontsize=16)

# Adjust plot
plt.xticks(fontsize=14)
#plt.xlim([0.5, 2.5])
plt.xticks([1.25, 1.75], ["Slow", "Fast"], fontsize=16)
plt.subplots_adjust(left=0.25)
u.despine()

plt.savefig(f"../../Figures/stim_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/stim_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()