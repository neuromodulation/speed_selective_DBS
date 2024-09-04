# Script for plotting the difference between the conditions

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
def statistic(x, y):
    return np.mean(x) - np.mean(y)

# Set analysis parameters
feature_name = "peak_speed"
med = "Off"  # "on", "off"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
feature_matrix = feature_matrix[1:-1, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_matrix)

# Reshape matrix such that blocks from one condition are concatenated
feature_matrix = np.reshape(feature_matrix, (n_datasets, 2, n_trials*2))

# Delete the first 5 movements
feature_matrix = feature_matrix[:, :, 5:]

# Normalize to average of first 5 movements
feature_matrix = u.norm_perc(feature_matrix)

# Compute mean difference between blocks
diff_effect = np.nanmean(feature_matrix[:, 1, :91], axis=1) - np.nanmean(feature_matrix[:, 0, :91], axis=1)

# Plot the average feature for each subject during each condition (stimulation and recovery)
fig = plt.figure(figsize=(6.5, 5.2))
color = "#233668"
plt.subplot(2, 1, 1)
plt.axhline(0, linewidth=2, color="#CC0000")
for i, diff in enumerate(diff_effect):
   plt.bar(i, diff, color=color, label="Slow", width=0.75, alpha=1)
plt.xticks([])
plt.yticks(fontsize=16)
plt.title("Stimulation", fontsize=20)
axes = plt.gca()
axes.spines[['right', 'top', 'bottom']].set_visible(False)

# Compute mean difference between blocks (recovery)
diff_effect = np.nanmean(feature_matrix[:, 1, 91:], axis=1) - np.nanmean(feature_matrix[:, 0, 91:], axis=1)
plt.subplot(2, 1, 2)
plt.axhline(0, linewidth=2, color="#CC0000")
for i, diff in enumerate(diff_effect):
   plt.bar(i, diff, color=color, label="Slow", width=0.75, alpha=1)
plt.xlabel("Participants", fontsize=20)
plt.xticks([])
plt.yticks(fontsize=16)
plt.title("Recovery", fontsize=20)
axes = plt.gca()
axes.spines[['right', 'top']].set_visible(False)

fig.supylabel("Difference in average peak speed\nFast – Slow [%]", fontsize=20, horizontalalignment='center')
plt.subplots_adjust(hspace=0.3, left=0.15)

# Save figure on group basis
plt.savefig(f"../../Figures/diff_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/diff_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()