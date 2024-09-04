# Script for statistical analysis of feature
# Correlate the raw peak speed and the time (quantification of bradykinesia)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
from scipy.stats import spearmanr
import seaborn as sb
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# Set analysis parameters
feature_name = "peak_speed"
med = "OFF"  # "on", "off", "all"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
feature_matrix = feature_matrix[1:, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_matrix)

# Reshape matrix such that blocks from one condition are concatenated
feature_matrix = np.reshape(feature_matrix, (n_datasets, 2, n_trials*2))

# Delete the first 5 movements
feature_matrix = feature_matrix[:, :, 5:]

# Normalize to average of first 5 movements
#feature_matrix = u.norm_perc(feature_matrix)

# Compute the correlation for every patient and condition
n_trials = 90
colors = ["#00863b", "#3b0086"]
x = np.arange(n_trials)
corrs = np.zeros((n_datasets, 2))
for dataset in range(n_datasets):
    plt.subplot(4, 5, dataset+1)
    for cond in range(2):
        corrs[dataset, cond] = spearmanr(feature_matrix[dataset, cond, :n_trials], x[:n_trials], nan_policy='omit')[0]
        sb.regplot(x=x[:n_trials], y=feature_matrix[dataset, cond, :n_trials], color=colors[cond])
plt.show()


# Plot the average correlation for each subject during each condition (fast vs slow)
fig = plt.figure(figsize=(5.6, 5.6))
color_slow = "#00863b"
color_fast = "#3b0086"
bar_pos_slow = 0.75
bar_pos_fast = 1.25
plt.bar(bar_pos_slow, np.mean(corrs[:, 0]), color=color_slow, label="Slow", width=0.5, alpha=0.5)
plt.bar(bar_pos_fast, np.mean(corrs[:, 1]), color=color_fast, label="Fast", width=0.5, alpha=0.5)

# Plot the individual points
for corr in corrs:
    plt.plot(bar_pos_slow, corr[0], marker='o', markersize=3, color=color_slow)
    plt.plot(bar_pos_fast, corr[1], marker='o', markersize=3, color=color_fast)
    # Add line connecting the points
    plt.plot([bar_pos_slow, bar_pos_fast], corr, color="black", linewidth=0.7, alpha=0.5)

# Add statistics
z, p = scipy.stats.wilcoxon(x=corrs[:, 0], y=corrs[:, 1])
sig = "bold" if p < 0.05 else "regular"
plt.text(bar_pos_slow, np.max(corrs), f"p = {np.round(p, 3)}", weight=sig, fontsize=18)
plt.yticks(fontsize=16)

# Adjust plot
feature_name_space = feature_name.replace("_", " ")
plt.ylabel(f"Correlation {feature_name_space} vs trials", fontsize=20)
plt.subplots_adjust(bottom=0.2, left=0.17)
plt.legend(loc="best",  prop={'size': 16})
u.despine()
axes = plt.gca()
axes.spines[['right', 'top']].set_visible(False)

# Save figure on group basis
plt.savefig(f"../../Figures/corr_time_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)

plt.show()