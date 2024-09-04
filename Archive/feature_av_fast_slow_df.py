# Script for statsitical analysis of feature

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
med = "OFF"  # "on", "off"

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

# Plot the average feature for each subject during each condition (stimulation and recovery)
fig = plt.figure(figsize=(5.6, 5.6))
color_slow = "#00863b"
color_fast = "#3b0086"
bar_pos = [1, 2]
for i in range(1, 3):

    # Median over all movements in that period
    feature_matrix_mean = np.nanmean(feature_matrix[:, :, int(91*(i-1)):int(91*i)], axis=2)

    # Plot the mean bar
    if i == 1:
        plt.bar(bar_pos[i-1]-0.25, np.mean(feature_matrix_mean[:, 0]), color=color_slow, label="Slow", width=0.5, alpha=0.5)
        plt.bar(bar_pos[i-1]+0.25, np.mean(feature_matrix_mean[:, 1]), color=color_fast, label="Fast", width=0.5, alpha=0.5)
    else:
        plt.bar(bar_pos[i - 1] - 0.25, np.mean(feature_matrix_mean[:, 0]), color=color_slow, width=0.5, alpha=0.5)
        plt.bar(bar_pos[i - 1] + 0.25, np.mean(feature_matrix_mean[:, 1]), color=color_fast, width=0.5, alpha=0.5)

    # Plot the individual points
    for dat in feature_matrix_mean:
        plt.plot(bar_pos[i-1]-0.25, dat[0], marker='o', markersize=3, color=color_slow)
        plt.plot(bar_pos[i-1] + 0.25, dat[1], marker='o', markersize=3, color=color_fast)
        # Add line connecting the points
        plt.plot([bar_pos[i-1]-0.25, bar_pos[i-1]+0.25], dat, color="black", linewidth=0.7, alpha=0.5)

    # Add statistics
    z, p = scipy.stats.ttest_rel(feature_matrix_mean[:, 0], feature_matrix_mean[:, 1])
    print(p)
    res = scipy.stats.permutation_test(data=(feature_matrix_mean[:, 0], feature_matrix_mean[:, 1]), statistic=statistic, permutation_type="samples")
    sig = "bold" if res.pvalue < 0.05 else "regular"
    plt.text(bar_pos[i-1]-0.25, np.max(feature_matrix_mean)+5, f"p = {np.round(res.pvalue, 3)}", weight=sig, fontsize=18)
    plt.yticks(fontsize=16)

# Adjust plot
plt.xticks(bar_pos, ["Stimulation", "Recovery"], fontsize=20)
feature_name_space = feature_name.replace("_", " ")
plt.ylabel(f"Change in {feature_name_space} [%]", fontsize=20)
plt.subplots_adjust(bottom=0.2, left=0.17)
plt.legend(loc="best",  prop={'size': 16})
u.despine()
axes = plt.gca()
axes.spines[['right', 'top']].set_visible(False)

# Save figure on group basis
plt.savefig(f"../../Figures/stats_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/stats_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()