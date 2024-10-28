# Analyze whether the stimulation changes the ongoing movement (deceleration

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib


# Set parameters
feature_name = "mean_dec"
med = "Off"

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy").astype(bool)
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)

# Load index of similar movements
similar_slow = np.load(f"../../../Data/{med}/processed_data/Slow_similar.npy").astype(bool)
similar_fast = np.load(f"../../../Data/{med}/processed_data/Fast_similar.npy").astype(bool)
similar = [similar_slow, similar_fast]

# Loop over the conditions
fig, ax= plt.subplots(1, 1, figsize=(5.5, 3.5))
colors_op = np.array([["#00863b", "#b2dac4"], ["#3b0086", "#b099ce"], ["#203D64", "#8f9eb1"]])
labels = np.array([["Stimulated Slow", "Slow"], ["Stimulated Fast", "Fast"]])
for i in range(2):

    # Get the stimulated and not stimulated feature
    feature_mean_stim = np.array([np.nanmean(feature[x, i, 0, :][stim[x, i, 0, :]]) for x in range(n_datasets)])
    feature_mean_no_stim = np.array([np.nanmean(feature[x, :, :, :][similar[i][x, :, :, :]]) for x in range(n_datasets)])
    feature_mean = np.vstack([feature_mean_stim, feature_mean_no_stim]).T

    # Plot average feature as boxplot with statistics
    box_width = 0.3
    bar_pos = [i - (box_width / 1.5), i + (box_width / 1.5)]
    # Loop over conditions
    for j in range(2):
        bp = ax.boxplot(x=feature_mean[:, j],
                    positions=[bar_pos[j]],
                    label=labels[i, j],
                    widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors_op[i, j], color=colors_op[i, j]),
                    capprops=dict(color=colors_op[i, j]),
                    whiskerprops=dict(color=colors_op[i, j]),
                    medianprops=dict(color="indianred", linewidth=1),
                    flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=5,
                                    markeredgecolor='none')
                    )

    # Add the individual lines
    for dat in feature_mean:
        ax.plot(bar_pos[0], dat[0], marker='o', markersize=2, color="dimgray")
        ax.plot(bar_pos[1], dat[1], marker='o', markersize=2, color="dimgray")
        # Add line connecting the points
        ax.plot(bar_pos, dat, color="black", linewidth=0.5, alpha=0.3)

    # Add statistics
    z, p = scipy.stats.ttest_ind(feature_mean[:, 0], feature_mean[:, 1])
    print(p)
    """res = scipy.stats.permutation_test(data=(feature_av[:, 0], feature_av[:, 1]),
                                       statistic=u.diff_mean_statistic,
                                       n_resamples=100000, permutation_type="samples")
    p = res.pvalue"""
    sig = "bold" if p < 0.05 else "regular"
    ax.text(i-box_width, np.nanpercentile(feature_mean, 99), f"p = {np.round(p, 3)}", weight=sig, fontsize=14)

# Adjust subplot
feature_name_plot = feature_name.replace("_", " ")
ax.set_ylabel(f"{feature_name_plot}", fontsize=14)
ax.set_xticks(ticks=[0, 1], labels=["Slow", "Fast"], fontsize=14)
ax.legend()
u.despine()

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.15, top=0.8, wspace=0.01)
plt.suptitle("Stimulated vs not stimulated", fontsize=16, y=0.95)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()