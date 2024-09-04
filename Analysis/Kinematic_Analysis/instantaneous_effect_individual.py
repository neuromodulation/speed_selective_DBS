# Analyze whether the stimulation changes the ongoing movement (deceleration) for each patient individually

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

fontsize = 7

# Loop over the conditions
for i in range(6, 7):

    fig, ax = plt.subplots(1, 1, figsize=(0.8, 1.7))

    colors_fill = np.array([["#EDB7B7", "white"], ["#EDB7B7", "white"]])
    colors = np.array([["#00863b", "#00863b"], ["#3b0086", "#3b0086"]])
    labels = np.array([["Stimulated Slow", "Slow"], ["Stimulated Fast", "Fast"]])

    # Loop over conditions
    for j in range(2):

        # Get the stimulated and not stimulated feature
        feature_stim = feature[i, j, 0, :][stim[i, j, 0, :]]
        feature_no_stim = feature[i, :, :, :][similar[j][i, :, :, :]]
        feature_all = [feature_stim, feature_no_stim]

        # Plot average feature as boxplot with statistics
        box_width = 0.25
        bar_pos = [j - (box_width / 1.5), j + (box_width / 1.5)]

        for l in range(2):
            bp = ax.boxplot(x=feature_all[l],
                            zorder=1,
                        positions=[bar_pos[l]],
                        #label=labels[j, l],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(linestyle='-.', facecolor=colors_fill[j, l], color=colors[j, l]),
                        capprops=dict(color=colors[j, l]),
                        whiskerprops=dict(color=colors[j, l]),
                        medianprops=dict(color="dimgray", linewidth=1),
                        flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=0,
                                        markeredgecolor='none')
                        )
            # Add the points
            jitter = np.random.uniform(-0.05, 0.05, size=len(feature_all[l]))
            ax.scatter(np.repeat(bar_pos[l], len(feature_all[l]))+jitter, feature_all[l], s=0.5, c="dimgray", zorder=2)

        # Add statistics
        z, p = scipy.stats.ttest_ind(feature_stim, feature_no_stim)
        print(p)
        """res = scipy.stats.permutation_test(data=(feature_av[:, 0], feature_av[:, 1]),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=100000, permutation_type="samples")
        p = res.pvalue"""
        if p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = "n.s."
        ymax = -15
        ymin = -190
        ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 10, ymax, ymax, ymax - 10], color="black",
                linewidth=0.8)
        ax.text(j, ymax+10, text, fontsize=fontsize, horizontalalignment='center')

    # Adjust subplot
    ax.set_ylabel(f"Average deceleration", fontsize=fontsize)
    ax.set_xticks(ticks=[0, 1], labels=["Slow", "Fast"], fontsize=fontsize)
    ax.set_yticks([])
    u.despine()

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.pdf", format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=False)

    plt.show()