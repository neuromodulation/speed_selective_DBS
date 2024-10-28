# Analyze and plot the instantaneous effect of the stimulation (next n movements)

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import pandas as pd
from scipy.io import savemat
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
feature_name = "mean_speed"
n = 3
method = "mean"

# Load matrix containing the feature values
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
# Load the stimulated trials
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
# Load the fast/slow movement
slow = np.load(f"../../../Data/{med}/processed_data/Slow.npy")
fast = np.load(f"../../../Data/{med}/processed_data/Fast.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)

# Loop over datasets
res = np.zeros((n_datasets, 2, n, 2))
for i in range(n_datasets):

    # Loop over the n next movements to look at for the instantaneous effect
    for j in range(2):

        feature_stim_all = []
        feature_similar_all = []

        # Loop over conditions
        for k in range(n):

            # Get the feature and stimulation for one patient
            feature_stim_tmp = feature[i, j, 0, :].flatten()
            stim_tmp = stim[i, j, 0, :].flatten()
            if med == "Healthy":
                if j == 0:
                    stim_tmp = slow[i, j, 0, :].flatten()
                else:
                    stim_tmp = fast[i, j, 0, :].flatten()

            # Get the index of the stimulated movements (same and other condition)
            stim_idx = np.where(stim_tmp == 1)[0]

            # Get the stimulated feature n after the current one
            stim_idx = stim_idx[stim_idx+k < len(feature_stim_tmp)]
            feature_stim_n = feature_stim_tmp[stim_idx + k]
            if k == 0:
                feature_stim_0 = feature_stim_n

            # Get fast/slow movements from recovery blocks (where they were not stimulated)
            feature_similar_tmp = feature[i, :, 1, :].flatten()
            if j == 0:
                slow_tmp = slow[i, :, 1, :].flatten()
                similar_idx = np.where(slow_tmp == 1)[0]
            else:
                fast_tmp = fast[i, :, 1, :].flatten()
                similar_idx = np.where(fast_tmp == 1)[0]
            similar_idx = similar_idx[similar_idx + k < len(feature_similar_tmp)]
            feature_similar_n = feature_similar_tmp[similar_idx + k]
            if k == 0:
                feature_similar_0 = feature_similar_n

            # Calculate the average feature
            tmp_stim = ((feature_stim_n - feature_stim_0[:len(feature_stim_n)])/ feature_stim_0[:len(feature_stim_n)]) * 100
            tmp_similar = ((feature_similar_n - feature_similar_0[:len(feature_similar_n)])/ feature_similar_0[:len(feature_similar_n)]) * 100
            if method == "mean":
                res[i, j, k, 0] = np.nanmean(tmp_stim)
                res[i, j, k, 1] = np.nanmean(tmp_similar)
            elif method == "median":
                res[i, j, k, 0] = np.nanmedian(tmp_stim)
                res[i, j, k, 1] = np.nanmedian(tmp_similar)

# Plot
# Prepare plotting
#res[:, :, 1, :] = np.mean([res[:, :, 1, :], res[:, :, 3, :], res[:, :, 5, :]], axis=0)
#res[:, :, 1, :] = np.mean([res[:, :, 1, :], res[:, :, 3, :]], axis=0)
#res[:, :, 3, :] = np.mean([res[:, :, 1, :], res[:, :, 3, :]], axis=0)
#res[:, :, 2, :] = np.mean([res[:, :, 2, :], res[:, :, 4, :], res[:, :, 6, :]], axis=0)
#res[:, :, 2, :] = np.mean([res[:, :, 2, :], res[:, :, 4, :]], axis=0)
fig = plt.figure(figsize=(5, 4.2))
colors_op = np.array([["#00863b", "#b2dac4"], ["#3b0086", "#b099ce"], ["#203D64", "#8f9eb1"]])
conditions = ["Slow", "Fast", "Difference \nFast-Slow"]
box_width = 0.3
bps = []
for i in range(3):
    plt.subplot(1, 3, i+1)
    if i == 2:
        # Plot the difference fast-slow for similar and stimulated movements
        res_tmp = res[:, 1, :, :] - res[:, 0, :, :]
    else:
        res_tmp = res[:, i, :, :]
    for j in range(1, n):
        bar_pos = [j - (box_width / 1.5), j + (box_width / 1.5)]
        for k in range(2):
            bp = plt.boxplot(x=res_tmp[:, j, k],
                             positions=[bar_pos[k]],
                             widths=box_width,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors_op[i, k], color=colors_op[i, k]),
                             capprops=dict(color=colors_op[i, k]),
                             whiskerprops=dict(color=colors_op[i, k]),
                             medianprops=dict(color="indianred", linewidth=0),
                             flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=5,
                                             markeredgecolor='none')
                             )
            bps.append(bp)
        # Add the individual lines
        for dat in res_tmp[:, j, :]:
            plt.plot(bar_pos[0], dat[0], marker='o', markersize=2, color="dimgray")
            plt.plot(bar_pos[1], dat[1], marker='o', markersize=2, color="dimgray")
            # Add line connecting the points
            plt.plot(bar_pos, dat, color="black", linewidth=0.5, alpha=0.3)

        # Add statistics
        z, p = scipy.stats.wilcoxon(res_tmp[:, j, 0], res_tmp[:, j, 1])
        """res_perm = scipy.stats.permutation_test(data=(res_tmp[:, j, 0], res_tmp[:, j, 1]),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=10000, permutation_type="samples")
        p = res_perm.pvalue"""
        if p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = "n.s."
        print(p)
        ymin, ymax = plt.ylim()
        plt.plot(bar_pos, [ymax, ymax], color="black", linewidth=1)
        plt.text(np.mean(bar_pos), ymax, text, ha="center", va="bottom", fontsize=16)

    # Adjust plot
    plt.xticks([j], [conditions[i]], fontsize=13)#ticks=np.arange(1, n), labels=[f"next {x}" for x in range(1, n)], fontsize=11)
    #plt.ylim([-30, 40])
    plt.axhline(0, linewidth=1, color="black", linestyle="dashed")
    plt.yticks(fontsize=12)
    feature_name_plot = feature_name.replace("_", " ")
    if i == 0:
        plt.ylabel(f"Change in speed [%]", fontsize=15)
        u.despine(['right', 'top'])
    elif i == 1:
        plt.yticks([])
        u.despine(['right', 'top', 'left'])
    else:
        plt.yticks([])
        u.despine(['top', 'left'])
    #plt.title(conditions[i], fontsize=13, y=1.1)

# Add legend
"""plt.legend([bps[-2]["boxes"][0], bps[-1]["boxes"][0]], ['Stimulated', 'Not \nstimulated'],
           bbox_to_anchor=(0.9, 0.6),
           prop={'size': 13})"""

plt.subplots_adjust(left=0.15, wspace=0.05)
#plt.suptitle(med, fontsize=14, y=1.1)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()

# Save difference in after second move
res_tmp = res[:, 1, 2, :] - res[:, 0, 2, :]
res = res_tmp[:, 0] - res_tmp[:, 1]
np.save(f"../../../Data/{med}/processed_data/res_inst_{feature_name}_{method}.npy", res)
# As mat file for imaging analysis
savemat(f"../../../Data/{med}/processed_data/res_inst_{feature_name}_{method}.mat", {"res": res})