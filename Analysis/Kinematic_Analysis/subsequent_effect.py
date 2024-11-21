# Results Figure 2: Effect on subsequent speed
# Slow vs Fast

# Import useful libraries
import os
import sys
sys.path.insert(1, "../../../Code")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
from scipy.stats import ttest_1samp, permutation_test, wilcoxon
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

    # Loop over conditions
    for j in range(2):

        feature_stim_all = []
        feature_similar_all = []

        # Loop over the n subsequent movements
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
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
titles = ["Opposite direction", "Same direction"]
box_width = 0.4
fontsize = 7
for i in range(1, n):

    # Calculate the difference between stimulated and not stimulated movement
    res_tmp = res[:, :, :, 0] - res[:, :, :, 1]

    fig = plt.figure(figsize=(1, 1.65))
    bar_pos = [0 - (box_width / 1.5), 1 + (box_width / 1.5)]

    for l in range(2):
        bp = plt.boxplot(x=res_tmp[:, l, i],
                         positions=[bar_pos[l]],
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor=colors_op[l], color=colors_op[l]),
                         capprops=dict(color=colors_op[l]),
                         whiskerprops=dict(color=colors_op[l]),
                         medianprops=dict(color="indianred", linewidth=0),
                         flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=0,
                                         markeredgecolor='none')
                         )
        # Test difference from 0 (one-sample ttest)
        t, p = ttest_1samp(res_tmp[:, l, i], 0)
        print(p)
    # Add the individual lines
    for dat in res_tmp[:, :, i]:
        plt.plot(bar_pos[0], dat[0], marker='o', markersize=0.5, color=colors[0])
        plt.plot(bar_pos[1], dat[1], marker='o', markersize=0.5, color=colors[1])
        # Add line connecting the points
        plt.plot(bar_pos, dat, color="black", linewidth=0.5, alpha=0.3)

    # Add statistics
    #z, p = scipy.stats.wilcoxon(res_tmp[:, i, 0], res_tmp[:, i, 1])
    res_perm = permutation_test(data=(res_tmp[:, 0, i], res_tmp[:, 1, i]),
                                       statistic=u.diff_mean_statistic,
                                       n_resamples=10000, permutation_type="samples")
    p = res_perm.pvalue
    text = u.get_sig_text(p)
    ymin, ymax = plt.ylim()
    ymax +=2
    plt.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 1, ymax, ymax, ymax - 1], color="black",
            linewidth=1)
    plt.text(np.mean(bar_pos), ymax, text, ha="center", va="bottom", fontsize=fontsize)

    # Adjust plot
    plt.axhline(0, linewidth=1, color="black", linestyle="dashed")
    plt.yticks(fontsize=fontsize-2)
    plt.xticks([0, 1],["Slow", "Fast"], fontsize=fontsize)
    plt.yticks([-10, 10], [-10, 10], fontsize=fontsize)
    plt.ylim([-12, 15])
    feature_name_plot = feature_name.replace("_", " ")
    plt.ylabel(f"Stimulation-induced \nspeed shift [%]", fontsize=fontsize)
    u.despine(['right', 'top'])
    plt.title(titles[i-1], fontsize=fontsize)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{i}.pdf", format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{i}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()
