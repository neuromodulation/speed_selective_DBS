# Analyze and plot the instantaneous effect of the stimulation (next n movements)

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parmeters
med = "Off"
feature_name = "mean_speed_raw"
n = 3
method = "mean"
norm = True

# Load matrix containing the feature values
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
# Load peak speed matrix (because stimulated trials were chosen based on the peak speed)
peak_speed = np.load(f"../../../Data/{med}/processed_data/peak_speed.npy")
# Load the stimulated trials
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
# Load the fast/slow movement
slow = np.load(f"../../../Data/{med}/processed_data/Slow.npy")
fast = np.load(f"../../../Data/{med}/processed_data/Fast.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)
np.apply_along_axis(lambda m: u.fill_outliers_nan_peak_speed(m), axis=3, arr=peak_speed)

# Loop over datasets
res = np.zeros((n_datasets, 2, n, 2))
for i in range(n_datasets):

    # Get the feature and stimulation for one patient
    feature_recov_tmp = feature[i, :, 1, :].flatten()
    peak_speed_recov_tmp = peak_speed[i, :, 1, :].flatten()

    # Loop over the n next movements to look at for the instantaneous effect
    for j in range(2):

        feature_stim_all = []
        feature_similar_all = []

        # Loop over conditions
        for k in range(n):

            # Get the feature and stimulation for one patient
            feature_stim_tmp = feature[i, j, 0, :].flatten()
            peak_speed_stim_tmp = peak_speed[i, j, 0, :].flatten()
            stim_tmp = stim[i, j, 0, :].flatten()

            # Get the index of the stimulated movements (same and other condition)
            stim_idx = np.where(stim_tmp == 1)[0]

            # Get the stimulated feature n after the current one
            stim_idx = stim_idx[stim_idx+k < len(feature_stim_tmp)]
            feature_stim_n = feature_stim_tmp[stim_idx + k]
            feature_stim_all.append(feature_stim_n)

            # Get array of the same size with comparable movements
            feature_similar_n = np.zeros(stim_idx.shape)
            for l, feature_samp in enumerate(feature_stim_tmp[stim_idx]):

                # Get the difference between the stimulated movement and all other movements
                tmp_idx = np.abs(feature_recov_tmp - feature_samp).argsort()

                # Loop over the difference and take the smallest one
                for m in tmp_idx:

                    # Check that movement was not already selected
                    if feature_samp not in feature_similar_n:

                        # Check whether the movement was slow/fast in comparison to previous movements (using the peak speed)
                        if (j == 0 and m > 2 and np.all(peak_speed_recov_tmp[m] < peak_speed_recov_tmp[m - 2:m])) or\
                                (j == 1 and m > 2 and np.all(peak_speed_recov_tmp[m] > peak_speed_recov_tmp[m - 2:m])):
                       # if True:

                            # Get the similar speed movement (n movements after the current one)
                            if m+k < len(tmp_idx):
                                feature_similar_n[l] = feature_recov_tmp[m+k]
                                break
            feature_similar_all.append(feature_similar_n)

            # Calculate the average feature for every
            if norm:
                tmp_stim = ((feature_stim_n - feature_stim_all[0][:len(feature_stim_n)]) / feature_stim_all[0][:len(feature_stim_n)]) * 100
                tmp_similar = ((feature_similar_n - feature_similar_all[0][:len(feature_similar_n)]) / feature_similar_all[0][:len(feature_similar_n)]) * 100
            else:
                tmp_stim = feature_stim_n
                tmp_similar = feature_similar_n
            if method == "mean":
                res[i, j, k, 0] = np.nanmean(tmp_stim)
                res[i, j, k, 1] = np.nanmean(tmp_similar)
            elif method == "median":
                res[i, j, k, 0] = np.nanmedian(tmp_stim)
                res[i, j, k, 1] = np.nanmedian(tmp_similar)


# Plot
# Prepare plotting
fig = plt.figure(figsize=(11.5, 3.5))
colors = np.array(["blue", "green"])
colors_op = np.array(["lightblue", "lightgreen"])
conditions = ["Slow", "Fast"]
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
                             boxprops=dict(facecolor=colors_op[k], color=colors_op[k]),
                             capprops=dict(color=colors_op[k]),
                             whiskerprops=dict(color=colors_op[k]),
                             medianprops=dict(color="indianred", linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor=colors[k], markersize=5,
                                             markeredgecolor='none')
                             )
            bps.append(bp)
        # Add the individual lines
        for dat in res_tmp[:, j, :]:
            plt.plot(bar_pos[0], dat[0], marker='o', markersize=2, color=colors[0])
            plt.plot(bar_pos[1], dat[1], marker='o', markersize=2, color=colors[1])
            # Add line connecting the points
            plt.plot(bar_pos, dat, color="black", linewidth=0.5, alpha=0.3)

        # Add statistics
        """res_perm = scipy.stats.permutation_test(data=(res_tmp[:, j, 0], res_tmp[:, j, 1]),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=100000, permutation_type="samples")
        p = res_perm.pvalue"""
        z, p = scipy.stats.wilcoxon(res_tmp[:, j, 0], res_tmp[:, j, 1])
        sig = "bold" if p < 0.05 else "regular"  #
        xmin, xmax, ymin, ymax = plt.axis()
        # plt.text(bar_pos[j, 0], ymax, p_str, weight=sig, fontsize=10)
        plt.text(bar_pos[0], ymax, np.round(p, 3), weight=sig, fontsize=10)

        # Adjust plot
        plt.xticks(ticks=np.arange(n), labels=[f"stim{x}" for x in range(1, n+1)], fontsize=11)
        plt.yticks(fontsize=12)
        feature_name_plot = feature_name.replace("_", " ")
        if norm:
            plt.ylabel(f"Normalized average \n{feature_name_plot} in %", fontsize=13)
        else:
            plt.ylabel(f"Average {feature_name_plot}", fontsize=13)
        u.despine()

plt.subplots_adjust(left=0.1, top=0.8, right=0.85, wspace=0.5)
plt.suptitle(med, fontsize=14, y=1.1)

plt.show()

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=True)
