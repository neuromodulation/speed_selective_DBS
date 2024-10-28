# Plot the feature of all trials for both conditions and compare the average feature values (raw and normalized)

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
matplotlib.use('TkAgg')

# How many blocks to divide the data in
n_bins_per_block = 2

# Set parameters
feature_name = "peak_speed"
med = "Off"
method = "median"
n_norm = 5
n_cutoff = 5

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
#feature = feature[4:, :, :, :]
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

# Delete the first n_cutoff movements
feature = feature[:, :, n_cutoff:]

if n_norm != 0:
    # Normalize to average of the first n_norm movements
    feature = u.norm_perc(feature, n_norm=n_norm)

# Loop over stimulation and recovery block and get the values
res = np.zeros((2, n_bins_per_block, n_datasets, 2))
blocks = ["Stimulation", "Recovery"]
x_labels = []
for i, _ in enumerate(blocks):

    # Get the data for the block
    if i == 0:
        feature_block = feature[:, :, n_norm:n_trials-n_cutoff]
    else:
        feature_block = feature[:, :, -n_trials:]
    n_trials_block = feature_block.shape[-1]
    n_trials_bin = np.floor(n_trials_block / n_bins_per_block)

    x_labels.append([f"{int(x * n_trials_bin)+1}-{int((x + 1) * n_trials_bin)+1}" for x in range(n_bins_per_block)])

    # Loop over bins
    for j in range(n_bins_per_block):

        # Get the features in a bin
        feature_bin = feature_block[:, :, int(j * n_trials_bin):int((j + 1) * n_trials_bin)]

        # Summarize the values in the bin
        if method == "mean":
            res[i, j, :, :] = np.nanmean(feature_bin, axis=-1)
        elif method == "median":
            res[i, j, :, :] = np.nanmedian(feature_bin, axis=-1)

# Plot the results______________________________________________________________________
box_width = 0.3
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]
plt.figure(figsize=(8, 4))
res_min = np.min(res)
res_max = np.max(res)
ylim_max = res_max + (res_max-res_min)*0.15
pos_p = res_max + (res_max-res_min)*0.1
ylim_min = res_min - (res_max-res_min)*0.15
for i, block_name in enumerate(blocks):
    plt.subplot(1, 2, i + 1)
    for j in range(n_bins_per_block):
        bar_pos = [j - (box_width / 1.5), j + (box_width / 1.5)]
        # Loop over conditions
        bps = []
        for k in range(2):
            bp = plt.boxplot(x=res[i, j, :, k],
                        positions=[bar_pos[k]],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors_op[k], color=colors_op[k]),
                        capprops=dict(color=colors_op[k]),
                        whiskerprops=dict(color=colors_op[k]),
                        medianprops=dict(color=colors[k], linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor=colors_op[k], markersize=5, markeredgecolor='none')
                        )
            bps.append(bp)
        # Add the individual lines
        for dat in res[i, j, :, :]:
            plt.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
            plt.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
            # Add line connecting the points
            plt.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

        # Add statistics
        r = scipy.stats.permutation_test(data=(res[i, j, :, 0], res[i, j, :, 1]),
                                       statistic=u.diff_mean_statistic, alternative='two-sided',
                                       n_resamples=10000, permutation_type="samples")
        p = r.pvalue
        #z, p = scipy.stats.wilcoxon(x=res[i, j, :, 0], y=res[i, j, :, 1])
        sig = "bold" if p < 0.05 else "regular"
        plt.text(bar_pos[0], pos_p, f"p = {np.round(p, 3)}", weight=sig, fontsize=11)

    # Adjust subplot
    if n_norm != 0:
        plt.axhline(0, linewidth=1, color="black", linestyle="dashed")
    plt.xticks(ticks=range(n_bins_per_block), labels=x_labels[i], fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel(block_name, fontsize=14)
    plt.ylim([ylim_min, ylim_max])
    plt.xlim([-1 + box_width*1.5, n_bins_per_block - box_width*1.5])
    if i == 0:
        u.despine(['right', 'top'])
        plt.ylabel(f"{feature_name} {method}", fontsize=14)
    else:
        u.despine(['left', 'top', 'right'])
        plt.yticks([])

        # Add legend
        plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['Slow', 'Fast'],
                   bbox_to_anchor=(0.9, 0.6),
                   prop={'size': 13})

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.1, top=0.8, wspace=0.01, right=0.85)
plt.suptitle(med, fontsize=16, y=0.95)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{med}_{method}_{n_cutoff}_{n_norm}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{med}_{method}_{n_cutoff}_{n_norm}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()