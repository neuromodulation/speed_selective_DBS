# Script for grouping the data into blocks depending on the amount of stimulated movements and compute outcome measure

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

# Set parameters
n_blocks = 4
n_moves = 4
feature_name = "peak_speed"
med = "Off"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
feature = feature[4:, :, :, :]
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
n_datasets, _, _, n_trials = feature.shape

# Select only stimulation block
feature = feature[:, :, 0, :]
stim = stim[:, :, 0, :]

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=2, arr=feature)

# Delete the first n_cutoff movements
feature = feature[:, :, n_cutoff:]

if n_norm != 0:
    # Normalize to average of the first n_norm movements
    feature = u.norm_perc(feature, n_norm=n_norm)

# Compute feature in a specific window (defined my the number of stimulated trials)
stim_block_av = np.zeros((n_datasets, 2, n_blocks))
# Get the average feature in blocks in which x stimulations occured
for i in range(n_datasets):
    for cond in range(2):
        stim_cumsum = np.cumsum(stim[i, cond, :])
        for block in range(n_blocks):
            try:
                idx_low = np.where(stim_cumsum == block * n_moves + 1)[0][0]
                idx_high = np.where(stim_cumsum == (block + 1) * n_moves + 1)[0][0]
                if method == "median":
                    stim_block_av[i, cond, block] = np.nanmedian(feature[i, cond, idx_low:idx_high], axis=0)
                else:
                    stim_block_av[i, cond, block] = np.nanmean(feature[i, cond, idx_low:idx_high], axis=0)
            except:
                stim_block_av[i, cond, block] = None
# Delete patients without sufficient data
stim_block_av = stim_block_av[~np.isnan(stim_block_av).any(axis=(1, 2)), :, :]

# Plot as grouped box plots
plt.figure(figsize=(8, 3.5))
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
bps = []
box_width = 0.3
for block in range(n_blocks):
    box_pos = [block - (box_width / 1.5), block + (box_width / 1.5)]
    for j in range(2):
        bp = plt.boxplot(x=stim_block_av[:, j, block],
                         positions=[box_pos[j]],
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                         capprops=dict(color=colors_op[j]),
                         whiskerprops=dict(color=colors_op[j]),
                         meanprops=dict(color=colors[j], linewidth=2),
                         medianprops=dict(color=colors[j], linewidth=0),
                         showmeans=True,
                         meanline=True,
                         flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=5,
                                         markeredgecolor='none')
                         )
        bps.append(bp)
    # Add the individual lines
    for dat in stim_block_av[:, :, block]:
        plt.plot(box_pos[0], dat[0], marker='o', markersize=2, color=colors[0])
        plt.plot(box_pos[1], dat[1], marker='o', markersize=2, color=colors[1])
        # Add line connecting the points
        plt.plot(box_pos, dat, color="black", linewidth=1, alpha=0.1)

    # Add statistics
   # z, p = scipy.stats.wilcoxon(x=stim_block_av[:, 0, block], y=stim_block_av[:, 1, block])
    res = scipy.stats.permutation_test(data=(stim_block_av[:, 0, block], stim_block_av[:, 1, block]),
                                       statistic=u.diff_mean_statistic, alternative='two-sided',
                                       n_resamples=10000, permutation_type="samples")
    p = res.pvalue
    sig = "bold" if p < 0.05 else "regular"
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(block - box_width, ymax, f"p = {np.round(p, 3)}", weight=sig, fontsize=12)

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.15)
feature_name_plot = feature_name.replace("_", " ")
plt.ylabel(f"{method} {feature_name_plot} ", fontsize=15)
labels = [f"stim {n_moves*i}-{n_moves*(i+1)}" for i in range(n_blocks)]
plt.xticks(ticks=np.arange(n_blocks), labels=labels, fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel(f"Number of stimulated movements", fontsize=14)
u.despine()

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()