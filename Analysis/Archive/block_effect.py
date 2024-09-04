# PLot the feature of all trials over all trials in the original block order

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
def statistic(x, y):
    return np.mean(x) - np.mean(y)
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "mean_speed_original_block_order"

# Loop over On and Off medication datasets
meds = "Off"

# Prepare plotting
colors = ["dimgrey", "dimgrey"]
colors_op = ["plum", "khaki"]

# Plot once with raw peak speed values and once with percentiles in the block

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, n_trials * 4))

# Plot (2 subplots, one with the feature over time and one with the boxplot)
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(10.5, 3.5))

# Plot the feature over time
x = np.arange(feature.shape[-1])
y = np.nanmean(feature, axis=0)
std = np.nanstd(feature, axis=0)
ax1.plot(x, y, color="dimgrey", linewidth=3)
# Add standard deviation as shaded area
ax1.fill_between(x, y - std, y + std, color="dimgrey", alpha=0.2)
ax1.fill_between(x, y - std, y + std, color="dimgrey", alpha=0.2)
ylim = ax1.get_ylim()
ax1.fill_between(x, ylim[0], ylim [1], where=(x < 96) & (x > 0), color=colors_op[0], alpha=0.5)
ax1.fill_between(x, ylim[0], ylim [1], where=(x < 96*3) & (x > 96*2), color=colors_op[1], alpha=0.5)

# Adjust plot
feature_name_plot = feature_name.replace("_", " ")
ax1.set_ylabel(f"Average\n {feature_name_plot} ", fontsize=15)
ax1.set_xlabel("Movement number", fontsize=15)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
ax1.spines[['right', 'top']].set_visible(False)
y_limits = ax1.get_ylim()
#ax1.text(25, y_limits[1], "Stimulation", rotation=0, fontsize=14)
#ax1.text(118, y_limits[1], "Recovery", rotation=0, fontsize=14)

# Plot average feature as boxplot with statistics
box_width = 0.3
bar_pos = [1 - (box_width / 1.5), 1 + (box_width / 1.5)]
bps = []
feature_tmp = np.vstack((np.nanmean(feature[:, 0:96], axis=-1), np.nanmean(feature[:, 192:288], axis=-1))).T
# Loop over stimulation and recovery block
for j in range(2):
    # Loop over conditions
    bp = ax2.boxplot(x=feature_tmp[:, j],
                positions=[bar_pos[j]],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                capprops=dict(color=colors_op[j]),
                whiskerprops=dict(color=colors_op[j]),
                medianprops=dict(color=colors[j], linewidth=2),
                flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=5, markeredgecolor='none')
                )
    bps.append(bp)  # Save boxplot for creating the legend

# Add the individual lines
for dat in feature_tmp:
    ax2.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
    ax2.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
    # Add line connecting the points
    ax2.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

    # Add statistics
    z, p = scipy.stats.wilcoxon(x=feature_tmp[:, 0], y=feature_tmp[:, 1])
    """res = scipy.stats.permutation_test(data=(feature_mean[:, 0], feature_mean[:, 1]),
                                       statistic=statistic,
                                       n_resamples=10000, permutation_type="samples")"""
    #p = res.pvalue
    sig = "bold" if p < 0.05 else "regular"
    ax2.text(1-box_width, np.max(feature_tmp)+2, f"p = {np.round(p, 3)}", weight=sig, fontsize=14)

# Add legend
ax2.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['First stim block', 'Second stim block'],
           loc='lower center', bbox_to_anchor=(0.05, 0.01),
           prop={'size': 14})

# Adjust subplot
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines[['left']].set_visible(False)
ax2.set_ylim([y_limits[0], y_limits[1]])
u.despine()

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.1, top=0.8, wspace=0.01)
plt.suptitle(med, fontsize=16, y=0.95)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()