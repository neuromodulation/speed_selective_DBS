# Results Figure 1: Main effect on average change in speed, statistics
# Slow vs Fast vs Healthy

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

# Set parameters
feature_name = "peak_speed"
med = "Off"
method = "mean"
n_norm = 5
n_cutoff = 5

# Prepare plotting
colors = ["#00863b", "#3b0086", "dimgrey"]
colors_op = ["#b2dac4", "#b099ce", "grey"]
labels = ["Slow", "Fast", "Healthy"]
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]}, figsize=(6, 4))

# Trace over time
meds = ["Off", "Off", "Healthy"]
for i, med in enumerate(meds):

    # Load matrix containing the feature values and the stimulated trials
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
    n_datasets, _, _, n_trials = feature.shape

    # Detect and fill outliers (e.g. when subject did not touch the screen)
    feature = u.fill_outliers_nan(feature)

    # Reshape matrix such that blocks from one condition are concatenated
    feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

    # Delete the first n_cutoff movements
    feature = feature[:, :, n_cutoff:]

    if n_norm != 0:
        # Normalize to average of the first n_norm movements
        feature = u.norm_perc(feature, n_norm=n_norm)

    # Smooth over 5 consecutive movements for plotting
    feature_smooth = u.smooth_moving_average(feature, window_size=5, axis=2)
    feature_smooth = feature_smooth[:, :, 2:feature.shape[-1]-2]

    # Plot the feature over time (compute mean over patients)
    x = np.arange(feature_smooth.shape[-1])
    if med == "Healthy":
        y = np.nanmean(feature_smooth, axis=(0, 1))
        std = np.nanstd(feature_smooth, axis=(0, 1))
    else:
        y = np.nanmean(feature_smooth, axis=0)[i, :]
        std = np.nanstd(feature_smooth, axis=0)[i, :]
    ax1.plot(x, y, color=colors[i], label=labels[i], linewidth=3, alpha=0.8)
    # Add standard deviation as shaded area
    ax1.fill_between(x, y - std, y + std, color=colors[i], alpha=0.13)

# Adjust suboplot
# Add line at y=0 and x=96
if n_norm != 0:
    ax1.axhline(0, linewidth=1, color="black", linestyle="dashed")
ax1.axvline(n_trials-n_cutoff, color="black", linewidth=1,  linestyle="dashed")
ax1.set_ylabel(f"Change in speed [%] ", fontsize=15)
ax1.set_xlabel(f"Movement number", fontsize=15)
ax1.spines[["top", "right"]].set_visible(False)
# get y limit
ax1.set_ylim([-40,50])
ymin, ymax = ax1.get_ylim()
ax1.legend(frameon=True, fontsize=12, loc="upper left", bbox_to_anchor=(-0.01, 0.95))
ax1.set_xticklabels(ax1.get_xticks(), fontsize=12)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=12)

# Plot boxplots
med = "Off"
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
feature_control = np.load(f"../../../Data/Healthy/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape
n_datasets_control, _, _, n_trials_control = feature_control.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_control)

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))
feature_control = np.reshape(feature_control, (n_datasets_control, 2, n_trials_control * 2))

# Delete the first n_cutoff movements
feature = feature[:, :, n_cutoff:]
feature_control = feature_control[:, :, n_cutoff:]

if n_norm != 0:
    # Normalize to average of the first n_norm movements
    feature = u.norm_perc(feature, n_norm=n_norm)
    feature_control = u.norm_perc(feature_control, n_norm=n_norm)
box_width = 0.22
test = ["samples", "independent", "independent"]
# Loop over stimulation and recovery block and get average feature change
for i in range(2):

    # Get the movements of the stimulation/recovery block
    if i == 0:
        feature_block = feature[:, :, n_norm:n_trials-n_cutoff]
        feature_block_control = feature_control[:, :, n_norm:n_trials_control-n_cutoff]
    else:
        feature_block = feature[:, :, -n_trials:]
        feature_block_control = feature_control[:, :, -n_trials_control:]

    # Summarize the values in the bin
    if method == "mean":
        feature_av = np.nanmean(feature_block, axis=-1)
        feature_av_control = np.nanmean(feature_block_control, axis=(1, 2))
    elif method == "median":
        feature_av = np.nanmedian(feature_block, axis=-1)
        feature_av_control = np.nanmean(feature_block_control, axis=(1, 2))

    # Define bar position
    bar_pos = [i - box_width - (box_width/2), i, i + box_width + (box_width/2)]

    # Plot
    feature_av_all = [feature_av[:, 0], feature_av[:, 1], feature_av_control]
    ypos = ymax
    for j, feat in enumerate(feature_av_all):

        ax2.boxplot(x=feat,
                         positions=[bar_pos[j]],
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                         capprops=dict(color=colors_op[j]),
                         whiskerprops=dict(color=colors_op[j]),
                         medianprops=dict(color=colors[j], linewidth=0),
                         showmeans=True,
                         meanline=True,
                        meanprops=dict(color=colors[j], linewidth=2, linestyle="solid"),
                    flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=5, markeredgecolor='none')
                         )
        # Add points
        for dat in feat:
            ax2.plot(bar_pos[j], dat, marker='o', markersize=2.5, color=colors[j])

        # Add statistics
        x1 = bar_pos[j]
        if j == 0:
            res = scipy.stats.permutation_test(data=(feat, feature_av_all[j+1]),
                                               statistic=u.diff_mean_statistic, alternative='two-sided',
                                               n_resamples=10000, permutation_type="samples")
            x2 = bar_pos[j + 1]
        else:
            if j == 1:
                feature_y = feature_av_all[j+1]
                x2 = bar_pos[j + 1]
            else:
                feature_y = feature_av_all[0]
                x2 = bar_pos[0]
            res = scipy.stats.permutation_test(data=(feat, feature_y),
                                               statistic=u.diff_mean_statistic,
                                               n_resamples=10000, permutation_type="independent")
        p = res.pvalue
        fontsize=16
        if p < 0.001:
            text = "***"
        elif p < 0.01:
            text = "**"
        elif p < 0.05:
            text = "*"
        else:
            text = "ns"
            fontsize=10
        if j == 0:
            ypos = [bar_pos[0], bar_pos[1]]
        elif j == 1:
            ypos = [bar_pos[1], bar_pos[2]]
        else:
            ypos = [bar_pos[0], bar_pos[2]]
        plt.plot(ypos, [ymax, ymax], color="black", linewidth=1)
        plt.text(np.mean(ypos), ymax, text, ha="center", va="bottom", fontsize=fontsize)
        ymax = ymax * 1.1

    for dat in feature_av:
        # Add line connecting the points
        ax2.plot(bar_pos[:2], dat, color="black", linewidth=0.6, alpha=0.3)

# Adjust plot
ax2.set_xlim([0 - box_width*3, 1 + box_width*3])
ax2.set_ylim([ymin, ymax])
if n_norm != 0:
    plt.axhline(0, linewidth=1, color="black", linestyle="dashed")
ax2.set_xticks(ticks=[0, 1], labels=["Stimulation", "Recovery"], fontsize=12, rotation=15)
ax2.spines[["top", "left"]].set_visible(False)
ax2.yaxis.set_visible(False)

# Adjust plot
plt.subplots_adjust(wspace=0.05)

# Save
plot_name = os.path.basename(__file__).split(".")[0][:10]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()