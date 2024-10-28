# Results Figure 2: Main effect on average change in speed
# Slow vs Fast

# Import useful libraries
import os
import sys
sys.path.insert(1, "../Code")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt


# Set parameters
feature_name = "mean_speed"
med = "Off"
method = "mean"
n_norm = 5
n_cutoff = 5
fontsize = 8

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

# Prepare plotting
f, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(3.5, 2))
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

# Plot the feature over time (compute mean over patients)
y_all = np.nanmean(feature_smooth, axis=0)
std_all = np.nanstd(feature_smooth, axis=0)
titles = ["Stimulation", "Recovery"]
for block in range(2):
    if block == 0:
        y = y_all[:, :n_trials-n_cutoff]
        std = std_all[:, :n_trials-n_cutoff]
    else:
        y = y_all[:, -n_trials:]
        std = std_all[:, -n_trials:]
    x = np.arange(y.shape[-1])
    ax = axes[int(block*2)]
    ax.plot(x, y[0, :], label=labels[0], color=colors[0], linewidth=1, alpha=0.8)
    ax.plot(x, y[1, :], label=labels[1], color=colors[1], linewidth=1, alpha=0.8)
    # Add line at y=0 and x=96
    ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
    # Add standard deviation as shaded area
    ax.fill_between(x, y[1, :] - std[1, :], y[1, :] + std[1, :], color=colors_op[1], alpha=0.3)
    ax.fill_between(x, y[0, :] - std[0, :], y[0, :] + std[0, :], color=colors_op[0], alpha=0.3)
    # Adjust plot
    if block == 0:
        ax.set_ylabel(f"Change in average speed [%]", fontsize=fontsize)
    if block == 0:
        ax.set_xticks([50], ["50"])
    else:
        ax.set_xticks([50], ["139"])
    ax.set_xlabel("Movement number", fontsize=fontsize-2)
    ax.xaxis.set_tick_params(labelsize=fontsize-2)
    ax.yaxis.set_tick_params(labelsize=fontsize-2)
    if block == 0:
        ax.spines[['right', 'top']].set_visible(False)
    else:
        ax.spines[['left', 'top', 'right']].set_visible(False)
        ax.set_yticks([])
    ymax = 50
    ymin = -30
    ax.set_ylim([ymin, ymax])
    ttl = ax.set_title(titles[block], fontsize=fontsize)
    ttl.set_position([0.5, 0.1])
    ax.legend(loc="upper left", fontsize=fontsize-2)

# Plot average feature as boxplot with statistics
box_width = 0.3
bps = []
# Loop over stimulation and recovery block
for block in range(2):
    ax = axes[int(block * 2)+1]

    # Get the data for the block
    if block == 0:
        feature_block = feature[:, :, n_norm:n_trials-n_cutoff]
    else:
        feature_block = feature[:, :, -n_trials:]

    # Summarize the values in the bin
    if method == "mean":
        feature_av = np.nanmean(feature_block, axis=-1)
    elif method == "median":
        feature_av = np.nanmedian(feature_block, axis=-1)

    bar_pos = [-(box_width * 1.5), (box_width * 1.5)]
    # Loop over conditions
    for j in range(2):
        bp = ax.boxplot(x=feature_av[:, j],
                    positions=[bar_pos[j]],
                    widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                    capprops=dict(color=colors_op[j]),
                    whiskerprops=dict(color=colors_op[j]),
                    medianprops=dict(color=colors[j], linewidth=0),
                    flierprops=dict(marker='.', markerfacecolor=colors_op[j], markersize=0, markeredgecolor='none')
                    )
        bps.append(bp)  # Save boxplot for creating the legend

    # Add the individual lines
    for dat in feature_av:
        ax.plot(bar_pos[0], dat[0], marker='o', markersize=0.5, color=colors[0])
        ax.plot(bar_pos[1], dat[1], marker='o', markersize=0.5, color=colors[1])
        # Add line connecting the points
        ax.plot(bar_pos, dat, color="black", linewidth=0.3, alpha=0.3)

    # Add statistics
    #z, p = scipy.stats.wilcoxon(x=feature_av[:, 0], y=feature_av[:, 1])
    res = scipy.stats.permutation_test(data=(feature_av[:, 0], feature_av[:, 1]),
                                       statistic=u.diff_mean_statistic,
                                       n_resamples=100000, permutation_type="samples")
    p = res.pvalue
    text = u.get_sig_text(p)
    ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax-1, ymax, ymax, ymax-1], color="black", linewidth=1)
    ax.text(0, ymax, text, ha="center", va="bottom", fontsize=fontsize+2)

    # Adjust plot
    ax.set_ylim([ymin, ymax])
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.15, top=0.8, wspace=0.01)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[-5:]}_{feature_name}_{method}_{med}_{n_norm}_{n_cutoff}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[-5:]}_{feature_name}_{method}_{med}_{n_norm}_{n_cutoff}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()