# Results Figure 2: Main effect on average change in speed
# Slow vs Fast vs Healthy

# Import useful libraries
import os
import sys
sys.path.insert(1, "../../../Code")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Set parameters
feature_name = "mean_speed"
med = "Off"
method = "mean"
n_norm = 5
n_cutoff = 5
fontsize = 7.5

# Prepare plotting
colors = ["#3b0086", "#00863b", "dimgrey"]
colors_op = ["#b099ce", "#b2dac4", "grey"]
labels = ["Fast", "Slow", "Healthy"]
titles = ["Stimulation", "Recovery"]
f, axes = plt.subplots(2, 4, figsize=(2.5, 2.5))

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
        y_all = np.nanmean(feature_smooth, axis=(0, 1))
        std_all = np.nanstd(feature_smooth, axis=(0, 1))
    else:
        y_all = np.nanmean(feature_smooth, axis=0)[[1,0][i], :]
        std_all = np.nanstd(feature_smooth, axis=0)[[1,0][i], :]

    # Plot each block separately
    for block in range(2):
        if block == 0:
            y = y_all[:n_trials-n_cutoff]
            std = std_all[:n_trials-n_cutoff]
        else:
            y = y_all[-n_trials:]
            std = std_all[-n_trials:]
        x = np.arange(y.shape[-1])
        if i < 2:
            ax = axes[i, int(block * 2)]
            ax.plot(x, y, label=labels[0], color=colors[i], linewidth=1, alpha=0.8)
            ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
            # Add standard deviation as shaded area
            ax.fill_between(x, y-std, y+std, color=colors_op[i], alpha=0.5)
        else:
            for j in range(2):
                ax = axes[j, int(block * 2)]
                ax.plot(x, y, label=labels[0], color=colors[i], linewidth=1, alpha=0.8)  # Add line at y=0 and x=96
                ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
                # Add standard deviation as shaded area
                ax.fill_between(x, y - std, y + std, color=colors_op[i], alpha=0.25)
        # Adjust plot
        ax.set_xticks([])
        ax.xaxis.set_tick_params(labelsize=fontsize - 2)
        ax.yaxis.set_tick_params(labelsize=fontsize - 2)
        if block == 0:
            ax.spines[['right', 'top']].set_visible(False)
        else:
            ax.spines[['left', 'top', 'right']].set_visible(False)
            ax.set_yticks([])
        ymax = 30
        ymin = -30
        ax.set_ylim([ymin, ymax])
        if i == 0:
            ax.set_title(titles[block], fontsize=fontsize)
axes[1,0].set_xlabel("Movement number", fontsize=fontsize - 2)
axes[1,2].set_xlabel("Movement number", fontsize=fontsize - 2)
axes[1,0].set_xticks([50], ["50"])
axes[1,2].set_xticks([50], ["139"])
axes[1,0].set_ylabel(f"Change in average speed [%]", fontsize=fontsize)
axes[1,0].yaxis.set_label_coords(-0.6, 1)

# Plot boxplots
med = "Off"
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
feature_control = np.load(f"../../../Data/Healthy/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape
n_datasets_control, _, _, n_trials_control = feature_control.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)
feature_control = u.fill_outliers_nan(feature_control)

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
box_width = 0.3

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

    for cond in range(2):

        ax = axes[cond, int(i * 2) + 1]

        # Define bar position
        bar_pos = [-(box_width * 1.5), (box_width * 1.5)]

        # Plot
        feature_av_all = [feature_av[:, [1,0][cond]], feature_av_control]

        if cond == 0:
            colors = ["#3b0086", "dimgrey"]
            colors_op = ["#b099ce", "grey"]
        else:
            colors = ["#00863b", "dimgrey"]
            colors_op = ["#b2dac4", "grey"]
        for j, feat in enumerate(feature_av_all):

            ax.boxplot(x=feat,
                             positions=[bar_pos[j]],
                             widths=box_width,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                             capprops=dict(color=colors_op[j]),
                             whiskerprops=dict(color=colors_op[j]),
                             medianprops=dict(color=colors[j], linewidth=0),
                             showmeans=True,
                             meanline=True,
                            meanprops=dict(color=colors[j], linewidth=0, linestyle="solid"),
                        flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=0.5, markeredgecolor='none')
                             )
            # Add the individual subject points
            for dat in feat:
                ax.plot(bar_pos[j], dat, marker='o', markersize=0.5, color=colors[j])

            # Add statistics
        #z, p = scipy.stats.ttest_ind(feature_av_all[0], feature_av_all[1])
        res = scipy.stats.permutation_test(data=(feature_av_all[0], feature_av_all[1]),
                                           statistic=u.diff_mean_statistic,
                                           n_resamples=100000, permutation_type="independent",)
        p = res.pvalue
        text = u.get_sig_text(p)
        ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 1, ymax, ymax, ymax - 1], color="black",
                linewidth=1)
        ax.text(0, ymax, text, ha="center", va="bottom", fontsize=fontsize-1)

        # Adjust plot
        ax.set_ylim([ymin, ymax])
        ax.xaxis.set_tick_params(labelsize=fontsize-2)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
        ax.set_xticks([])
        ax.set_yticks([])

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.15, top=0.8, wspace=0.01)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{method}_{med}_{n_norm}_{n_cutoff}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{method}_{med}_{n_norm}_{n_cutoff}.png", format="png", transparent=True, bbox_inches="tight")

plt.show()