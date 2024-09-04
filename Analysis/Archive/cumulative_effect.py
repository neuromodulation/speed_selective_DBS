# Plot the cumulative change in the feature

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
feature_name = "mean_speed"

# Set normalization window and cut-off of interest
cutoff = 5
n_norm = 5

# define medication state
med = "Off"

# define method
method = "normalized"

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

# Cutoff the first x movements
feature = feature[:, :, cutoff:]
#feature = u.norm(feature, n_norm=n_norm)

if method == "normalized":
    # Delete the first 5 movements
    # Normalize to average of the next 5 movements
    #feature = u.norm_perc(feature, n_norm=n_norm)
    feature = u.norm_perc(feature, n_norm=n_norm)

# Smooth over 5 consecutive movements for plotting
feature = u.smooth_moving_average(feature, window_size=5, axis=2)

# Compute the cumulative difference
#perc_diff = np.diff(feature, axis=2) / feature[:, :, :-1]
#feature = np.nancumsum(perc_diff, axis=2)
#feature = perc_diff
#feature = np.nancumsum(np.diff(feature, axis=2), axis=2)
#feature = np.nancumsum(feature, axis=2)

# Plot (2 subplots, one with the feature over time and one with the boxplot)
f, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(10.5, 3.5))

# Plot the feature over time
x = np.arange(feature.shape[-1])
y = np.nanmedian(feature, axis=0)
std = np.nanstd(feature, axis=0)
ax1.plot(x, y[0, :], label=labels[0], color=colors[0], linewidth=3, alpha=0.8)
ax1.plot(x, y[1, :], label=labels[1], color=colors[1], linewidth=3, alpha=0.8)
# Add line at y=0 and x=96
if method == "normalized": ax1.axhline(0, linewidth=2, color="black", linestyle="dashed")
ax1.axvline(96, linewidth=1, color="black", linestyle="dashed")
ax1.axhline(0, linewidth=1, color="black", linestyle="dashed")
# Add standard deviation as shaded area
ax1.fill_between(x, y[0, :] - std[0, :], y[0, :] + std[0, :], color=colors_op[0], alpha=0.5)
ax1.fill_between(x, y[1, :] - std[1, :], y[1, :] + std[1, :], color=colors_op[1], alpha=0.5)

# Add lines displaying statistics
for i in range(feature.shape[-1]):
    z, p = scipy.stats.wilcoxon(x=feature[:, 0, i], y=feature[:, 1, i])
    if p < 0.05:
        ax1.axvline(i, linewidth=1, color="black", alpha=0.5)

# Adjust plot
feature_name_plot = feature_name.replace("_", " ")
ax1.set_ylabel(f"Average {method} \n {feature_name_plot} ", fontsize=15)
ax1.set_xlabel("Movement number", fontsize=15)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
ax1.spines[['right', 'top']].set_visible(False)
y_limits = ax1.get_ylim()
ax1.text(25, y_limits[1], "Stimulation", rotation=0, fontsize=14)
ax1.text(118, y_limits[1], "Recovery", rotation=0, fontsize=14)

# Plot average feature as boxplot with statistics
box_width = 0.3
bps = []
# Loop over stimulation and recovery block
for block in range(2):

    if method == "raw":
        feature_mean = feature[:, :, int((block + 1) * feature.shape[-1] / 2)-1]
        feature_mean = np.nanmean(feature[:, :, 75:95], axis=-1)
        #feature_mean = feature[:, :, 32]
    else:
        if block == 0:
            feature_mean = np.nanmedian(feature[:, :, n_norm:96 - cutoff], axis=-1)
        else:
            feature_mean = np.nanmedian(feature[:, :, 96 - cutoff:], axis=-1)

    bar_pos = [block - (box_width / 1.5), block + (box_width / 1.5)]
    # Loop over conditions
    for j in range(2):
        bp = ax2.boxplot(x=feature_mean[:, j],
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
        for dat in feature_mean:
            ax2.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
            ax2.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
            # Add line connecting the points
            ax2.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

        # Add statistics
        z, p = scipy.stats.wilcoxon(x=feature_mean[:, 0], y=feature_mean[:, 1])
        """res = scipy.stats.permutation_test(data=(feature_mean[:, 0], feature_mean[:, 1]),
                                           statistic=statistic,
                                           n_resamples=10000, permutation_type="samples")"""
        #p = res.pvalue
        sig = "bold" if p < 0.05 else "regular"
        ax2.text(block-box_width, np.nanpercentile(feature_mean, 95), f"p = {np.round(p, 3)}", weight=sig, fontsize=14)

    # Add legend
    ax2.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['Slow', 'Fast'],
               loc='lower center', bbox_to_anchor=(0.05, 0.01),
               prop={'size': 14})

# Adjust subplot
if method == "normalized": ax2.axhline(0, linewidth=1, color="black", linestyle="dashed")
ax2.set_xticks(ticks=[0, 1], labels=["Stimulation", "Recovery"], fontsize=14)
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
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()