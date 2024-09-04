# Comparison of average change in feature to healthy control data

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
feature_name = "mean_speed_300"
med = "Off"
method = "mean"
n_norm = 5
n_cutoff = 5

# Prepare plotting
colors = ["#00863b", "#3b0086", "dimgrey"]
colors_op = ["#b2dac4", "#b099ce", "grey"]
labels = ["Slow", "Fast", "Healthy"]

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
feature_control = np.load(f"../../../Data/Healthy/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape
n_datasets_control, _, _, n_trials_control = feature_control.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

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

# Plot
plt.figure(figsize=(8, 4))
# Plot average feature as boxplot with statistics
box_width = 0.2
bps = []
# Loop over stimulation and recovery block and get average feature change
for block in range(2):

    # Get the movements of the stimulation/recovery block
    if block == 0:
        feature_block = feature[:, :, n_norm:n_trials-n_cutoff]
        feature_block_control = feature_control[:, :, n_norm:n_trials_control-n_cutoff]
    else:
        feature_block = feature[:, :, -n_trials:-60]
        feature_block_control = feature_control[:, :, -n_trials_control:-60]

    # Summarize the values in the bin
    if method == "mean":
        feature_av = np.nanmean(feature_block, axis=-1)
        feature_av_control = np.nanmean(feature_block_control, axis=(1, 2))
    elif method == "median":
        feature_av = np.nanmedian(feature_block, axis=-1)
        feature_av_control = np.nanmean(feature_block_control, axis=(1, 2))

    # Define bar position
    bar_pos = [block - box_width - (box_width/6), block + box_width + (box_width/6), block]

    # Healthy controls
    bp = plt.boxplot(x=feature_av_control,
                positions=[bar_pos[2]],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=colors_op[2], color=colors_op[2]),
                capprops=dict(color=colors_op[2]),
                whiskerprops=dict(color=colors_op[2]),
                medianprops=dict(color=colors[2], linewidth=2),
                flierprops=dict(marker='o', markerfacecolor=colors_op[2], markersize=5, markeredgecolor='none')
                )
    bps.append(bp)
    # Add points
    for dat in feature_av_control:
        plt.plot(bar_pos[2], dat, marker='o', markersize=2.5, color=colors[2])

    # Add the boxplot for the healthy controls
    # Loop over conditions
    for j in range(2):
        bp = plt.boxplot(x=feature_av[:, j],
                    positions=[bar_pos[j]],
                    widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                    capprops=dict(color=colors_op[j]),
                    whiskerprops=dict(color=colors_op[j]),
                    medianprops=dict(color=colors[j], linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=5, markeredgecolor='none')
                    )
        bps.append(bp)

        # Add points
        for dat in feature_av[:, j]:
            plt.plot(bar_pos[j], dat, marker='o', markersize=2.5, color=colors[j])

    # Add statistics
    # Slow
    res = scipy.stats.permutation_test(data=(feature_av[:, 0], feature_av_control),
                                       statistic=u.diff_mean_statistic,
                                       n_resamples=100000, permutation_type="independent")
    p = res.pvalue
    sig = "bold" if p < 0.05 else "regular"
    ymin, ymax = plt.ylim()
    plt.text(block-box_width-(box_width/2), ymax, f"p = {np.round(p, 3)}", weight=sig, fontsize=12)
    # Fast
    res = scipy.stats.permutation_test(data=(feature_av[:, 1], feature_av_control),
                                       statistic=u.diff_mean_statistic,
                                       n_resamples=100000, permutation_type="independent")
    p = res.pvalue
    sig = "bold" if p < 0.05 else "regular"
    ymin, ymax = plt.ylim()
    plt.text(block+box_width-(box_width/2), ymax, f"p = {np.round(p, 3)}", weight=sig, fontsize=12)

# Add legend
plt.legend([bps[1]["boxes"][0], bps[0]["boxes"][0], bps[2]["boxes"][0]], ['Slow', 'Healthy', 'Fast'],
           loc='lower center', bbox_to_anchor=(1, 0.6),
           prop={'size': 13})

# Adjust plot
feature_name_plot = feature_name.replace("_", " ")
if n_norm != 0:
    plt.axhline(0, linewidth=1, color="black", linestyle="dashed")
    plt.ylabel(f"{method} normalized \n {feature_name_plot} in % ", fontsize=15)
else:
    plt.ylabel(f"{method} {feature_name_plot} ", fontsize=15)
plt.xticks(ticks=[0, 1], labels=["Stimulation", "Recovery"], fontsize=14)
u.despine()
plt.yticks(fontsize=13)

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.17)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()