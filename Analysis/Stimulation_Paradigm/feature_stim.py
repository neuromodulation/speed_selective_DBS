# PLot the feature of the stimulated trials

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

# Set feature to analyze
feature_name = "peak_speed"

# Loop over On and Off medication datasets
med = "Off"
fig = plt.figure(figsize=(6, 4.5))

# Plot once with raw peak speed values and once with percentiles in the block
methods = ["percentile", "percentile"]
for i, method in enumerate(methods):

    # Load matrix containing the feature values and the stimulated trials
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")

    # Select only stimulation blocks
    stim = stim[:, :, 0, :]
    feature = feature[:, :, 0, :]

    # Detect and fill outliers (e.g. when subject did not touch the screen)
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=2, arr=feature)

    # Plot as boxplot
    box_width = 0.25
    bar_pos = [i-(box_width/1.5), i+(box_width/1.5)]
    ax = plt.gca()
    if i == 1:
        ax.spines[["right", "left", "bottom"]].set_visible(False)
        ax = ax.twinx()
    colors = ["#00863b", "#3b0086"]
    colors_op = ["#b2dac4", "#b099ce"]
    labels = ["Slow", "Fast"]
    bps = []
    feature_stim = np.zeros((stim.shape[0], 2))
    for j in range(2):

        # Get the average feature of the stimulated trials
        for k in range(stim.shape[0]):
            if i == 0:
                feature_stim[k, j] = np.nanmean(feature[k, j, stim[k, j, :] == 1])
            else:
                feature_stim[k, j] = np.nanmean([percentileofscore(feature[k, j, :], x, nan_policy='omit') for x in feature[k, j, :][stim[k, j, :] == 1]])

        # Plot axis on both sides
        bp = ax.boxplot(x=feature_stim[:, j],
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
    for dat in feature_stim:
        ax.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
        ax.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
        # Add line connecting the points
        ax.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

    # Add statistics
    r = scipy.stats.permutation_test(data=(feature_stim[:, 0], feature_stim[:, 1]),
                                     statistic=u.diff_mean_statistic, alternative='two-sided',
                                     n_resamples=100000, permutation_type="samples")
    p = r.pvalue
    if p < 0.05:
        text = "*"
    if p < 0.01:
        text = "**"
    if p < 0.001:
        text = "***"
    else:
        text = "na"
    ymin, ymax = ax.get_ylim()
    ax.plot(bar_pos, [ymax, ymax], color="black", linewidth=1)
    ax.text(i, ymax, text, ha="center", va="bottom", fontsize=16)
    print(f"min {np.round(min(feature_stim[:, 0]), 2)}, max {np.round(max(feature_stim[:, 0]), 2)}, min {np.round(min(feature_stim[:, 1]), 2)}, max {np.round(max(feature_stim[:, 1]), 2)}")

    # Add legend
    ax.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['Slow', 'Fast'],
               loc='lower center', bbox_to_anchor=(0.85, 0.1),
               prop={'size': 14})

    # Adjust plot
    ax.set_xticks([])#ticks=[0, 1], labels=methods, fontsize=14)
    y_ticks= ax.get_yticklabels()
    ax.set_yticklabels(y_ticks, fontsize=12)
    feature_name_plot = feature_name.replace("_", " ")
    if i == 0:
        #ax.set_ylabel(f"Average {feature_name_plot} of\nstimulated movements [pixel/second]", fontsize=13)
        ax.set_ylabel(f"Raw speed of\nstimulated movements [pixel/second]", fontsize=13)
    else:
        ax.set_ylabel("Percentile\nof stimulated movements [%]", fontsize=13)

    ax.spines["top"].set_visible(False)
plt.subplots_adjust(bottom=0.15, left=0.15, right=0.85, wspace=0.4)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()