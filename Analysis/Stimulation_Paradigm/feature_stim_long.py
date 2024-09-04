# Plot the feature during stimulation

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
feature_name = "speed_during_stim"

# Loop over On and Off medication datasets
med = "Off"
fig = plt.figure(figsize=(10, 4))

# Plot with different percentile values
methods = [20, 50, 80, 90, "mean"]
for i, perc in enumerate(methods):

    # Load matrix containing the feature values
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")

    # Select only stimulation blocks
    feature = feature[:, :, 0, :, :]

    # Detect and fill outliers (e.g. when subject did not touch the screen)
    #np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

    # Compute the average feature of the stimulated trials
    if perc == "mean":
        feature_stim_long = np.nanmean(feature, axis=(2, 3))
    else:
        feature_stim_long = np.nanpercentile(feature, q=perc, axis=(2, 3))

    # Plot as boxplot
    box_width = 0.3
    bar_pos = [i-(box_width/1.5), i+(box_width/1.5)]
    colors = ["#00863b", "#3b0086"]
    colors_op = ["#b2dac4", "#b099ce"]
    labels = ["Slow", "Fast"]
    bps = []
    for j in range(2):

        bp = plt.boxplot(x=feature_stim_long[:, j],
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
    for dat in feature_stim_long:
        plt.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
        plt.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
        # Add line connecting the points
        plt.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

    # Add statistics
    r = scipy.stats.permutation_test(data=(feature_stim_long[:, 0], feature_stim_long[:, 1]),
                                     statistic=u.diff_mean_statistic, alternative='two-sided',
                                     n_resamples=100000, permutation_type="samples")
    p = r.pvalue
    sig = "bold" if p < 0.05 else "regular"
    plt.text(i-box_width, np.max(feature_stim_long)+2, f"p = {np.round(p, 3)}", weight=sig, fontsize=12)

# Add legend
plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['Slow', 'Fast'],
           loc='lower center', bbox_to_anchor=(0.95, 0.8),
           prop={'size': 14})

# Adjust plot
plt.xticks(ticks=range(len(methods)), labels=[f"{x}th percentile" if x != "mean" else x for x in methods], fontsize=11)
plt.yticks(fontsize=12)
feature_name_plot = feature_name.replace("_", " ")
plt.ylabel(f"{feature_name_plot} \n [pixel/second]", fontsize=14)
u.despine()

plt.subplots_adjust(bottom=0.15, left=0.15, wspace=0.5)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()