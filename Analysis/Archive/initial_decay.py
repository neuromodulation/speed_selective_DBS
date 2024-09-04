# Analyze the first 20 movements only form the stimulation blocks

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from scipy.stats import percentileofscore, spearmanr
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb
def statistic(x, y):
    return np.mean(x) - np.mean(y)
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "mean_speed"

# Set the number of movements to analyze
n_movements = 10

# Loop over On and Off medication datasets
meds = ["Off", "Off"]

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

for i, med in enumerate(meds):

    # Load matrix containing the feature values and the stimulated trials
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
    n_datasets, _, _, n_trials = feature.shape

    # Detect and fill outliers (e.g. when subject did not touch the screen)
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

    # Choose the first x movements from the stimulation blocks
    #feature = feature[:, :, 0, 3:n_movements]
    feature = feature[:, :, 0, :3]

    # Compute the average feature of all trials for each subject
    feature_mean = np.nanmean(feature, axis=-1)

    # Compute correlation between average peak speed and number of stimulated trials
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    stim = stim[:, :, 0, 3:n_movements]
    stim_sum = np.sum(stim, axis=-1)

    # Plot (2 subplots, one with the feature over time and one with the boxplot)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]}, figsize=(10.5, 3.5))

    for j in range(2):

        # Plot the feature over time
        x = np.arange(feature.shape[-1])
        y = np.nanmean(feature[:, j, :], axis=0)
        std = np.nanstd(feature[:, j, :], axis=0)
        ax1.plot(x, y, label=labels[j], color=colors[j], linewidth=3)
        # Add standard deviation as shaded area
        ax1.fill_between(x, y-std, y+std, color=colors_op[j], alpha=0.5)

        # Plot average feature as boxplot with statistics
        box_width = 0.3
        bar_pos = [1 - (box_width / 1.5), 1 + (box_width / 1.5)]
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

        # Plot correlation between peak speed and number of stimulated trials
        x = stim_sum[:, j].flatten()
        y = feature_mean[:, j].flatten()
        #x = stim_sum[:, 0] - stim_sum[:, 1]
        #y = feature_mean[:, 0] - feature_mean[:, 1]
        corr, p = spearmanr(x, y, nan_policy='omit')
        p = np.round(p, 3)
        if p < 0.05:
            label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
        else:
            label = f" R = {np.round(corr, 2)} p = {p}"
        sb.regplot(x=x, y=y, label=label, scatter_kws={"color": colors_op[j]}, line_kws={"color": colors[j]}, ax=ax3)
        ax3.legend(loc="upper right", fontsize=11)
        ax3.set_yticks([])
        ax3.set_xlabel("Number of \n stimulated trials", fontsize=12)
        ax3.spines[['left', 'right', 'top']].set_visible(False)

    # Adjust subplots
    feature_name_plot = feature_name.replace("_", " ")
    ax1.set_ylabel(f"Average \n {feature_name_plot} ", fontsize=15)
    ax1.set_xlabel("Movement number", fontsize=15)
    ax1.xaxis.set_tick_params(labelsize=12)
    #ax1.set_xticks([0, 1, 2, 3, 4], labels=[5, 6, 7, 8, 9], fontsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    ax1.spines[['right', 'top']].set_visible(False)
    y_limits = ax1.get_ylim()
    ax1.legend(fontsize=11)

    # Add the individual lines
    for dat in feature_mean:
        ax2.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
        ax2.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
        # Add line connecting the points
        ax2.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

    # Add statistics
    z, p = scipy.stats.wilcoxon(x=feature_mean[:, 0], y=feature_mean[:, 1])
    res = scipy.stats.permutation_test(data=(feature_mean[:, 0], feature_mean[:, 1]),
                                       statistic=statistic,
                                       n_resamples=10000, permutation_type="samples")
    #p = res.pvalue
    sig = "bold" if p < 0.05 else "regular"
    ax2.text(j-box_width, np.max(feature_mean)+2, f"p = {np.round(p, 3)}", weight=sig, fontsize=12)
    ax2.set_xticks(bar_pos, labels=labels, fontsize=14)
    ax2.set_yticks([])
    ax2.spines[['left', 'right', 'top']].set_visible(False)
    ax2.set_ylim([y_limits[0], y_limits[1]])

    # Adjust figure
    plt.subplots_adjust(bottom=0.25, left=0.1, top=0.8, wspace=0.01)
    plt.suptitle(med, fontsize=16, y=0.95)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()