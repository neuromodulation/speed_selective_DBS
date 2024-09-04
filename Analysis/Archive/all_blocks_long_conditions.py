# PLot the feature over all trials in the original block order (seperated by conditions)

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
import pandas as pd
def statistic(x, y):
    return np.mean(x) - np.mean(y)
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "peak_speed_original_block_order"

# Loop over On and Off medication datasets
meds = ["Off", "On"]

# Prepare plotting
colors = ["dimgrey", "dimgrey"]
colors_op = ["plum", "khaki"]

for i, med in enumerate(meds):

    # Prepare plotting
    f, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 1]}, figsize=(10.5, 6.5))

    # Load matrix containing the feature values and the stimulated trials
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")

    # Load excel sheet containing the information of the block order
    df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
    if med == "Off":
        slow_first = df["Block order"].to_numpy()[1:feature.shape[0] + 1]
    else:
        slow_first = df["Block order"].to_numpy()[:feature.shape[0]]
    slow_fast = np.where(slow_first == "Slow")[0]
    fast_slow = np.where(slow_first == "Fast")[0]

    cond_order_names = ["Slow-Fast", "Fast-Slow"]
    for c, cond_first in enumerate([slow_fast, fast_slow]):

        # Detect and fill outliers (e.g. when subject did not touch the screen)
        #np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

        # Reshape matrix such that all blocks are concatenated
        feature_tmp = feature[cond_first, :, :, :]
        n_datasets, _, _, n_trials = feature_tmp.shape
        feature_tmp = np.reshape(feature_tmp, (n_datasets, n_trials * 4))

        # Delete the first 5 movements
        feature_tmp = feature_tmp[:, 3:]

        # Normalize??
        #feature_tmp = u.norm_perc(feature_tmp)
        #feature_tmp = scipy.stats.zscore(feature_tmp, axis=1)

        # Test
        """feature_test = np.load(f"../../../Data/{med}/processed_data/peak_speed.npy")
        plt.plot(feature_test[1, :, :, :].flatten())
        plt.plot(feature[1, :])
        plt.show()"""

        # Smooth over 5 consecutive movements for plotting
        #feature = u.smooth_moving_average(feature, window_size=5, axis=2)

        # Plot (2 subplots, one with the feature over time and one with the boxplot)
        ax1 = axes[c, 0]
        ax2 = axes[c, 1]

        # Plot the feature over time
        x = np.arange(feature_tmp.shape[-1])
        y = np.nanmean(feature_tmp, axis=0)
        std = np.nanstd(feature_tmp, axis=0)
        ax1.plot(x, y, color="dimgrey", linewidth=3)
        # Add standard deviation as shaded area
        ax1.fill_between(x, y - std, y + std, color="dimgrey", alpha=0.2)
        ax1.fill_between(x, y - std, y + std, color="dimgrey", alpha=0.2)
        ylim = ax1.get_ylim()
        ax1.fill_between(x, ylim[0], ylim [1], where=(x < 96) & (x > 0), color=colors_op[0], alpha=0.5)
        ax1.fill_between(x, ylim[0], ylim [1], where=(x < 96*3) & (x > 96*2), color=colors_op[1], alpha=0.5)
        ax1.set_title(cond_order_names[c], fontsize=15)

        # Adjust plot
        feature_name_plot = feature_name.replace("_", " ")
        ax1.set_ylabel(f"{med} Average\n {feature_name_plot} ", fontsize=13)
        ax1.set_xlabel("Movement number", fontsize=13)
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
        feature_tmp_blocks = np.vstack((np.nanmean(feature_tmp[:, 0:96], axis=-1), np.nanmean(feature_tmp[:, 192:288], axis=-1))).T
        # Loop over stimulation and recovery block
        for j in range(2):
            # Loop over conditions
            bp = ax2.boxplot(x=feature_tmp_blocks[:, j],
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
        for dat in feature_tmp_blocks:
            ax2.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
            ax2.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
            # Add line connecting the points
            ax2.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

            # Add statistics
            z, p = scipy.stats.wilcoxon(x=feature_tmp_blocks[:, 0], y=feature_tmp_blocks[:, 1])
            """res = scipy.stats.permutation_test(data=(feature_mean[:, 0], feature_mean[:, 1]),
                                               statistic=statistic,
                                               n_resamples=10000, permutation_type="samples")"""
            #p = res.pvalue
            sig = "bold" if p < 0.05 else "regular"
            ax2.text(1-box_width, np.max(feature_tmp_blocks)+2, f"p = {np.round(p, 3)}", weight=sig, fontsize=14)

        # Add legend
        ax2.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['First stim block', 'Second stim block'],
                   loc='lower center', bbox_to_anchor=(0.05, 0.8),
                   prop={'size': 14})

        # Adjust subplot
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines[['left', 'right', 'top']].set_visible(False)
        ax2.set_ylim([y_limits[0], y_limits[1]])
        #u.despine()

    # Adjust plot
    plt.subplots_adjust(bottom=0.1, left=0.15, top=0.9, wspace=0.01, hspace=0.5)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/all_blocks_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/all_blocks_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()