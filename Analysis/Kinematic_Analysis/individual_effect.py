# Plot the feature of all trials for both conditions for each patient

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
feature_name = "peak_speed"

# Set normalization window and cut-off of interest
cutoff = 5
n_norm = 5

# Define medication state
med = "Off"

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

# Load matrix containing the feature values and the stimulated trials
feature_all = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature_all.shape

path_table = "C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - " \
             "PROJECT ReinforceVigor\\vigor_stim_task\\Data\\Dataset_list.xlsx"
df_info = pd.read_excel(path_table)
IDs = df_info["ID Berlin_Neurophys"][1:25].to_numpy()

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_all)

# Reshape matrix such that blocks from one condition are concatenated
feature_all = np.reshape(feature_all, (n_datasets, 2, n_trials * 2))

# Loop over patients
for i in range(n_datasets):

    # Prepare plotting
    f, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(11.5, 7))

    # PLot one normalized and once raw
    for j, method in enumerate(["raw", "normalized"]):

        if method == "normalized":
            feature = feature_all[i, :, cutoff:]
            feature = u.norm_perc(feature, n_norm=n_norm)
        else:
            feature = feature_all[i, :, :]

        # Smooth
        feature_smooth = u.smooth_moving_average(feature, window_size=5, axis=-1)
        feature_smooth = feature_smooth[:, 2:feature.shape[-1] - 2]

        # Plot the feature over time
        ax = axes[j, 0]
        x = np.arange(feature_smooth.shape[-1])
        y = feature_smooth
        ax.plot(x, y[0, :], label=labels[0], color=colors[0], linewidth=1.5, alpha=0.5)
        ax.plot(x, y[1, :], label=labels[1], color=colors[1], linewidth=1.5, alpha=0.5)
        # Add line at y=0 and x=96
        if method == "normalized": ax.axhline(0, linewidth=1, color="black", linestyle="dashed")
        ax.axvline(96, linewidth=1, color="black", linestyle="dashed")

        # Adjust plot
        feature_name_plot = feature_name.replace("_", " ")
        ax.set_ylabel(f"Average {method} \n {feature_name_plot} ", fontsize=15)
        if j == 1: ax.set_xlabel("Movement number", fontsize=15)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.spines[['right', 'top']].set_visible(False)
        y_limits = ax.get_ylim()
        if j == 0:
            ax.text(25, y_limits[1], "Stimulation", rotation=0, fontsize=14)
            ax.text(118, y_limits[1], "Recovery", rotation=0, fontsize=14)

        # Plot values as box plot with statistics
        ax2 = axes[j, 1]
        box_width = 0.3
        for block in range(2):

            if method == "raw":
                feature_block = feature[:, 96*block:96*(block+1)]
            else:
                if block == 0:
                    feature_block = feature[:, n_norm:96-cutoff]
                else:
                    feature_block = feature[:, 96-cutoff:]

            # Get rid of nan values
            slow = feature_block[0, :]
            slow = slow[~np.isnan(slow)]
            fast = feature_block[1, :]
            fast = fast[~np.isnan(fast)]
            feature_block = [slow, fast]

            # Plot as boxplots
            box_pos = [block - (box_width / 1.5), block + (box_width / 1.5)]

            for k in range(2):
                x = feature_block[k]
                ax2.boxplot(x=x,
                                 positions=[box_pos[k]],
                                 widths=box_width,
                                 patch_artist=True,
                                 boxprops=dict(facecolor=colors_op[k], color=colors_op[k]),
                                 capprops=dict(color=colors_op[k]),
                                 whiskerprops=dict(color=colors_op[k]),
                                 medianprops=dict(color=colors[k], linewidth=2),
                                 flierprops=dict(marker='o', markerfacecolor=colors_op[k], markersize=5, markeredgecolor='none')
                                 )
            # Add statistics
            z, p = scipy.stats.ttest_ind(a=slow, b=fast)
            """res = scipy.stats.permutation_test(data=(feature_mean[:, 0], feature_mean[:, 1]),
                                               statistic=statistic,
                                               n_resamples=10000, permutation_type="samples")"""
            # p = res.pvalue
            sig = "bold" if p < 0.05 else "regular"
            ax2.text(block-box_width, np.nanpercentile(feature, 99), f"p = {np.round(p, 3)}", weight=sig, fontsize=14)

        # Adjust plot
        ax2.set_xticks(ticks=[0, 1], labels=["Stimulation", "Recovery"], fontsize=12)
        ax2.spines[['left', 'top', 'right']].set_visible(False)
        ax2.set_yticks([])
        ax2.set_ylim([y_limits[0], y_limits[1]])

    # Adjust plot
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.85, wspace=0.01, hspace=0.4, right=0.95)
    plt.suptitle(IDs[i], fontsize=16, y=0.95)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/Individual/{IDs[i]}_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=False)
    plt.savefig(f"../../../Figures/{dir_name}/Individual/{IDs[i]}_{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=False)

plt.show()