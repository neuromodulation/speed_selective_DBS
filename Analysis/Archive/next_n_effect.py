# Analyze and plot the instantaneous effect of the stimulation (next n movements)

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "mean_speed_raw"
# set medication condition
med = "Off"
# Set normalization
norm = False

# Prepare plotting
fig = plt.figure(figsize=(11.5, 3.5))
colors = np.array([["#00863b", "#b2dac4"], ["#3b0086", "#b099ce"]])
colors_op = np.array([["#00863b", "#b2dac4"], ["#3b0086", "#b099ce"]])
conditions = ["Slow", "Fast"]
box_width = 0.15
bps = []

# Load matrix containing the feature values
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

# Load the index of the stimulated trials
stim_0 = np.load(f"../../../Data/{med}/processed_data/stim.npy")
stim_1 = np.load(f"../../../Data/{med}/processed_data/stim_1.npy")
stim_2 = np.load(f"../../../Data/{med}/processed_data/stim_2.npy")
stim_3 = np.load(f"../../../Data/{med}/processed_data/stim_3.npy")
stim_all = [stim_0, stim_1, stim_2, stim_3]

# Load the index of comparable and subsequent trials
similar_slow_0 = np.load(f"../../../Data/{med}/processed_data/Slow_0_similar.npy")
similar_slow_1 = np.load(f"../../../Data/{med}/processed_data/Slow_1_similar.npy")
similar_slow_2 = np.load(f"../../../Data/{med}/processed_data/Slow_2_similar.npy")
similar_slow_3 = np.load(f"../../../Data/{med}/processed_data/Slow_3_similar.npy")
similar_fast_0 = np.load(f"../../../Data/{med}/processed_data/Fast_0_similar.npy")
similar_fast_1 = np.load(f"../../../Data/{med}/processed_data/Fast_1_similar.npy")
similar_fast_2 = np.load(f"../../../Data/{med}/processed_data/Fast_2_similar.npy")
similar_fast_3 = np.load(f"../../../Data/{med}/processed_data/Fast_3_similar.npy")
similar_all = [[similar_slow_0, similar_slow_1, similar_slow_2, similar_slow_3], [similar_fast_0, similar_fast_1, similar_fast_2, similar_fast_3]]

# Load excel sheet containing the information of the block order
"""df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
slow_first = df["Block order"].to_numpy()[1:n_datasets + 1]
slow_fast = np.where(slow_first == "Slow")[0]
fast_slow = np.where(slow_first == "Fast")[0]
cond_orders = ["slow-fast", "fast-slow"]"""


# Loop over movements (after the stimulated trial)
for i in range(4):

    # Get the average feature of the stimulated trials
    feature_stim_similar = np.zeros((n_datasets, 2, 2))
    #feature_stim_similar = np.zeros((len(fast_slow), 2, 2))

    # Loop over the conditions
    for j, cond in enumerate(conditions):

        # Loop over patients
        for k in range(n_datasets):
        #for k, _ in enumerate(fast_slow):

            # Compute the mean of the stimulated trials and comparable trials
            stim = stim_all[i]
            similar = similar_all[j][i]
            if norm:
                # Get values from the first trial
                stim_0 = feature[k, j, stim_all[0][k, j, :, :] == 1]
                similar_0 = feature[k, similar_all[j][0][k, :, :, :] == 1]
                # Compute the change in speed after a stimulated trial
                feature_stim = feature[k, j, stim[k, j, :, :] == 1]
                feature_stim_similar[k, j, 0] = np.nanmean((feature_stim - stim_0[:len(feature_stim)]) / stim_0[:len(feature_stim)]) * 100
                # Compute the change in speed after a comparable trial
                feature_stim_similar[k, j, 1] = np.nanmean((feature[k, similar[k, :, :, :] == 1] - similar_0) / similar_0) * 100
            else:
                feature_stim_similar[k, j, 0] = np.nanmean(feature[k, j, stim[k, j, :, :] == 1])
                feature_stim_similar[k, j, 1] = np.nanmean(feature[k, similar[k, :, :, :] == 1])

            """
            # Verify method
            x = feature[k, :, :, :].flatten()
            plt.plot(x)
            stim_idx = np.where(stim[k, j, :, :].flatten() == 1)[0]
            for x in stim_idx:
                plt.axvline(x=x, color="red", linewidth=1, alpha=0.5)
            stim_idx = np.where(similar[k, :, :, :].flatten() == 1)[0]
            for x in stim_idx:
                plt.axvline(x=x, color="green", linewidth=1, alpha=0.5)
            plt.show()"""

    # Plot as boxplot
    bar_pos = np.array([[-0.3, -0.1], [0.1, 0.3]]) + i
    bps = []
    for j in range(2):
        for m in range(2):
            bp = plt.boxplot(x=feature_stim_similar[:, j, m],
                             positions=[bar_pos[j, m]],
                             widths=box_width,
                             patch_artist=True,
                             boxprops=dict(facecolor=colors_op[j, m], color=colors_op[j, m]),
                             capprops=dict(color=colors_op[j, m]),
                             whiskerprops=dict(color=colors_op[j, m]),
                             medianprops=dict(color="indianred", linewidth=2),
                             flierprops=dict(marker='o', markerfacecolor=colors_op[j, m], markersize=5,
                                             markeredgecolor='none')
                             )
            bps.append(bp)
            # Add the individual lines
            for dat in feature_stim_similar[:, j, :]:
                plt.plot(bar_pos[j, 0], dat[0], marker='o', markersize=2, color="dimgrey")
                plt.plot(bar_pos[j, 1], dat[1], marker='o', markersize=2, color="dimgrey")
                # Add line connecting the points
                plt.plot(bar_pos[j, :], dat, color="black", linewidth=0.5, alpha=0.3)

        # Add statistics
        if i > 0 and norm or not norm:
            res = scipy.stats.permutation_test(data=(feature_stim_similar[:, j, 0],feature_stim_similar[:, j, 1]),
                                               statistic=u.diff_mean_statistic,
                                               n_resamples=10000, permutation_type="samples")
            p = res.pvalue
            #z, p = scipy.stats.wilcoxon(x=feature_stim_similar[:, j, 0], y=feature_stim_similar[:, j, 1])
            sig = "bold" if p < 0.05 else "regular"#
            xmin, xmax, ymin, ymax = plt.axis()
            #plt.text(bar_pos[j, 0], ymax, p_str, weight=sig, fontsize=10)
            plt.text(bar_pos[j, 0], ymax, np.round(p, 3), weight=sig, fontsize=10)

# Adjust plot
plt.xticks(ticks=[0, 1, 2, 3], labels=["stim0", "stim1", "stim2", "stim3"], fontsize=13)
plt.yticks(fontsize=12)
feature_name_plot = feature_name.replace("_", " ")
if norm:
    plt.ylabel(f"Normalized average \n{feature_name_plot} in %", fontsize=13)
else:
    plt.ylabel(f"Average {feature_name_plot}", fontsize=13)
u.despine()
plt.subplots_adjust(left=0.1, top=0.8, right=0.85)
plt.title(med, fontsize=14, y=1.1)

# Add legend
plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0], bps[2]["boxes"][0], bps[3]["boxes"][0]],
           ['Slow Stim', 'Slow Comparable', 'Fast Stim', 'Fast Comparable'],
           loc='lower center', bbox_to_anchor=(1.05, 0.3),
           prop={'size': 12})

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()