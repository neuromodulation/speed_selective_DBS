# Compare the features for movements from left-right with right-left

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "peak_speed"

# Loop over On and Off medication datasets
meds = ["Off", "On"]

# Prepare plotting
fig = plt.figure(figsize=(5.5, 5.5))
colors = ["dimgrey", "dimgrey"]
colors_op = ["plum", "khaki"]
labels = ["right->left", "left->right"]
feature_name_plot = feature_name.replace("_", " ")

for i, med in enumerate(meds):

    # Load matrix containing featture values and direction of movement
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
    trial_side = np.load(f"../../../Data/{med}/processed_data/trial_side.npy")

    # Extract average feature for each direction (0=right->left, 1=left->right)
    feature_tmp = np.array([[np.nanmean(feature[k, trial_side[k, :, :, :] == 0]), np.nanmean(feature[k, trial_side[k, :, :, :] == 1])]
                            for k in range(feature.shape[0])])
    # Plot as boxplot
    box_width = 0.3
    bar_pos = [i-(box_width/1.5), i+(box_width/1.5)]
    bps = []
    for j in range(2):
        bp = plt.boxplot(x=feature_tmp[:, j],
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
    for dat in feature_tmp:
        plt.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
        plt.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
        # Add line connecting the points
        plt.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

    # Add statistics
    z, p = scipy.stats.wilcoxon(x=feature_tmp[:, 0], y=feature_tmp[:, 1])
    if p < 0.001:
        p_str = "p < 0.001"
    elif p < 0.05:
        p_str = "p < 0.05"
    else:
        p_str = "p > 0.05"
    sig = "bold" if p < 0.05 else "regular"
    plt.text(i-box_width, np.max(feature_tmp)+2, p_str, weight=sig, fontsize=14)

    # Add legend
    plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], labels,
               loc='lower center', bbox_to_anchor=(0.95, 0.7),
               prop={'size': 14})

# Adjust plot
plt.xticks(ticks=[0, 1], labels=meds, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(f"Average {feature_name_plot}", fontsize=15)
plt.xlabel("Medication state", fontsize=15)
u.despine()
plt.subplots_adjust(bottom=0.15, left=0.2, right=0.8)


# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()