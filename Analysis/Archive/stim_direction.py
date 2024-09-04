# Check whether one direction is stimulated more or less than the other

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

# Loop over On and Off medication datasets
meds = ["Off", "On"]

# Prepare plotting
fig = plt.figure(figsize=(7.5, 3.5))
colors = ["dimgrey", "dimgrey"]
colors_op = ["plum", "khaki"]
labels = ["Slow", "Fast"]
labels_dir = ["right->left", "left->right"]

for i, med in enumerate(meds):

    # Load matrix containing feature values and direction of movement
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")

    # Load matrix containing 0/1 indicating which trial was stimulated
    trial_side = np.load(f"../../../Data/{med}/processed_data/trial_side.npy")

    # Select only stimulation blocks
    stim = stim[:, :, 0, :]
    trial_side = trial_side[:, :, 0, :]

    # Plot
    plt.subplot(1, 2, i+1)
    colors = [(np.random.random(1)[0], np.random.random(1)[0], np.random.random(1)[0], 1) for x in range(stim.shape[0])]

    # Loop over conditions
    for j in range(2):

        # Extract average feature for each direction (0=right->left, 1=left->right)
        feature_tmp = np.array(
            [[np.sum(stim[k, j, trial_side[k, j, :] == 0])/np.sum(stim[k, j, :]),
              np.sum(stim[k, j, trial_side[k, j, :] == 1])/np.sum(stim[k, j, :])]
             for k in range(stim.shape[0])]) * 100

        # Loop over directions
        box_width = 0.3
        bar_pos = [j - (box_width / 1.5), j + (box_width / 1.5)]
        for k in range(2):

            # Plot as boxplot
            bps = []
            bp = plt.boxplot(x=feature_tmp[:, k],
                        positions=[bar_pos[k]],
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors_op[k], color=colors_op[k]),
                        capprops=dict(color=colors_op[k]),
                        whiskerprops=dict(color=colors_op[k]),
                        medianprops=dict(color=colors[k], linewidth=0),
                        flierprops=dict(marker='o', markerfacecolor=colors_op[k], markersize=5, markeredgecolor='none')
                        )
            bps.append(bp)  # Save boxplot for creating the legend

        # Add the individual lines
        for l, dat in enumerate(feature_tmp):
            plt.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color="grey")#colors[l])
            plt.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color="grey")#colors[l])
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
        plt.text(bar_pos[0]-box_width, np.max(feature_tmp), p_str, weight=sig, fontsize=14)

        # Add legend
        """plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], labels,
                   loc='lower center', bbox_to_anchor=(0.95, 0.7),
                   prop={'size': 14})"""

    # Adjust plot
    plt.xticks(ticks=[0, 1], labels=labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel(f"% stimulated", fontsize=15)
    plt.title(med, fontsize=15, y=1.1)
    u.despine()

plt.subplots_adjust(bottom=0.15, left=0.2, wspace=0.3, top=0.8)


# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()