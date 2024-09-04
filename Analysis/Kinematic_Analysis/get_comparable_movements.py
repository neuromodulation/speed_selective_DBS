# Get the index of movements which are comparable to the stimulated slow/fast movements in regard to thier speed but were not stimulated
# Option to leave out movements which follow/are followed by a stimulated movement
# Try to get the same number of movements with a distribution which is as similar as possible

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Define mediation condition
med = "Off"
feature_name = "peak_speed"

# Define the number of movement after the target movement

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
cond_names = ["Slow", "Fast"]

# Prepare the figure
plt.figure(figsize=[12, 2.5])
colors = [(np.random.random(1)[0], np.random.random(1)[0], np.random.random(1)[0], 1) for x in range(stim.shape[0])]
colors_op = [(color[0], color[1], color[2], 0.5) for color in colors]

# Loop over conditions
for j in range(2):

    # Build up a matrix of 0s with 1 marking the comparable not-stimulated slow/fast movements
    similar_speed = np.zeros(stim.shape)

    # Loop over datasets
    for k in range(stim.shape[0]):
        k =6

        # Get the feature and stimulation for one patient
        feature_tmp = feature[k, :, :, :].flatten()
        stim_tmp = stim[k, :, :, :].flatten()

        # Get the index of the stimulated movements (same and other condition)
        idx_all = np.where(stim_tmp == 1)[0]
        if j == 0:
            stim_idx = idx_all[idx_all < 96]
        else:
            stim_idx = idx_all[idx_all > 96]
        recovery = np.hstack((np.arange(96, 96*2), np.arange(96*3, 96*4)))

        # Loop over every stimulated movement and find a non-stimulated movement with the closest peak speed
        feature_similar = np.zeros(stim_idx.shape)
        for l, idx in enumerate(stim_idx):
            # Get the difference between the stimulated movement and all other movements
            tmp_idx = np.abs(feature_tmp - feature_tmp[idx]).argsort()
            # Loop over the difference and take the smallest one
            for x in tmp_idx:
                # Check that movement was not already selected
                if (x not in idx_all) and (feature_tmp[x] not in feature_similar):
                    # Check whether the movement was slow/fast in comparison to previous movements
                    feature_similar[l] = feature_tmp[x]
                    similar_speed[np.where(feature == feature_tmp[x])] = 1
                    break

        # Compare the distributions
        color = colors[k]
        color_op = colors_op[k]
        feature_stim = feature_tmp[stim_idx]
        plt.boxplot(feature_stim, positions=[k + (j*0.5) + 0.1], widths=0.1,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color_op, color=color_op),
                    capprops=dict(color=color_op),
                    whiskerprops=dict(color=color_op),
                    medianprops=dict(color=color, linewidth=3),
                    flierprops=dict(marker='o', markerfacecolor=color_op, markersize=5, markeredgecolor='none')
                    )
        plt.boxplot(feature_similar, positions=[k + (j*0.5) - 0.1], widths=0.1,
                    patch_artist=True,showfliers=False,
                    boxprops=dict(facecolor=color_op, color=color_op),
                    capprops=dict(color=color_op),
                    whiskerprops=dict(color=color_op),
                    medianprops=dict(color=color, linewidth=3),
                    flierprops=dict(marker='o', markerfacecolor=color_op, markersize=5, markeredgecolor='none')
                    )

        # Add statistics
        z, p = scipy.stats.wilcoxon(x=feature_stim, y=feature_similar)
        if p < 0.001:
            p_str = "***"
        elif p < 0.05:
            p_str = "*"
        else:
            p_str = ""
        sig = "bold" if p < 0.05 else "regular"
        plt.text(k- (j*0.2) - 0.1, np.percentile(feature_stim, 95) + 2, p_str, weight=sig, fontsize=11)

        # Adjust plot
        u.despine()
        plt.xticks([])
        plt.ylim([0, 9000])
        plt.title(med)
        plt.ylabel(f"{feature_name} of stimulated \n and comparable movement", fontsize=11)

    # Save the similar speed matrix
    np.save(f"../../../Data/{med}/processed_data/{cond_names[j]}_similar.npy",
            similar_speed)

# Save plot
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/comprarable_verification_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/comprarable_verification_{med}.png", format="png", bbox_inches="tight", transparent=True)
plt.show()









