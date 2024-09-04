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

# Set feature to analyze
feature_name = "median_speed_raw"
feature_name_plot = feature_name.replace("_", " ")

# Loop over On and Off medication datasets
med = "Off"

# Prepare plotting
fig = plt.figure(figsize=(10.5, 3))
colors = ["dimgrey", "dimgrey"]
colors_op = ["plum", "khaki"]
labels = ["right->left", "left->right"]

# Load matrix containing feature values and direction of movement
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_dataset, _, _, n_trials = feature.shape

# Load matrix containing 0/1 indicating which trial was stimulated
trial_side = np.load(f"../../../Data/{med}/processed_data/trial_side.npy")

for i in range(n_dataset):
    feature_dir_all = []
    for dir in range(2):
        feature_dir = feature[i, trial_side[i, :, :, :] == dir]
        # Plot as thin bar
        plt.boxplot(feature_dir, positions=[i+(0.3*dir)], patch_artist=True,showfliers=False,
           boxprops=dict(facecolor=colors_op[dir], color=colors_op[dir]),
                    capprops=dict(color=colors_op[dir]),
                    whiskerprops=dict(color=colors_op[dir]),
                    medianprops=dict(color=colors[dir], linewidth=3),
                    flierprops=dict(marker='o', markerfacecolor=colors_op[dir], markersize=5, markeredgecolor='none')
                    )
        feature_dir_all.append(feature_dir)

    # Add statistics
    z, p = scipy.stats.wilcoxon(feature_dir_all[0], feature_dir_all[1])
    if p < 0.001:
        p_str = "***"
    elif p < 0.05:
        p_str = "*"
    else:
        p_str = "na"
    sig = "bold" if p < 0.05 else "regular"
    plt.text(i - (dir * 0.2) - 0.1, np.percentile(feature_dir[0], 95) + 2, p_str, weight=sig, fontsize=11)

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.2, wspace=0.3, top=0.8)
plt.xticks(np.arange(n_dataset)+0.15, labels=np.arange(n_dataset)+1)
plt.ylabel(f"{feature_name_plot}", fontsize=13)
plt.title(med, fontsize=14)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()