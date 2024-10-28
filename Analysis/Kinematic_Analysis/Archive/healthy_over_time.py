# Results Figure 1: Main effect on average change in speed
# Slow vs Fast vs Healthy

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

# Set parameters
feature_name = "mean_speed"
n_norm = 5
n_cutoff = 5
meds = ["Off", "Off", "Healthy"]

# Prepare plotting
plt.figure(figsize=(7, 4))
colors = ["#00863b", "#3b0086", "dimgrey"]
colors_op = ["#b2dac4", "#b099ce", "grey"]
labels = ["Slow", "Fast", "Healthy"]

for i, med in enumerate(meds):

    # Load matrix containing the feature values and the stimulated trials
    feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
    n_datasets, _, _, n_trials = feature.shape

    # Detect and fill outliers (e.g. when subject did not touch the screen)
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)

    # Reshape matrix such that blocks from one condition are concatenated
    feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

    # Delete the first n_cutoff movements
    feature = feature[:, :, n_cutoff:]

    if n_norm != 0:
        # Normalize to average of the first n_norm movements
        feature = u.norm_perc(feature, n_norm=n_norm)

    # Smooth over 5 consecutive movements for plotting
    feature_smooth = u.smooth_moving_average(feature, window_size=5, axis=2)
    feature_smooth = feature_smooth[:, :, 2:feature.shape[-1]-2]

    # Plot the feature over time (compute mean over patients)
    x = np.arange(feature_smooth.shape[-1])
    if med == "Healthy":
        y = np.nanmean(feature_smooth, axis=(0, 1))
        std = np.nanstd(feature_smooth, axis=(0, 1))
    else:
        y = np.nanmean(feature_smooth, axis=0)[i, :]
        std = np.nanstd(feature_smooth, axis=0)[i, :]
    plt.plot(x, y, color=colors[i], label=labels[i], linewidth=3, alpha=0.8)
    # Add standard deviation as shaded area
    plt.fill_between(x, y - std, y + std, color=colors[i], alpha=0.13)
    # Add line at y=0 and x=96
    if n_norm != 0 and i == 0:
        plt.axhline(0, linewidth=1, color="black", linestyle="dashed")
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.axvline(n_trials-n_cutoff, color="black", linewidth=1)

# Adjust plot
feature_name_plot = feature_name.replace("_", " ")
plt.ylabel(f"mean normalized {feature_name_plot} in % ", fontsize=15)
plt.xlabel("Movement number", fontsize=15)
u.despine()
plt.legend(loc="upper right")

# Adjust plot
#plt.subplots_adjust(bottom=0.15, left=0.15, top=0.8, wspace=0.01)
#plt.suptitle(med, fontsize=16, y=0.95)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{n_norm}_{n_cutoff}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{n_norm}_{n_cutoff}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()