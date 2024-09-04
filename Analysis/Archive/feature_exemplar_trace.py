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
feature_name = "mean_speed_300"

# Define medication state
med = "Off"

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets = feature.shape[0]
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")

# Select only stimulation blocks
stim = stim[:, :, , :]
feature = feature[:, :, 0, :]

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=2, arr=feature)

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
conditions = ["Slow", "Fast"]

# Loop over patients
for i in range(n_datasets):

    # Prepare plotting
    fig = plt.figure(figsize=(12.5, 5.5))

    # Loop over conditions
    for j in range(2):

        plt.subplot(1, 2, j+1)

        # Plot feature over time
        plt.plot(feature[i, j, :], color=colors[j], linewidth=3)

        # Add stimulated trials
        stim_idx = np.where(stim[i, j, :])[0]
        for idx in stim_idx:
            plt.text(idx, feature[i, j, idx], "*", color="red", fontsize=18, ha="center", va="center")

        # Adjust subplot
        feature_name_plot = feature_name.replace("_", " ")
        plt.ylabel(f"Raw {feature_name_plot} ", fontsize=13)
        if j == 1: plt.xlabel("Movement number", fontsize=13)
        u.despine()

    # Adjust plot
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.85, hspace=0.4, right=0.95)
    plt.suptitle(med, fontsize=16, y=0.95)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/Individual/stimtrace_{i}_{feature_name}_{med}.svg", format="svg",
                bbox_inches="tight", transparent=False)
    plt.savefig(f"../../../Figures/{dir_name}/Individual/stimtrace_{i}_{feature_name}_{med}.png", format="png",
                bbox_inches="tight", transparent=False)

plt.show()

