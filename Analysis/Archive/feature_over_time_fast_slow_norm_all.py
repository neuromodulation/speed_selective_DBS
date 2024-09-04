# Script for plotting feature over time

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# Set analysis parameters
feature_name = "peak_speed"
med = "OFF"  # "on", "off", "all"
plot_individual = False

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
feature_matrix = feature_matrix[1:, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_matrix)

# Reshape matrix such that blocks from one condition are concatenated
feature_matrix = np.reshape(feature_matrix, (n_datasets, 2, n_trials*2))

# Delete the first 5 movements
feature_matrix = feature_matrix[:, :, 5:]

# Normalize my substracting the average of both conditions
feature_matrix = u.norm_perc(feature_matrix)
#feature_matrix = u.norm_all(feature_matrix)

# Smooth over 5 consecutive movements for plotting
feature_matrix = u.smooth_moving_average(feature_matrix, window_size=5, axis=2)

# Plot individual if needed
if plot_individual:
    for i in range(n_datasets):
        # Plot feature over time
        plt.figure(figsize=(10, 3))
        u.plot_conds(feature_matrix[i, :, :])
        plt.xlabel("Movement number", fontsize=14)
        feature_name_space = feature_name.replace("_", " ")
        plt.ylabel(f"Change in {feature_name_space} [%]", fontsize=14)
        plt.title(f"dataset_{i}_{feature_name}_{med}")
        # Save figure on individual basis
        #plt.savefig(f"../../Plots/dataset_{i}_{feature_name}_{med}.png", format="png", bbox_inches="tight")
        #plt.close()

# Average over all datasets
feature_matrix_mean = np.nanmean(feature_matrix, axis=0)
feature_matrix_std = np.nanstd(feature_matrix, axis=0)

# Plot feature over time
fig = plt.figure()
u.plot_conds(feature_matrix_mean, feature_matrix_std)
plt.xlabel("Movement number", fontsize=14)
feature_name_space = feature_name.replace("_", " ")
#plt.ylabel(f"Change in {feature_name_space} [%]", fontsize=14)

# Add line to mark end of stimulation
n_trials = feature_matrix.shape[-1]
plt.axvline(n_trials/2, color="black", linewidth=1)
axes = plt.gca()
ymin, ymax = axes.get_ylim()
axes.spines[['right', 'top']].set_visible(False)
plt.text(25, ymax+2, "Stimulation", rotation=0, fontsize=12)
plt.text(118, ymax+2, "Recovery", rotation=0, fontsize=12)

# Adjust plot
plt.xlim([0, n_trials-1])
plt.subplots_adjust(bottom=0.2, left=0.15)
u.despine()
plt.legend()

# Save figure on group basis
plt.savefig(f"../../Figures/over_time_{feature_name}_{med}_norm3.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/over_time_{feature_name}_{med}_norm3.png", format="png", bbox_inches="tight", transparent=True)

plt.show()