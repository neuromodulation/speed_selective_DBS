# Find the outlier threshold

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "mean_speed"

# Set normalization window and cut-off of interest
cutoff = 5
n_norm = 5

# set medication state
med = "Off"

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

# Load matrix containing the feature values and the stimulated trials
feature_all = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature_all.shape

# Plot as histogram
feature_flat = feature_all.flatten()
plt.figure(figsize=(10, 5))
plt.hist(feature_flat, bins=100, color="dimgrey")

# Calculate the thresholds
thres_max = np.nanmean(feature_flat) + 3 * np.nanstd(feature_flat)
thres_min = np.nanmean(feature_flat) - 3 * np.nanstd(feature_flat)

# Plot thresholds in histogram
plt.axvline(thres_max, color="red")
plt.axvline(thres_min, color="red")
# Add text with values
plt.text(thres_max+100, 100, f"{thres_max:.2f}", color="red")
plt.text(thres_min+100, 100, f"{thres_min:.2f}", color="red")

# Save plot
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.svg", format="svg", bbox_inches="tight", transparent=False)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}.png", format="png", bbox_inches="tight", transparent=False)

plt.show()
