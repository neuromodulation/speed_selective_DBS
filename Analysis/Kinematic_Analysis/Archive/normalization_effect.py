# Analyze the influence of the cut-off and normalization window on the difference
# between Slow and Fast features (average)

# Import toolboxes
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
matplotlib.use('TkAgg')

# Set feature and medication to analyze
feature_name = "mean_speed"
med = "Off"
method = "mean"

# Define normalization window and cut-off of interest
cutoff_array = np.arange(0, 20)
n_norm_array = np.arange(5, 20)

# Initialize matrix storing p-values for all parameter combinations (for stim and recovery)
res = np.zeros((len(cutoff_array), len(n_norm_array), 3, 2))

# Load matrix containing the feature values
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

# Prepare plotting
feature_name_plot = feature_name.replace("_", " ")

# Loop over both variables
for i, cutoff in enumerate(cutoff_array):
    for j, n_norm in enumerate(n_norm_array):

        # Cutoff
        feature_cut = feature[:, :, cutoff:]

        # Normalize to average of the next 5 movements
        feature_norm = u.norm_perc(feature_cut, n_norm=n_norm)

        feature_norm_window = np.nanmean(feature_cut[:, :, :n_norm], axis=-1)
        if method == "mean":
            feature_stim = np.nanmean(feature_norm[:, :, n_norm:n_trials-cutoff], axis=-1)
            feature_recov = np.nanmean(feature_norm[:, :, n_trials-cutoff:], axis=-1)
        elif method == "median":
            feature_stim = np.nanmedian(feature_norm[:, :, n_norm:n_trials-cutoff], axis=-1)
            feature_recov = np.nanmedian(feature_norm[:, :, n_trials-cutoff:], axis=-1)
        elif method == "mean_median":
            feature_stim = np.nanmean(np.stack((np.nanmedian(feature_norm[:, :, n_norm:n_trials-cutoff], axis=-1),
                                                 np.nanmedian(feature_norm[:, :, n_norm:n_trials-cutoff], axis=-1))), axis=0)
            feature_recov = np.nanmean(np.stack((np.nanmedian(feature_norm[:, :, n_trials-cutoff:], axis=-1),
                                                 np.nanmedian(feature_norm[:, :, n_trials-cutoff:], axis=-1))), axis=0)
        feature_av = [feature_stim, feature_recov, feature_norm_window]

        # Loop over stimulation and recovery block and compute statistics
        for k in range(3):

            # Compute average difference
            res[i, j, k, 0] = np.mean(feature_av[k][:, 0]) - np.mean(feature_av[k][:, 1])

            # Add statistics
            z, p = scipy.stats.wilcoxon(x=feature_av[k][:, 0], y=feature_av[k][:, 1])
            res[i, j, k, 1] = p

# Plot result
plt.figure(figsize=(12, 4))
titles = ["Stimulation", "Recovery", "Normalization window"]
for i in range(3):
    plt.subplot(1, 3, i+1)
    norm = colors.TwoSlopeNorm(vcenter=0)
    plt.imshow(res[:, :, i, 0], aspect="auto", cmap="seismic")
    # Add starts in significant positions
    for j in range(len(cutoff_array)):
        for k in range(len(n_norm_array)):
            if res[j, k, i, 1] < 0.05:
                plt.plot(k, j, "*", color="yellow")
    plt.title(titles[i], fontsize=13)
    plt.yticks(range(len(cutoff_array)), cutoff_array)
    plt.xticks(range(len(n_norm_array)), n_norm_array)
    plt.xlabel("# samples normalization", fontsize=12)
    plt.ylabel("# samples cutoff", fontsize=12)
    cbar = plt.colorbar()
    cbar.set_label(f'Average {feature_name_plot} difference')
    if i < 2:
        plt.clim(vmax=8, vmin=-8)
    else:
        plt.clim(vmax=300, vmin=-300)

# Adjust plot
plt.subplots_adjust(wspace=0.3, top=0.83, bottom=0.2, left=0.05, right=0.95)
plt.suptitle(med, fontsize=16, y=0.95)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}_{method}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{med}_{method}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()