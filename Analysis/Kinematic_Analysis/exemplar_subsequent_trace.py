# Results Figure 1: Effect on subsequent speed (Exemplar trace)
# Slow vs Fast

# Import useful libraries
import os
import sys
sys.path.insert(1, "../Code")
import utils as u
import numpy as np
from scipy.stats import percentileofscore, pearsonr, spearmanr
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
feature_name = "mean_speed"
n = 3
method = "mean"

# Load matrix containing the feature values
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
# Load the stimulated trials
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
# Load the fast/slow movement
slow = np.load(f"../../../Data/{med}/processed_data/Slow.npy")
fast = np.load(f"../../../Data/{med}/processed_data/Fast.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)

# Loop over datasets
res = np.zeros((n_datasets, 2, n, 2))
for i in range(n_datasets):

    # Loop over conditions
    for j in range(2):

        feature_stim_all = []
        feature_similar_all = []

        # Loop over the n subsequent movements
        for k in range(n):

            # Get the feature and stimulation for one patient
            feature_stim_tmp = feature[i, j, 0, :].flatten()
            stim_tmp = stim[i, j, 0, :].flatten()
            if med == "Healthy":
                if j == 0:
                    stim_tmp = slow[i, j, 0, :].flatten()
                else:
                    stim_tmp = fast[i, j, 0, :].flatten()

            # Get the index of the stimulated movements (same and other condition)
            stim_idx = np.where(stim_tmp == 1)[0]

            # Get the stimulated feature n after the current one
            stim_idx = stim_idx[stim_idx+k < len(feature_stim_tmp)]
            feature_stim_n = feature_stim_tmp[stim_idx + k]
            if k == 0:
                feature_stim_0 = feature_stim_n

            # Get fast/slow movements from recovery blocks (where they were not stimulated)
            feature_similar_tmp = feature[i, :, 1, :].flatten()
            if j == 0:
                slow_tmp = slow[i, :, 1, :].flatten()
                similar_idx = np.where(slow_tmp == 1)[0]
            else:
                fast_tmp = fast[i, :, 1, :].flatten()
                similar_idx = np.where(fast_tmp == 1)[0]
            similar_idx = similar_idx[similar_idx + k < len(feature_similar_tmp)]
            feature_similar_n = feature_similar_tmp[similar_idx + k]
            if k == 0:
                feature_similar_0 = feature_similar_n

            # Calculate the average feature
            tmp_stim = ((feature_stim_n - feature_stim_0[:len(feature_stim_n)])/ feature_stim_0[:len(feature_stim_n)]) * 100
            tmp_similar = ((feature_similar_n - feature_similar_0[:len(feature_similar_n)])/ feature_similar_0[:len(feature_similar_n)]) * 100
            if method == "mean":
                res[i, j, k, 0] = np.nanmean(tmp_stim)
                res[i, j, k, 1] = np.nanmean(tmp_similar)
            elif method == "median":
                res[i, j, k, 0] = np.nanmedian(tmp_stim)
                res[i, j, k, 1] = np.nanmedian(tmp_similar)

# Plot
colors = ["#00863b", "#3b0086", "#203D64"]
colors_op = ["#b2dac4", "#b099ce", "#203D64"]
line_style = ["-", "--"]
hatch = ["", "--"]
desc = [["Slow stimulated", "Slow"], ["Fast stimulated", "Fast"], ["Fast Stimulated - \nSlow Stimulated", "Fast - Slow"]]
conditions = ["Slow", "Fast"]
fontsize = 6
for sub in range(24):
    f, axes = plt.subplots(1, 2, figsize=(1.8, 1.5))
    for i in range(2):
        ax = axes[i]
        if i == 2:
            # Calculate the difference fast-slow for similar and stimulated movements
            res_tmp = res[:, 0, :, :] - res[:, 1, :, :]
        else:
            res_tmp = res[:, i, :, :]

        # Loop over stimulated and not stimuated movements
        for j in range(2):

            mean_res = res_tmp[sub, :, j]
            ax.plot(mean_res, color=colors[i], linestyle=line_style[j], label=desc[i][j], alpha=0.5)

        # Adjust the plot
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xticks([0, 1, 2], [conditions[i], "Opposite", "Same"], rotation=45)
        ax.yaxis.set_tick_params(labelsize=fontsize-2)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        if i == 0:
            ax.set_ylabel(f"Change in average speed [%]", fontsize=fontsize)
    plt.subplots_adjust(wspace=0.5)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{method}_{med}_{sub}.pdf", format="pdf", transparent=True, bbox_inches="tight")
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name[:5]}_{feature_name}_{method}_{med}_{sub}.png", format="png", transparent=True, bbox_inches="tight")

plt.show()
