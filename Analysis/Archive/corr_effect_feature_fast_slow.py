# Script for statistical analysis of feature
# Correlate the stimulation effect (diff fast-slow) with the percentile of the stimulated movements

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
from scipy.stats import pearsonr, spearmanr, percentileofscore
import seaborn as sb
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# Set analysis parameters
feature_name = "peak_speed"
med = "OFF"  # "on", "off", "all"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
fast = np.load(f"../../Data/{med}/processed_data/fast.npy")
slow = np.load(f"../../Data/{med}/processed_data/slow.npy")
stim = np.load(f"../../Data/{med}/processed_data/stim.npy")

# Select the dataset of interest
mask = np.arange(feature_matrix.shape[0])
mask = mask[(mask != 0)]
feature_matrix = feature_matrix[mask, :, :, :]
stim = stim[mask, :, :, :]
fast = fast[mask, :, :, :]
slow = slow[mask, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Delete outliers
np.apply_along_axis(lambda m: u.fill_outliers_nan(m, threshold=3), axis=3, arr=feature_matrix)
#np.apply_along_axis(lambda m: utils.fill_outliers_nan(m, threshold=3), axis=3, arr=feature_matrix)

# Loop over stimulation and recovery blocks
block_names = ["Stimulation", "Recovery"]
fig = plt.figure(figsize=(6, 5))
colors = ["#763c29", "sandybrown"]
for block in [1, 0]:

    # Select only the block of interest and delete the first 5 movements
    feature_matrix_tmp = feature_matrix[:, :, block, 5:]
    stim_tmp = stim[:, :, block, 5:]
    fast_tmp = fast[:, :, block, 5:]
    slow_tmp = slow[:, :, block, 5:]

    # Normalize to average of next 5 movements
    feature_matrix_tmp = u.norm_perc(feature_matrix_tmp)

    # Get the difference in peak speed between the conditions for all datasets
    diff_effect = np.nanmean(feature_matrix_tmp[:, 1, :], axis=1) - np.nanmean(feature_matrix_tmp[:, 0, :], axis=1)

    # Get the difference between percentile of Slow vs Fast movements for all datasets
    diff_percentile_stim = np.zeros(n_datasets)

    for i in range(n_datasets):
        # Select stimulated movements if block == 1 and post-hoc determined slow/fast movements for block == 1
        if block == 0:
            diff_percentile_stim[i] = np.nanmedian([percentileofscore(feature_matrix_tmp[i, 1, :], x, nan_policy='omit')
                                                    for x in feature_matrix_tmp[i, 1, :][stim_tmp[i, 1, :] == 1]]) - \
                                     np.nanmedian([percentileofscore(feature_matrix_tmp[i, 0, :], x, nan_policy='omit')
                                                   for x in feature_matrix_tmp[i, 0, :][stim_tmp[i, 0, :] == 1]])
        else:
            diff_percentile_stim[i] = np.nanmedian([percentileofscore(feature_matrix_tmp[i, 1, :], x, nan_policy='omit')
                                                    for x in feature_matrix_tmp[i, 1, :][fast_tmp[i, 1, :] == 1]]) - \
                                      np.nanmedian([percentileofscore(feature_matrix_tmp[i, 0, :], x, nan_policy='omit')
                                                    for x in feature_matrix_tmp[i, 0, :][slow_tmp[i, 0, :] == 1]])
    # Compute correlation coefficient and plot
    #plt.subplot(1, 2, block+1)
    x = diff_effect
    y = diff_percentile_stim
    corr, p = spearmanr(x, y, nan_policy='omit')
    p = np.round(p, 3)
    if p < 0.05:
        sb.regplot(x=x, y=y, label=f"{block_names[block]}: r = {np.round(corr, 2)} "+"$\\bf{p=}$"+f"$\\bf{p}$", color=colors[block])
    else:
        sb.regplot(x=x, y=y, label=f"{block_names[block]}: r = {np.round(corr, 2)} " + f"p={p}",
                   color=colors[block])
    #plt.title(f"corr = {np.round(corr, 2)}, p = {np.round(p, 3)}", fontweight='bold')
    feature_name_space = feature_name.replace("_", " ")
    plt.xlabel(f"Difference of mean peak \n speed [Fast - Slow %]", fontsize=20)
    plt.ylabel(f"Difference of peak speed \n of stimulated movements \n[Fast - Slow percentile]", fontsize=20)

plt.subplots_adjust(left=0.25, bottom=0.2)
u.despine()
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
legend = plt.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.1), prop={'size': 16}, handlelength=0, markerscale=0)
legend.get_frame().set_alpha(0)
colors = ["sandybrown", "#763c29"]
for i, text in enumerate(legend.get_texts()):
    text.set_color(colors[i])

# Save figure on group basis
plt.savefig(f"../../Figures/corr_effect_stim_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)

plt.show()