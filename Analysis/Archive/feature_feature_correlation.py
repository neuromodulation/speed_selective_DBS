# Compute and plot the correaltion between different features

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set feature to analyze
feature_names = ["peak_speed", "peak_speed_next"]
feature_name_plot = [feature_name.replace("_", " ") for feature_name in feature_names]

# Loop over On and Off medication datasets
meds = ["Off", "On"]
fig = plt.figure(figsize=(9, 3))

# Plot once with raw peak speed values and once with percentiles in the block
for i, med in enumerate(meds):

    plt.subplot(1, 2, i+1)

    # Load matrix containing the feature values
    feature_1 = np.load(f"../../../Data/{med}/processed_data/{feature_names[0]}.npy")
    feature_2 = np.load(f"../../../Data/{med}/processed_data/{feature_names[1]}.npy")

    # Detect and fill outliers (e.g. when subject did not touch the screen)
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_1)
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_2)

    # Compute the correlation
    x = feature_1.flatten()
    y = feature_2.flatten()
    corr, p = spearmanr(x, y, nan_policy='omit')
    p = np.round(p, 3)
    if p < 0.05:
        label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
    else:
        label = f" R = {np.round(corr, 2)} p = {p}"
    sb.regplot(x=x, y=y, label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"})

    # Adjust plot
    plt.title(med, fontsize=14)
    plt.legend(loc="upper right", fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(feature_name_plot[0], fontsize=14)
    plt.ylabel(feature_name_plot[1], fontsize=14)
    u.despine()

# Adjust figure
plt.subplots_adjust(bottom=0.15, left=0.15, wspace=0.4)


# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_names[0]}_{feature_names[1]}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_names[0]}_{feature_names[1]}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()