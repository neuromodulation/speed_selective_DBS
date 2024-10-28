# Calculate/plot thw sampling frequency of the behavioral data (acquired with psychtoolbox)

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import scipy.stats
from os.path import dirname as up
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Define folder of raw data
data_folder = up(up(up(up(__file__))))+"/Data/"

# Loop over On and Off medication datasets
meds = ["Off", "On"]
fig = plt.figure(figsize=(10.5, 4))

for i, med in enumerate(meds):

    # Get all raw files for the medication
    root = f"{data_folder}{med}/raw_data/"
    files_list = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.mat'):
                files_list.append(os.path.join(root, file))

    # Loop over all files in folder
    plt.subplot(1, 2, i+1)
    for j, file in enumerate(files_list):

        # Load behavioral data
        data = loadmat(file)
        data = data["struct"][0][0][1]

        # Calculate the time difference between the samples
        samp_diff = np.diff(data[:, 2])

        # Replace outliers with nan
        samp_diff = u.fill_outliers_nan(samp_diff)
        # Delete nans
        samp_diff = samp_diff[~np.isnan(samp_diff)]


        print(np.median(samp_diff))

        # Plot the histogram of the differences
        color_op = "lightgrey"
        color = "plum"
        plt.boxplot(x=samp_diff, positions=[j], widths=0.2,
                    patch_artist=True,
                    boxprops=dict(facecolor=color_op, color=color_op),
                    capprops=dict(color=color_op),
                    whiskerprops=dict(color=color_op),
                    medianprops=dict(color=color, linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor=color_op, markersize=1, markeredgecolor='none')
                    )

    # Adjust subplot
    plt.title(med, fontsize=16)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=14)
    plt.ylabel(f"Time difference \n between samples [s]", fontsize=15)
    plt.xlabel("Patient", fontsize=15)
    u.despine()

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.1)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()

