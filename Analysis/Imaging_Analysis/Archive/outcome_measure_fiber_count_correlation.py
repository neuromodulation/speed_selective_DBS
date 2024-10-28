# Correlate the number of the fibers with an outcome measure

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import mat73
from scipy.stats import percentileofscore
import scipy.stats
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
Fz = True
feature_name = "peak_speed"
mode = "diff"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load matrix containing the outcome measure
X = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
# Delete subject 3 (different electrode type)
X = X[np.delete(np.arange(24), 3), :]

# Load the excel sheet with information on the used hand
path_table = "C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - " \
             "PROJECT ReinforceVigor\\vigor_stim_task\\Data\\Dataset_list_Off.xlsx"
df_info = pd.read_excel(path_table)
hand = df_info["Hand"][1:25].to_numpy()
hand = hand[np.delete(np.arange(24), 3)]

# Load the fiber counts
fiber_counts = loadmat("fiber_count_VTA.mat")['res']

# Seperate right and left in contra and ipsilateral depending on the hand that was used
n_ipsi = np.array([fiber_counts[i, 0, :] if hand[i] == "R" else fiber_counts[i, 1, :] for i in range(len(fiber_counts))])
n_contra = np.array([fiber_counts[i, 0, :] if hand[i] == "L" else fiber_counts[i, 1, :] for i in range(len(fiber_counts))])
n_av = np.mean(fiber_counts, axis=1)
n = [n_contra, n_ipsi, n_av]

# Correlate and plot

# Loop over blocks
blocks = ["Stimulation", "Recovery"]
fibers = ["GPe-STN", "STN-GPi"]
sides = ["Contralateral", "Ipsilateral", "Average"]
for i in range(2):

    x = X[:, i]
    # One figure per block
    plt.figure(figsize=(8, 9))

    # Loop over side/sum/average
    for j in range(len(n)):

        # Loop over fiber type (GPe-STN, STN-GPi)
        for k in range(n_av.shape[-1]):

            # Get the fiber count
            y = n[j][:, k]
            #y = y[np.delete(np.arange(23), 15)]

            # Compute and plot corralation
            corr, p = spearmanr(x, y, nan_policy='omit')

            p = np.round(p, 3)
            if p < 0.05:
                label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
            else:
                label = f" R = {np.round(corr, 2)} p = {p}"
            plt.subplot(len(n), n_av.shape[-1], j * n_av.shape[-1] + k + 1)
            sb.regplot(x=x, y=y, label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"})

            # Adjust plot
            plt.legend(loc="upper right", fontsize=11)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            #plt.xlabel(f"Average {mode} {feature_name_plot}", fontsize=12)
            u.despine()
            plt.ylabel(f"Fiber count \n{sides[j]} \n{fibers[k]}", fontsize=13)
            #plt.title(f"{sides[j]} {fibers[k]}", fontsize=13)

    # Adjust figure
    plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9, bottom=0.15, left=0.15, right=0.9)
    plt.suptitle(f"{blocks[i]} {med}", fontsize=13, y=0.97)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/fiber_corr_{blocks[i]}_{feature_name}_{mode}_{method}.svg",
                format="svg", bbox_inches="tight", transparent=False)
    plt.savefig(f"../../../Figures/{dir_name}/fiber_corr_{blocks[i]}_{feature_name}_{mode}_{method}.png",
                format="png", bbox_inches="tight", transparent=False)
plt.show()

