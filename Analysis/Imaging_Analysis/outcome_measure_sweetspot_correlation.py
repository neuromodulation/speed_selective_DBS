# Correlate the distance to the STN sweetspot reported in the literature with an outcome measure

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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
sweetspot_paper = "Horn et al (2017)" #"Bot et al (2016)"# "Akram et al (2017)" #"Horn et al (2017)" "Bot et al (2016)"
med = "Off"
feature_name = "peak_dec"
mode = "diff"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load matrix containing the outcome measure
X = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")

# Load the excel sheet containing the subjects coordinates
path_table = "C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - " \
             "PROJECT ReinforceVigor\\vigor_stim_task\\Data\\Dataset_list_Off.xlsx"
df_info = pd.read_excel(path_table)
mni_stim = df_info.iloc[1:25, -6:]
mni_stim = mni_stim.to_numpy()
mni_stim_r = mni_stim[:, :3]
mni_stim_l = mni_stim[:, 3:]

# Load the excel sheet containing the sweetspot coordinates as reported in the literature
df = pd.read_excel("atlas/STN_sweetspots.xlsx")
df_row = df[df["Authors"] == sweetspot_paper]
mni_ss_r = df_row.iloc[0, 1:4].to_numpy()
mni_ss_l = df_row.iloc[0, 4:].to_numpy()

# Compute the distance to the sweetspot for each subject and side
d_stim_r = np.linalg.norm((mni_stim_r - mni_ss_r).astype(float), axis=1)
d_stim_l = np.linalg.norm((mni_stim_l - mni_ss_l).astype(float), axis=1)

# Seperate right and left in contra and ipsilateral depending on the hand that was used
hand = df_info["Hand"][1:25].to_numpy()
d_ipsi = [d_stim_r[i] if hand[i] == "R" else d_stim_l[i] for i in range(len(d_stim_r))]
d_contra = [d_stim_r[i] if hand[i] == "L" else d_stim_l[i] for i in range(len(d_stim_r))]

# Correlate
plt.figure(figsize=(9, 7))
# Loop over blocks
blocks = ["Stimulation", "Recovery"]
sides = ["Contralateral", "Ipsilateral"]
for i in range(2):
    x = X[:, i]

    # Loop over side
    for j in range(2):
        y = d_contra if j == 0 else d_ipsi

        # Compute and plot corralation
        corr, p = spearmanr(x, y, nan_policy='omit')

        p = np.round(p, 3)
        if p < 0.05:
            label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
        else:
            label = f" R = {np.round(corr, 2)} p = {p}"
        plt.subplot(2, 2, i * 2 + j + 1)
        sb.regplot(x=x, y=y, label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"})

        # Adjust plot
        plt.legend(loc="upper right", fontsize=11)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel(f"Average {mode} {feature_name}", fontsize=12)
        u.despine()
        plt.ylabel("Distance to STN sweetspot", fontsize=13)
        plt.title(f"{blocks[i]} {sides[j]}", fontsize=13)

# Adjust figure
plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.85, bottom=0.15, left=0.1, right=0.9)
plt.suptitle(f"{sweetspot_paper} {med}", fontsize=13, y=0.97)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/corr_ss_{feature_name}_{mode}_{method}.svg",
            format="svg", bbox_inches="tight", transparent=False)
plt.savefig(f"../../../Figures/{dir_name}/cor_ss_{feature_name}_{mode}_{mode}_{method}.png",
            format="png", bbox_inches="tight", transparent=False)
plt.show()

