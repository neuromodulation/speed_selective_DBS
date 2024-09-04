# Correlate the connectivity computed with Lead DBS/mapper with an outcome measure

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
med = "Off"
Fz = True
feature_name = "mean_speed"
feature_name_plot = feature_name.replace("_", " ")
mode = "diff"
cutoff = True
norm = True
smooth = True
cutoff = 5
n_norm = 5
smooth_window = 5
cutoff_name = f"cut_{cutoff }" if cutoff else ""
norm_name = f"norm_{n_norm}" if norm else ""
smooth_name = f"smooth_{smooth_window}" if smooth else ""
av = "Median"

# Load matrix containing the outcome measure
x = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{cutoff_name}_{norm_name}_{smooth_name}_{av}.npy")
# Delete subject 3 (different electrode type)
x = x[np.delete(np.arange(24), 3), :]
#x = x[np.delete(np.arange(23), 15)]

# Load the mat file containing the connectivity values
if Fz:
    conn_mat = mat73.loadmat('GPe_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat')
else:
    conn_mat = mat73.loadmat('GPe_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat')
conn_mat = conn_mat["X"]

# Define the regions of interest (in order according to the text file based on which the connectivity matrix was created)
targets = ["GPe", "GPi", "Primary motor cortex", "Putamen", "Supp_Motor_Area"]

"""
# Load excel sheet containing the information of the block order
df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
slow_first = df["Block order"].to_numpy()[1:24 + 1]
slow_first = slow_first[np.delete(np.arange(24), 3)]
slow_fast = np.where(slow_first == "Slow")[0]
fast_slow = np.where(slow_first == "Fast")[0]"""

plt.figure(figsize=(10, 7))
for i, target in enumerate(targets):

    # Loop over blocks
    for j in range(2):
        x_block = x[:, j]
        #x_block = x[fast_slow, j]
        # Calculate the correlation
        y = conn_mat[i, len(targets):]
        #y = y[fast_slow]
        #y = y[np.delete(np.arange(23), 15)]
        #x = x[:len(y)]
        corr, p = spearmanr(x_block, y, nan_policy='omit')
        p = np.round(p, 3)
        if p < 0.05:
            label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
        else:
            label = f" R = {np.round(corr, 2)} p = {p}"

        # Plot
        plt.subplot(2, len(targets), len(targets) * j + i + 1)
        sb.regplot(x=x_block, y=y, label=label, scatter_kws={"color": "indianred"}, line_kws={"color": "teal"})

        # Adjust plot
        plt.legend(loc="upper right", fontsize=11)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel(f"{av} {feature_name_plot} \n {mode}", fontsize=12)
        block_name = "Stim" if j == 0 else "Recovery"
        plt.title(f"{target} {block_name}", fontsize=12)
        u.despine()
        plt.ylabel("Functional connectivity", fontsize=13)

# Adjust figure
plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9, bottom=0.15, left=0.1, right=0.9)
plt.suptitle(f"{med} {cutoff_name} {norm_name} {smooth_name}", fontsize=13, y=0.97)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/corr_{feature_name}_{mode}_{av}_{cutoff_name}_{norm_name}_{smooth_name}.svg",
            format="svg", bbox_inches="tight", transparent=False)
plt.savefig(f"../../../Figures/{dir_name}/corr_{feature_name}_{mode}_{av}_{cutoff_name}_{norm_name}_{smooth_name}.png",
            format="png", bbox_inches="tight", transparent=False)
plt.show()
