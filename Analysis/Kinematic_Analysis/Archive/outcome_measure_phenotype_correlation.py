# Compute and plot the correaltion between a phenotype e.g. UPDRS and the outcome measure

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

# Set parameters
phenotype = "Disease duration" #"UPDRS_III" #"subscore_bradykinesia_total" #"Disease duration"
med = "Off"
feature_name = "peak_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load matrix containing the outcome measure
x = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
#x = np.load(f"../../../Data/{med}/processed_data/res_inst_{feature_name}_{method}.npy")
#x = np.vstack((x, x)).T

# Load the excel sheet containing the phenotype data
df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
if med == "Off":
    # Skip the first patient
    y = df[phenotype].to_numpy()[1:x.shape[0] + 1]
else:
    y = df[phenotype].to_numpy()[:x.shape[0]]
#y = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_slow_{method}_{n_norm}_{n_cutoff}.npy")

# Compute correlation and plot for both blocks
plt.figure(figsize=(8, 3.5))
for i, block in enumerate(["Stimulation", "Recovery"]):

    #corr, p = u.permutation_correlation(x[:, i], y, n_perm=10000, method="pearson")
    corr, p = pearsonr(x[:, i], y)
    p = np.round(p, 3)
    if p < 0.05:
        label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
    else:
        label = f" R = {np.round(corr, 2)} p = {p}"
    plt.subplot(1, 2, i+1)
    sb.regplot(x=x[:, i], y=y, label=label, scatter_kws={"color": "indianred"}, line_kws={"color": "teal"})

    # Adjust plot
    plt.legend(loc="upper right", fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(f"{feature_name} \n difference Slow-Fast", fontsize=12)
    plt.title(block, fontsize=12)
    u.despine()
    plt.ylabel(phenotype, fontsize=13)

# Adjust figure
plt.subplots_adjust(wspace=0.3, top=0.8, bottom=0.2)
plt.suptitle(f"{med} {mode} {method} {n_norm} {n_cutoff}", fontsize=13, y=0.97)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/corr_{phenotype[:3]}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.svg",
            format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/corr_{phenotype[:3]}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.png",
            format="png", bbox_inches="tight", transparent=True)
plt.show()