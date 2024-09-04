# Compute and plot the correaltion between a feature (e.g. average peak speed) and full UPDRS, bradykinesia subscore and disease duration

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
med = "Off"
feature_name = "mean_speed"
phenotype_name = "subscore_bradykinesia_total"#"subscore_bradykinesia_total" #  "UPDRS_III"
feature_name_plot = feature_name.replace("_", " ")
n_cutoff = 5
n_norm = 10
method = "mean"

# Prepare plot
fig = plt.figure(figsize=(5.5, 4.5))

# Load the excel sheet containing the phenotype data
df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)

# Reshape matrix such that blocks from one condition are concatenated
feature = feature[:, :, 1, :]

if n_norm != 0:
    # Normalize to average of the first n_norm movements
    feature = u.norm_perc(feature, n_norm=n_norm)

# Compute the average feature of all trials for each subject
if method == "mean":
    y = np.nanmean(feature, axis=(1, 2))
elif method == "median":
    y = np.nanmedian(feature, axis=(1, 2))

# Correlate the average feature with the phenotype

# Get subscore of side with which task was performed
if phenotype_name == "subscore_bradykinesia_used":
    if med == "Off":
        right = df["subscore_bradykinesia_right"].to_numpy()[1:y.shape[0] + 1]
        left = df["subscore_bradykinesia_left"].to_numpy()[1:y.shape[0] + 1]
        # Get the hand which was used
        hand = df["Hand"][1:y.shape[0] + 1].to_numpy()
    else:
        right = df["subscore_bradykinesia_right"].to_numpy()[:y.shape[0]]
        left = df["subscore_bradykinesia_left"].to_numpy()[:y.shape[0]]
        hand = df["Hand"].to_numpy()[:y.shape[0]]
    x = [right[i] if hand[i] == "R" else left[i] for i in range(len(y))]
else:
    if med == "Off":
        x = df[phenotype_name].to_numpy()[1:y.shape[0] + 1]
    else:
        x = df[phenotype_name].to_numpy()[:y.shape[0]]

corr, p = pearsonr(x, y)
#corr, p = u.permutation_correlation(x, y, n_perm=100000, method='pearson')
p = np.round(p, 3)
if p < 0.05:
    label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
else:
    label = f" R = {np.round(corr, 2)} p = {p}"
sb.regplot(x=x, y=y, label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"})


# Adjust plot
plt.legend(loc="upper right", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel(f"Raw speed [pixel/second]", fontsize=15)
#plt.ylabel(f"Average {feature_name_plot}", fontsize=15)
#u.despine()
ax = plt.gca()
ax.spines["top"].set_visible(False)
plt.xlabel(phenotype_name, fontsize=15)
#plt.title(med, fontsize=13)

# Save
plot_name = "phenotype"
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{phenotype_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{phenotype_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()