# Plot the outcome measure

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import pandas as pd
from scipy.io import savemat
import numpy as np
from scipy.stats import percentileofscore, pearsonr, spearmanr
import scipy.stats
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('TkAgg')

med = "Off"
feature_name = "mean_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5
# Load matrix containing the outcome measure
x = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
# Load the colors
colors = loadmat("..\Imaging_Analysis\colors_mni.mat")["colors"]
# Plot as bar plot
fontsize= 7
x = x[:, 1]
x = x[np.delete(np.arange(x.shape[0]), 3)]
idx = np.arange(23)
#np.random.shuffle(idx)
x = x[idx]
colors = colors[idx]
f, ax = plt.subplots(1, 1, figsize=(2.2, 1.5))
ax.bar(range(len(x)), x, color=colors)
ax.set_xlabel("Patient #", fontsize=fontsize)
ax.set_ylabel("Stimulation effect [%]", fontsize=fontsize)
#ax.set_title(r"$\bf{Stimulation-induced speed-dependent effect}$", fontsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize-2)
ax.spines[['right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}_{n_norm}_{n_cutoff}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{method}_{med}_{n_norm}_{n_cutoff}.png", format="png", transparent=True, bbox_inches="tight")

plt.show()