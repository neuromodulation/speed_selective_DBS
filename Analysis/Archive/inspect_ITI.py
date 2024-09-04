# Inspect ITI

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import percentileofscore
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
def statistic(x, y):
    return np.mean(x) - np.mean(y)

# Set analysis parameters
feature_name = "ITI_mean_speed"
med = "OFF"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
n_dataset = len(feature_matrix)

# Plot feature "slow" and "fast" stimulated trials
plt.figure(figsize=(10, 5))
color = "grey"
color_op = "lightgrey"
for i in range(n_dataset):
    x = feature_matrix[i, :, 0, :].flatten()
    # Plot as thin bar
    plt.boxplot(x, positions=[i], patch_artist=True,
       boxprops=dict(facecolor=color_op, color=color_op),
                capprops=dict(color=color_op),
                whiskerprops=dict(color=color_op),
                medianprops=dict(color=color, linewidth=3),
                flierprops=dict(marker='o', markerfacecolor=color_op, markersize=5, markeredgecolor='none'))

# Adjust plot
plt.axhline(300, color="black", linewidth=1)
#plt.text(1, 67, "66 %")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(np.arange(n_dataset)+0.15, labels=np.arange(n_dataset)+1)
plt.ylabel(f"{feature_name}", fontsize=16)
plt.xlabel("Patient Number", fontsize=16)
u.despine()

# Save
plt.savefig(f"../../Figures/{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../Figures/{feature_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()