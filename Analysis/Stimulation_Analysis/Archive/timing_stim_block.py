# Plot the distribution of stimulated trials over the whole block (96 movements)

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


med = "Off"
fig = plt.figure(figsize=(8.5, 4))

# Load matrix containing 0/1 indicating which trial was stimulated
stim = np.load(f"../../../Data/{med}/processed_data/fast.npy")

# Select only stimulation blocks
stim = stim[:, :, 0, :]

# Get % of stimulated movements per trial over participants
stim_perc = np.mean(stim, axis=0) * 100

# Plot as barplot
bar_width = 0.5
bar_pos = np.arange(stim_perc.shape[1])
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]
bps = []
for j in range(2):
    height = stim_perc[j, :] if j == 0 else stim_perc[j, :] * -1
    plt.bar(x=bar_pos, height=height, color=colors_op[j], label=labels[j], width=bar_width, alpha=1)

# Add legend
plt.legend(loc="upper right", prop={'size': 14})

# Adjust subplot
plt.title(med, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(ticks=[-60, -40, -20, 0, 20, 40, 60], labels=[60, 40, 20, 0, 20, 40, 60], fontsize=14)
plt.ylabel(f"Percentage of stimulated \n movements (out of N={stim.shape[0]}) [%]", fontsize=15)
plt.xlabel("Movement number", fontsize=15)
u.despine()

# Adjust figure
plt.subplots_adjust(bottom=0.15, left=0.15, wspace=0.45, top=0.85)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.png", format="png", bbox_inches="tight", transparent=True)

plt.show()