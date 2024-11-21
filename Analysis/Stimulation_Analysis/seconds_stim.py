# Get teh total stimulation time in seis

# Import useful libraries
import os
import sys
sys.path.append('../../../Code')
import utils as u
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

med = "Off"

# Load matrix containing 0/1 indicating which trial was stimulated
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")

# Select only stimulation blocks
stim = stim[:, :, 0, :]

# Get total sec of stimulated movements
stim_sec = ((np.sum(stim, axis=2)) * 300) / 1000

# Print mean and std values
print(np.round(np.mean(stim_sec, axis=0), 3))
print(np.round(np.std(stim_sec, axis=0), 3))

# Prepare plotting
fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
box_width = 0.25
bar_pos = [-(box_width), (box_width)]
colors = np.array(["#860008", "darkgrey"])
colors_op = np.array(["#860008", "darkgrey"])
labels = ["DBS", "Block \nduration"]
fontsize = 6

# Load matrix containing the first time point of each trial (to get an estimate of the length of one block)
time_sample = np.load(f"../../../Data/{med}/processed_data/first_time_sample.npy")
# Select only stimulation blocks
time_sample = time_sample[:, :, 0, :]
av_stim_sec = [9.44, 7.56]
av_dur = (np.max(time_sample, axis=-1) - np.min(time_sample, axis=-1)).flatten() / 60
stim_sec = stim_sec.flatten() / 60
for i, av in enumerate([stim_sec, av_dur]):
    print(np.round(np.mean(av), 2))
    print(np.round(np.std(av), 2))
    # Plot axis on both sides
    bp = ax.boxplot(x=av,
                positions=[bar_pos[i]],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=colors_op[i], color=colors_op[i]),
                capprops=dict(color=colors_op[i]),
                whiskerprops=dict(color=colors_op[i]),
                medianprops=dict(color=colors[i], linewidth=1),
                flierprops=dict(marker='o', markerfacecolor=colors_op[i], markersize=5, markeredgecolor='none')
                )
    # Add the individual points
    ymin, ymax = ax.get_ylim()
    ax.scatter(np.repeat(bar_pos[i], len(av)), av, s=1, c="dimgray", marker='o', zorder=2)
# Adjust plot
ax.set_ylim([ymin, ymax])
ax.yaxis.set_tick_params(labelsize=fontsize)
ax.set_xticklabels(labels, fontsize=fontsize)
ax.set_ylabel("Time [min]", fontsize=fontsize)
ax.spines[["top", "right"]].set_visible(False)
plt.subplots_adjust(bottom=0.15, left=0.15, right=0.85, wspace=0.4)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.png", format="png", bbox_inches="tight", transparent=True)

# Creating plot
fig = plt.figure(figsize=(10, 7))
plt.pie([95, 5])
# show plot
# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_pie.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_pie.png", format="png", bbox_inches="tight", transparent=True)
plt.show()