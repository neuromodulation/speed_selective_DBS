# Plot the % of stimulated movements

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


# define medication state
med = "Of"
fig = plt.figure(figsize=(5.5, 5.5))

# Load matrix containing 0/1 indicating which trial was stimulated
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")

# Select only stimulation blocks
stim = stim[:, :, 0, :]

# Get % of stimulated movements
stim_perc = (np.sum(stim, axis=2) / stim.shape[2]) * 100

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

# Plot as boxplot
box_width = 0.3
bar_pos = [1, 1.5]

for i in range(2):
    bp = plt.boxplot(x=stim_perc[:, i],
                positions=[bar_pos[i]],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=colors_op[i], color=colors_op[i]),
                capprops=dict(color=colors_op[i]),
                whiskerprops=dict(color=colors_op[i]),
                medianprops=dict(color=colors[i], linewidth=2),
                flierprops=dict(marker='o', markerfacecolor=colors_op[i], markersize=5, markeredgecolor='none')
                )

# Add the individual lines
for dat in stim_perc:
    plt.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
    plt.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
    # Add line connecting the points
    plt.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

# Add statistics
r = scipy.stats.permutation_test(data=(stim_perc[:, 0], stim_perc[:, 1]),
                                 statistic=u.diff_mean_statistic, alternative='two-sided',
                                 n_resamples=100000, permutation_type="samples")
p = r.pvalue
sig = "bold" if p < 0.05 else "regular"
plt.text(np.mean(bar_pos), np.max(stim_perc), f"p = {np.round(p, 3)}", weight=sig, fontsize=12)

# Adiust plot
plt.xticks(bar_pos, ["Slow", "Fast"], fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(f"Percentage of stimulated movements \n (out of 96) [%]", fontsize=15)
plt.ylim([15, 50])
plt.title(med, fontsize=15)
u.despine()
plt.subplots_adjust(bottom=0.15, left=0.2)


# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)

# Print mean and std values
print(np.round(np.mean(stim_perc, axis=0), 3))
print(np.round(np.std(stim_perc, axis=0), 3))

plt.show()