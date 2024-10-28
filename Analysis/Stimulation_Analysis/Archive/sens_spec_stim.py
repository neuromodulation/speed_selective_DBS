# Plot the sensitivity and specificity of stimulation

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


# Define medication state
med = "Off"

# Load matrix containing 0/1 indicating which trial was stimulated
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
slow = np.load(f"../../../Data/{med}/processed_data/slow.npy")
fast = np.load(f"../../../Data/{med}/processed_data/fast.npy")

# Select only stimulation blocks
stim = stim[:, :, 0, :]
slow = slow[:, :, 0, :]
fast = fast[:, :, 0, :]

# Compute the sensitivity
sens_slow = np.sum((stim[:, 0, :] == 1) & (slow[:, 0, :] == stim[:, 0, :]), axis=1) / np.sum(slow[:, 0, :] == 1, axis=1)
sens_fast = np.sum((stim[:, 1, :] == 1) & (fast[:, 1, :] == stim[:, 1, :]), axis=1) / np.sum(fast[:, 1, :] == 1, axis=1)

# Compute the specificity
spec_slow = np.sum((stim[:, 0, :] == 0) & (slow[:, 0, :] == stim[:, 0, :]), axis=1) / np.sum(slow[:, 0, :] == 0, axis=1)
spec_fast = np.sum((stim[:, 1, :] == 0) & (fast[:, 1, :] == stim[:, 1, :]), axis=1) / np.sum(fast[:, 1, :] == 0, axis=1)

perf = [np.vstack((sens_slow, sens_fast)), np.vstack((spec_slow, spec_fast))]

plt.figure(figsize=(5.5, 4.5))
for i in range(2):

    # Plot as boxplot
    box_width = 0.3
    bar_pos = [i-(box_width/1.5), i+(box_width/1.5)]
    colors = ["#00863b", "#3b0086"]
    colors_op = ["#b2dac4", "#b099ce"]
    labels = ["Slow", "Fast"]
    bps = []

    for j in range(2):
        x = perf[i].T
        bp = plt.boxplot(x=x[:, j],
                    positions=[bar_pos[j]],
                    widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                    capprops=dict(color=colors_op[j]),
                    whiskerprops=dict(color=colors_op[j]),
                    medianprops=dict(color=colors[j], linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=5, markeredgecolor='none')
                    )
        bps.append(bp)  # Save boxplot for creating the legend

    # Add the individual lines
    for dat in x:
        plt.plot(bar_pos[0], dat[0], marker='o', markersize=2.5, color=colors[0])
        plt.plot(bar_pos[1], dat[1], marker='o', markersize=2.5, color=colors[1])
        # Add line connecting the points
        plt.plot(bar_pos, dat, color="black", linewidth=0.6, alpha=0.3)

    # Add statistics
    r = scipy.stats.permutation_test(data=(x[:, 0], x[:, 1]),
                                     statistic=u.diff_mean_statistic, alternative='two-sided',
                                     n_resamples=100000, permutation_type="samples")
    p = r.pvalue
    sig = "bold" if p < 0.05 else "regular"
    plt.text(i-box_width, np.max(x)+0.01, f"p = {np.round(p, 3)}", weight=sig, fontsize=11)

    # Add legend
    plt.legend([bps[0]["boxes"][0], bps[1]["boxes"][0]], ['Slow', 'Fast'],
               loc='lower center', bbox_to_anchor=(0.95, 0.7),
               prop={'size': 14})

    # Print mean and std values
    print(np.round(np.mean(x, axis=0)*100, 3))
    print(np.round(np.std(x, axis=0)*100, 3))

# Adjust plot
plt.xticks(ticks=[0, 1], labels=["sensitivity", "specificity"], fontsize=13)
plt.yticks(fontsize=14)
plt.ylabel(f"Percentage [%]", fontsize=14)
plt.title(med, fontsize=15)
u.despine()
plt.subplots_adjust(bottom=0.15, left=0.2)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{med}.png", format="png", bbox_inches="tight", transparent=True)


plt.show()