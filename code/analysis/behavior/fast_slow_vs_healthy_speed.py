# Results Figure 2: Main effect on average change in speed
# Slow/Fast vs Healthy

# Prepare environment
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Define helper functions
def diff_mean_statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def get_sig_text(p):
    if p < 0.001:
        text = "***"
    elif p < 0.01:
        text = "**"
    elif p < 0.05:
        text = "*"
    else:
        text = "n.s."
    return text

# Prepare plotting
colors = ["#3b0086", "#00863b", "dimgrey"]
colors_op = ["#b099ce", "#b2dac4", "grey"]
labels = ["Fast", "Slow", "Healthy"]
titles = ["Stimulation", "Recovery"]
f, axes = plt.subplots(2, 4, figsize=(8, 8))
fontsize = 12
box_width = 0.3

# Trace over time
meds = ["Fast", "Slow", "Healthy"]
for i, med in enumerate(meds):

    # Load matrix containing the normalized average speed values (change in respect to beginning of the stimulation block)
    if med == "Healthy":
        speed = np.load("../data/behavior/speed_norm_healthy.npy")
    else:
        speed = np.load("../data/behavior/speed_norm.npy")

    # Separate the data for stimulation and recovery block
    speed_stim_block = speed[:, :, :-96]
    speed_recovery_block = speed[:, :, -96:]

    for block, speed in enumerate([speed_stim_block, speed_recovery_block]):

        # Visualization of patient-averaged speed traces ______________________________________________________________________________________________________________________________
        
        # Smooth speed traces for plotting
        box = np.ones(5) / 5
        speed_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis=2, arr=speed)
        speed_smooth = speed_smooth[:, :, 2:-2]

        # Calculate the average and standard deviation across subjects
        # For healthy subject average across "stimulation conditions"
        if med == "Healthy":
            y = np.nanmean(speed_smooth, axis=(0, 1))
            std = np.nanstd(speed_smooth, axis=(0, 1))
        else:
            y = np.nanmean(speed_smooth, axis=0)[i, :]
            std = np.nanstd(speed_smooth, axis=0)[i, :]
        x = np.arange(y.shape[-1])

        # Plot average trace and std as shaded area
        if i < 2:
            ax = axes[i, int(block * 2)]
            ax.plot(x, y, label=labels[0], color=colors[i], linewidth=2, alpha=0.8)
            ax.axhline(0, linewidth=1, color="black", linestyle="dashed")
            ax.fill_between(x, y-std, y+std, color=colors_op[i], alpha=0.5)
        else:
            for j in range(2):
                ax = axes[j, int(block * 2)]
                ax.plot(x, y, label=labels[0], color=colors[i], linewidth=2, alpha=0.8)  # Add line at y=0 and x=96
                ax.axhline(0, linewidth=1, color="black", linestyle="dashed")
                ax.fill_between(x, y - std, y + std, color=colors_op[i], alpha=0.25)
    
        # Adjust plot
        ax.set_xticks([])
        ax.xaxis.set_tick_params(labelsize=fontsize - 2)
        ax.yaxis.set_tick_params(labelsize=fontsize - 2)
        if block == 0:
            ax.spines[['right', 'top']].set_visible(False)
        else:
            ax.spines[['left', 'top', 'right']].set_visible(False)
            ax.set_yticks([])
        ymax = 30
        ymin = -30
        ax.set_ylim([ymin, ymax])
        if i == 0:
            ax.set_title(titles[block], fontsize=fontsize)
    axes[1,0].set_xlabel("Movement number", fontsize=fontsize - 2)
    axes[1,2].set_xlabel("Movement number", fontsize=fontsize - 2)
    axes[1,0].set_xticks([49], ["50"])
    axes[1,2].set_xticks([50], ["140"])
    axes[1,0].set_ylabel(f"Change in average speed [%]", fontsize=fontsize)
    axes[1,0].yaxis.set_label_coords(-0.6, 1)


# Statistical comparison of block-averaged speed values______________________________________________________________________________________________________________________
speed_healthy = np.load("../data/behavior/speed_norm_healthy.npy")
speed = np.load("../data/behavior/speed_norm.npy")

# Separate the data for stimulation and recovery block
speed_stim_block = speed[:, :, :-96]
speed_recovery_block = speed[:, :, -96:]
speed_stim_block_healthy = speed_healthy[:, :, :-96]
speed_recovery_block_healthy = speed_healthy[:, :, -96:]

for block, (speed, speed_healthy) in enumerate(zip([speed_stim_block, speed_recovery_block], 
                                                   [speed_stim_block_healthy, speed_recovery_block_healthy])):

    # Calculate the average across a block (for healthy subjects average "stimulation" conditions, as no dbs was applied)
    speed_mean = np.nanmean(speed, axis=-1)
    speed_mean_healthy = np.nanmean(speed_healthy, axis=(1, 2))

    for cond in range(2):
        ax = axes[cond, int(block * 2) + 1]
        bar_pos = [-(box_width * 1.5), (box_width * 1.5)]

        speed_mean_all = [speed_mean[:, cond], speed_mean_healthy]

        if cond == 0:
            colors = ["#3b0086", "dimgrey"]
            colors_op = ["#b099ce", "grey"]
        else:
            colors = ["#00863b", "dimgrey"]
            colors_op = ["#b2dac4", "grey"]

        # Plot averaged values in box plot
        for j, speed_tmp in enumerate(speed_mean_all):
            ax.boxplot(x=speed_tmp,
                            positions=[bar_pos[j]],
                            widths=box_width,
                            patch_artist=True,
                            boxprops=dict(facecolor=colors_op[j], color=colors_op[j]),
                            capprops=dict(color=colors_op[j]),
                            whiskerprops=dict(color=colors_op[j]),
                            medianprops=dict(color=colors[j], linewidth=0),
                            showmeans=True,
                            meanline=True,
                            meanprops=dict(color=colors[j], linewidth=0, linestyle="solid"),
                        flierprops=dict(marker='o', markerfacecolor=colors_op[j], markersize=1.5, markeredgecolor='none')
                            )
            # Add the individual subject points
            for speed_mean_subject in speed_tmp:
                ax.plot(bar_pos[j], speed_mean_subject, marker='o', markersize=1.5, color=colors[j])

        # Add statistics
        res = scipy.stats.permutation_test(data=(speed_mean_all[0], speed_mean_all[1]),
                                        statistic=diff_mean_statistic,
                                        n_resamples=100000, permutation_type="independent",)
        p = res.pvalue
        text = get_sig_text(p)
        ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 1, ymax, ymax, ymax - 1], color="black",
                linewidth=1)
        ax.text(0, ymax, text, ha="center", va="bottom", fontsize=fontsize-1)

        # Adjust plot
        ax.set_ylim([ymin, ymax])
        ax.xaxis.set_tick_params(labelsize=fontsize-2)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
        ax.set_xticks([])
        ax.set_yticks([])

# Adjust plot
plt.subplots_adjust(bottom=0.15, left=0.15, top=0.8, wspace=0.01)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
plt.savefig(f"../figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../figures/{plot_name}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()