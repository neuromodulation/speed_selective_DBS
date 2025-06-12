# Results Figure 2: Main effect on average change in speed
# Fast vs slow

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

# Load matrix containing the normalized average speed values (change in respect to beginning of the stimulation block)
speed = np.load("../data/behavior/speed_norm.npy")

# Separate the data for stimulation and recovery block
speed_stim_block = speed[:, :, :-96]
speed_recovery_block = speed[:, :, -96:]

# Prepare plotting
f, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1]}, figsize=(10, 7))
colors = ["#3b0086","#00863b"]
colors_op = ["#b099ce", "#b2dac4"]
labels = ["Fast", "Slow"]
fontsize= 12
box_width = 0.3

# Plot the feature over time (average across patients)
titles = ["Stimulation", "Recovery"]
for block, speed in enumerate([speed_stim_block, speed_recovery_block]):

    # Visualization of patient-averaged speed traces ______________________________________________________________________________________________________________________________
    # Smooth speed traces for plotting
    box = np.ones(5) / 5
    speed_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis=2, arr=speed)
    speed_smooth = speed_smooth[:, :, 2:-2]

    # Calculate the average and standard deviation across subjects
    y = np.nanmean(speed_smooth, axis=0)
    std = np.nanstd(speed_smooth, axis=0)
    x = np.arange(y.shape[-1])

    # Plot average trace and std as shaded area
    ax = axes[int(block*2)]
    for cond in range(2):
        ax.plot(x, y[cond, :], label=labels[cond], color=colors[cond], linewidth=2, alpha=0.8)
        ax.fill_between(x, y[cond, :] - std[cond, :], y[cond, :] + std[cond, :], color=colors_op[cond], alpha=0.3)

    # Adjust plot
    ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
    if block == 0:
        ax.set_ylabel(f"Change in average speed [%]", fontsize=fontsize)
    if block == 0:
        ax.set_xticks([49], ["50"])
    else:
        ax.set_xticks([50], ["140"])
    ax.set_xlabel("Movement number", fontsize=fontsize-2)
    ax.xaxis.set_tick_params(labelsize=fontsize-2)
    ax.yaxis.set_tick_params(labelsize=fontsize-2)
    if block == 0:
        ax.spines[['right', 'top']].set_visible(False)
    else:
        ax.spines[['left', 'top', 'right']].set_visible(False)
        ax.set_yticks([])
    ymax = 50
    ymin = -30
    ax.set_ylim([ymin, ymax])
    ttl = ax.set_title(titles[block], fontsize=fontsize)
    ttl.set_position([0.5, 0.1])
    ax.legend(loc="upper left", fontsize=fontsize-2)

    # Statistical comparison of block-averaged speed values______________________________________________________________________________________________________________________
    ax = axes[int(block * 2)+1]
    bar_pos = [-(box_width * 1.5), (box_width * 1.5)]

    # Calculate the average across a block
    speed_mean = np.nanmean(speed, axis=-1)

    # Plot averaged values in box plot
    for cond in range(2):
        bp = ax.boxplot(x=speed_mean[:, cond],
                    positions=[bar_pos[cond]],
                    widths=box_width,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors_op[cond], color=colors_op[cond]),
                    capprops=dict(color=colors_op[cond]),
                    whiskerprops=dict(color=colors_op[cond]),
                    medianprops=dict(color=colors[cond], linewidth=0),
                    flierprops=dict(marker='.', markerfacecolor=colors_op[cond], markersize=0, markeredgecolor='none')
                    )

    # Add the individual lines
    for speed_mean_patient in speed_mean:
        ax.plot(bar_pos[0], speed_mean_patient[0], marker='o', markersize=0.5, color=colors[0])
        ax.plot(bar_pos[1], speed_mean_patient[1], marker='o', markersize=0.5, color=colors[1])
        # Add line connecting the points
        ax.plot(bar_pos, speed_mean_patient, color="black", linewidth=0.3, alpha=0.3)

    # Add statistics
    res = scipy.stats.permutation_test(data=(speed_mean[:, 0], speed_mean[:, 1]),
                                       statistic=diff_mean_statistic,
                                       n_resamples=100000, permutation_type="samples")
    p = res.pvalue
    text = get_sig_text(p)
    ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax-1, ymax, ymax, ymax-1], color="black", linewidth=1)
    ax.text(0, ymax, text, ha="center", va="bottom", fontsize=fontsize+2)

    # Adjust plot
    ax.set_ylim([ymin, ymax])
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
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