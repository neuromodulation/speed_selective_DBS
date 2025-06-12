# Results Figure 2: Effect on subsequent speed
# Slow vs Fast (Opposite and same direction)

# Prepare environment
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Define helper functions
def generate_sign_flipped_permutations(data, num_permutations=10000):
    permuted_means = []
    for _ in range(num_permutations):
        flipped_data = data * np.random.choice([-1, 1], size=len(data), replace=True)
        permuted_means.append(np.mean(flipped_data))
    return np.array(permuted_means)

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
f, axes = plt.subplots(1, 2)
colors = ["#3b0086","#00863b"]
colors_op = ["#b099ce", "#b2dac4"]
titles = ["Opposite direction", "Same direction"]
box_width = 0.4
fontsize = 12
bar_pos = [0 - (box_width / 1.5), 1 + (box_width / 1.5)]

# Load matrix containing the raw average speed values
speed = np.load("../data/behavior/speed_raw.npy")

# Load the stimulated trials
stim = np.load("../data/behavior/stim.npy")

# Load the fast/slow movements
# Classfied according to the online algortihm = Fast (Peak speed > 2 previous movements), Slow (Peak speed < 2 previous movements)
slow = np.load("../data/behavior/slow.npy")
fast = np.load("../data/behavior/fast.npy")

# Calculate the average change in movement speed for each patient, stimulation condition and stimulation (yes/no)
n_subjects = speed.shape[0]
speed_norm = np.zeros((n_subjects, 2, 3, 2))
for sub in range(n_subjects):

    # Loop over conditions
    for cond in range(2):

        # Loop over the 3 subsequent movements
        for subsequent in range(3):

            # Get the speed and stimulation for one patient during the stimulation block
            speed_tmp = speed[sub, cond, 0, :].flatten()
            stim_tmp = stim[sub, cond, 0, :].flatten()

            # Get the index of the stimulated movements (same and other condition)
            stim_idx = np.where(stim_tmp == 1)[0]

            # Get the speed n after the current one (ignore indexes that extend beyond the length of the block)
            stim_idx = stim_idx[stim_idx + subsequent < len(speed_tmp)]
            speed_stim_n = speed_tmp[stim_idx + subsequent]
            if subsequent == 0:
                speed_stim_0 = speed_stim_n

            # Get the speed during the recovery blocks 
            speed_no_stim_tmp = speed[sub, :, 1, :].flatten()

            # Get the index of the fast/slow movements (NOT stimulated, but classified with same criteria and stimulated movements)
            if cond == 0:
                fast_tmp = fast[sub, :, 1, :].flatten()
                no_stim_idx = np.where(fast_tmp == 1)[0]
            else:
                slow_tmp = slow[sub, :, 1, :].flatten()
                no_stim_idx = np.where(slow_tmp == 1)[0]

            # Get the speed n after the current one (ignore indexes that extend beyond the length of the block)
            no_stim_idx = no_stim_idx[no_stim_idx + subsequent < len(speed_no_stim_tmp)]
            speed_no_stim_n = speed_no_stim_tmp[no_stim_idx + subsequent]
            if subsequent == 0:
                speed_no_stim_0 = speed_no_stim_n

            # Calculate the average change in speed for an individual patient (in respect to the stimulated/not-stimulated fast/slow movement in %)
            speed_stim_n_norm = ((speed_stim_n - speed_stim_0[:len(speed_stim_n)])/ speed_stim_0[:len(speed_stim_n)]) * 100
            speed_no_stim_n_norm = ((speed_no_stim_n - speed_no_stim_0[:len(speed_no_stim_n)])/ speed_no_stim_0[:len(speed_no_stim_n)]) * 100
            speed_norm[sub, cond, subsequent, 0] = np.nanmean(speed_stim_n_norm)
            speed_norm[sub, cond, subsequent, 1] = np.nanmean(speed_no_stim_n_norm)

# Plot the speed shifts and compare statistically
for i, subsequent in enumerate([2, 1]):

    # Calculate the difference between stimulated and not stimulated movement (speed-shift)
    speed_shift = speed_norm[:, :, :, 0] - speed_norm[:, :, :, 1]

    # Plot speed shift values in box plot
    ax = axes[i]
    for cond in range(2):
        bp = ax.boxplot(x=speed_shift[:, cond, subsequent],
                         positions=[bar_pos[cond]],
                         widths=box_width,
                         patch_artist=True,
                         boxprops=dict(facecolor=colors_op[cond], color=colors_op[cond]),
                         capprops=dict(color=colors_op[cond]),
                         whiskerprops=dict(color=colors_op[cond]),
                         medianprops=dict(color="indianred", linewidth=0),
                         flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=0,
                                         markeredgecolor='none')
                         )
        # Test difference from 0
        observed_mean = np.mean(speed_shift[:, cond, subsequent])
        permuted_means = generate_sign_flipped_permutations(speed_shift[:, cond, subsequent])
        p_value = np.mean(np.abs(permuted_means) >= np.abs(observed_mean))
        print(p_value)

    # Add the individual lines
    for speed_shift_patient in speed_shift[:, :, subsequent]:
        ax.plot(bar_pos[0], speed_shift_patient[0], marker='o', markersize=1.5, color=colors[0])
        ax.plot(bar_pos[1], speed_shift_patient[1], marker='o', markersize=1.5, color=colors[1])
        # Add line connecting the points
        ax.plot(bar_pos, speed_shift_patient, color="black", linewidth=0.5, alpha=0.3)

    # Add statistics
    res_perm = scipy.stats.permutation_test(data=(speed_shift[:, 0, subsequent], speed_shift[:, 1, subsequent]),
                                       statistic=diff_mean_statistic,
                                       n_resamples=100000, permutation_type="samples")
    p = res_perm.pvalue
    text = get_sig_text(p)
    ymin, ymax = ax.get_ylim()
    ymax +=2
    ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - 1, ymax, ymax, ymax - 1], color="black",
            linewidth=1)
    ax.text(np.mean(bar_pos), ymax, text, ha="center", va="bottom", fontsize=fontsize)

    # Adjust plot
    ax.axhline(0, linewidth=1, color="black", linestyle="dashed")
    ax.yaxis.set_tick_params(labelsize=fontsize-2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fast", "Slow"], fontsize=fontsize)
    ax.set_yticks([-10, 10])
    ax.set_yticklabels([-10, 10], fontsize=fontsize)
    ax.set_ylim([-12, 15])
    if i == 0:
        ax.set_ylabel(f"Stimulation-induced \nspeed shift [%]", fontsize=fontsize)
    ax.set_title(titles[subsequent-1], fontsize=fontsize)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
plt.savefig(f"../figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../figures/{plot_name}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()
