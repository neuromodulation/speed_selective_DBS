# Results Figure 4: Compare beta power between stimulated and not stimulated movements

# Prepare environment
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import scipy
import matplotlib
matplotlib.use('TkAgg')

# Define helper functions
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

def diff_mean_statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

# Prepare plotting
labels = ["Stim", "No Stim"]
colors = np.array(["#860008", "darkgrey"])
colors_op = np.array(["#860008", "darkgrey"])
line_style = ["-", "-"]
fig, axes = plt.subplots(4, 1, figsize=(10, 7), height_ratios=[1, 0.5, 0.5, 3])
fontsize = 12
plot_name = os.path.basename(__file__).split(".")[0]

# Set parameters
target_ch_name = "ECOG_R_1_CAR_12345"
baseline = (-0.4, -0.1)
mode = "zscore"
freq_min = 20
freq_max = 35
frequencies = np.arange(freq_min, freq_max, 1)
tmin = -0.4
tmax = 1

# Load the epochs aligned to the peak speed 
epochs = mne.read_epochs("../data/electrophysiology/epochs_peak_speed_aligned.fif")
sfreq = epochs.info['sfreq']

# Load the index of stimulated and speed-matching non stimulated movements 
stim_idx = np.load("../data/electrophysiology/stim_idx.npy")
no_stim_idx = np.load("../data/electrophysiology/speed_matching_no_stim_idx.npy")

# Load the time point of the stimulation burst edges for artifact handeling
t_edges_stim_burst = np.load("../data/electrophysiology/t_edges_stim_burst.npy")

# Compute the time-frequency representation using morlet wavelets
tfr = epochs.compute_tfr(method="morlet", freqs=frequencies, picks=target_ch_name, average=False, n_cycles=4)

# Crop to exclude edge artifacts
tfr = tfr.crop(tmin=tmin, tmax=tmax)

# Normalize to baseline
tfr = tfr.apply_baseline(baseline=baseline, mode=mode)

# Handle the DBS artifact______________________________________________________________________________________________________________________________________________
# Replace the edges of the stimulation bursts, which lead to artifacts in the time-frequency spectrum, with None (period of 65 ms around the edge)
buffer_t = 0.065
for i, (tfr_stim, t_edge) in enumerate(zip(tfr[stim_idx], t_edges_stim_burst)):

    # Replace with None
    tfr_stim_new = tfr_stim.copy()
    for t in t_edge:
        tmin_art = int(np.abs(tmin - (t - buffer_t)) * sfreq)
        tmax_art = int(np.abs(tmin - (t + buffer_t)) * sfreq)
        tfr_stim_new[:, :, tmin_art:tmax_art] = None

    # Replace in original tfr object
    tfr._data[stim_idx[i], :, :, :] = tfr_stim_new


# Calculate the average edge of the stimulation burst
t_mean_edges_stim_burst = np.mean(t_edges_stim_burst, axis=0)
# Define the artifact windows (where data is not physiological)
art_window_1 = [t_mean_edges_stim_burst[0]-buffer_t, t_mean_edges_stim_burst[0]+buffer_t]
art_window_2 = [t_mean_edges_stim_burst[1]-buffer_t, t_mean_edges_stim_burst[1]+buffer_t]

# Linearly interpolate Nan values for plotting
def interpolate_nan(array):
    array = array.copy()
    nans = np.isnan(array)
    array[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), array[~nans])
    return array
data_interp = np.apply_along_axis(interpolate_nan, -1, tfr[stim_idx].get_data())
tfr_interp = tfr.copy()
tfr_interp._data[stim_idx] = data_interp

# If < 15 % are None replace with median of remaining distribution (necessary for cluster statistics)
data = tfr[stim_idx].get_data()
thres_exclude = 0.15
for t in range(data.shape[-1]):
    nan_ratio = np.mean(np.isnan(data[:, 0, 0, t]))
    if 0 < nan_ratio < thres_exclude:
        # Replace NaNs across trials at time t with median over non-NaNs
        replace_val = np.nanmedian(data[:, :, :, t], axis=0)
        nan_trials = np.isnan(data[:, 0, 0, t])
        data[nan_trials, :, :, t] = replace_val
tfr._data[stim_idx] = data


# Plot traces and compute cluster statistics comparing stim and no stim_____________________________________________________________________________________________________________________________________

times = tfr.times
for i, idx in enumerate([stim_idx, no_stim_idx]):

    # Add speed
    mean_behav = np.mean(epochs[idx].get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    std_behav = np.std(epochs[idx].get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    ax = axes[0]
    ax.plot(times[:len(mean_behav)], mean_behav, color=colors[i], linewidth=2, alpha=0.8, linestyle=line_style[i])
    ax.fill_between(times[:len(mean_behav)], mean_behav - std_behav, mean_behav + std_behav, color=colors[i], alpha=0.3)

    # Add raw trace of exemplar stimulated and not stimulated movement
    raw_no_stim = np.mean(epochs[idx[0]].get_data(target_ch_name, tmax=tmax, tmin=tmin), axis=0).squeeze()
    axes[[2, 1][i]].plot(times[:len(raw_no_stim)], raw_no_stim, color=colors[i], linewidth=2, alpha=0.8, linestyle=line_style[i])

    # Add trace of average beta power
    mean_power = np.mean(tfr_interp[idx].get_data(picks=target_ch_name), axis=(0, 1, 2)) 
    sem_power = scipy.stats.sem(tfr_interp[idx].get_data(picks=target_ch_name), axis=(0, 1, 2)) 
    ax = axes[3]
    ax.plot(times, mean_power, linestyle=line_style[i], label=labels[i], color=colors[i], linewidth=2, alpha=0.8)
    ax.fill_between(times, mean_power - sem_power, mean_power + sem_power, color=colors_op[i], alpha=0.1)

# Run cluster permutation test
# Remove the baseline window as no difference is expected here
tfr_short = tfr.copy().crop(tmin=baseline[-1])
power_stim = np.nanmean(tfr_short[stim_idx].get_data(picks=target_ch_name), axis=(1, 2))
power_no_stim = np.nanmean(tfr_short[no_stim_idx].get_data(picks=target_ch_name), axis=(1, 2))
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([power_stim, power_no_stim],
                                                                           n_permutations=1000, tail=0, threshold=None, seed=40)
print(cluster_p_values)
for i, cluster in enumerate(clusters):
    if cluster_p_values[i] < 0.05:

        # Extract the time points for this cluster
        cluster_times = tfr_short.times[cluster]

        # Fill in the significant cluster area
        ymin, ymax = ax.get_ylim()
        ax.fill_between(cluster_times, ymin, ymax, color="grey", alpha=0.3)

# Adjust plot
for i in range(3):
    ax = axes[i]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["right", "bottom", "top"]].set_visible(False)
    ax.set_ylabel("Raw signal (Exemplar)", fontsize=fontsize, rotation=0, ha="right")
ax = axes[0]
ax.axvline(0, color="black", linewidth=1)
ax.set_ylabel("Speed", fontsize=fontsize, rotation=0, ha="right")
ax.set_title(f"{target_ch_name}", fontsize=fontsize)
ax = axes[3]
ymin, ymax = ax.get_ylim()
ax.fill_between(baseline, ymin, ymax, color="lightgrey", alpha=0.3)
# Mark the artifact windows
ax.fill_between(art_window_1, ymin, ymax, color="white", alpha=0.5, hatch='/', zorder=2)
ax.fill_between(art_window_2, ymin, ymax, color="white", alpha=0.5, hatch='/', zorder=2)
ax.axvline(0, color="black", linewidth=1)
ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
ax.set_ylabel("High $\\beta$ (20-35 Hz) \n[normalized zscore]", fontsize=fontsize, rotation=0, ha="right")
ax.set_xlabel("Time aligned to peak speed [sec]", fontsize=fontsize)
ax.xaxis.set_tick_params(labelsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize)
ax.legend(loc="lower left")
ax.spines[["right", "top"]].set_visible(False)
plt.subplots_adjust(hspace=0.1, left=0.25)

# Save figure
plt.savefig(f"../figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../figures/{plot_name}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()

# Compare average power values in defined time bins for stim and no stim_________________________________________________________________________________________________________________________

# Define windows of interest (during and after the stimulation burst)
stats_window1 = [art_window_1[1], art_window_2[0]]
stats_window2 = [art_window_2[1], tmax]

# Plot average power values as boxplot
for i, window in enumerate([stats_window1, stats_window2]):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    bar_pos = [0, 1]
    box_width = 0.5
    power_all = []
    for j, idx in enumerate([stim_idx, no_stim_idx]):
        power = np.nanmean(tfr[idx].get_data(picks=target_ch_name, tmin=window[0], tmax=window[1]),
                                axis=(1, 2, 3))
        jitter = np.random.uniform(-0.05, 0.05, size=len(power))
        bp = ax.boxplot(x=power,
                        positions=[bar_pos[j]],
                        showfliers=False,
                        widths=box_width,
                        patch_artist=True,
                        boxprops=dict(color=colors[j], facecolor=colors[j]),
                        capprops=dict(color=colors[j]),
                        whiskerprops=dict(color=colors[j]),
                        medianprops=dict(color=colors[j], linewidth=1),
                        flierprops=dict(marker='o', markerfacecolor="dimgray", markersize=0,
                                        markeredgecolor='none')
                        )
        ymin, ymax = ax.get_ylim()
        ax.scatter(np.repeat(bar_pos[j], len(power)) + jitter, power, s=2.5, c="dimgray", marker='o',
                   zorder=2)
        power_all.append(power)

    # Compare distributions statistically
    res = scipy.stats.permutation_test(data=(power_all[0], power_all[1]),
                                       statistic=diff_mean_statistic,
                                       n_resamples=100000, permutation_type="independent")
    text = get_sig_text(res.pvalue)
    cap_length = (ymax - ymin)/30
    ymax += cap_length
    ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - cap_length, ymax, ymax, ymax - cap_length], color="black",
            linewidth=0.8)
    ax.text(np.mean(bar_pos), ymax + 0.1, text, fontsize=fontsize, horizontalalignment='center')

    # Adjust plot
    ax.set_ylim([ymin, ymax+cap_length*3])
    ax.set_ylabel("$\\beta$ [zscore]", fontsize=fontsize)
    ax.set_xticks(ticks=[0, 1], labels=labels, fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize - 1)
    ax.set_title(f"{np.round(window[0], 2)}-{np.round(window[1], 2)} sec", fontsize=fontsize)
    ax.spines[['right', 'top']].set_visible(False)

    # Save figure
    plt.savefig(f"../figures/{plot_name}_{window[0]}_{window[1]}.pdf", format="pdf", transparent=True, bbox_inches="tight")
    plt.savefig(f"../figures/{plot_name}_{window[0]}_{window[1]}.svg", format="svg", transparent=True, bbox_inches="tight")

    plt.show()