
import os
import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import scipy
import sys
from scipy.stats import pearsonr, spearmanr, ttest_ind
sys.path.insert(1, "../../../Code")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Set parameters
baseline = (-0.4, -0.1)
mode = "zscore"
freq_min = 20
freq_max = 35
frequencies = np.arange(freq_min, freq_max, 1)
tmin = -0.4
tmax = 0.8
fontsize = 6

# Load the data
path = f"EL012_ECoG_CAR_LFP_BIP.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
#target_chan_name = "LFP_R_2_BIP_234_8"
target_chan_name = "ECOG_R_1_CAR_12345"
#raw.pick(ch_names[-8:]).plot(block=True)
#raw.filter(None, 100)

# Load index of similar movements
id = 6
similar_slow = np.load(f"../../../Data/Off/processed_data/Slow_similar.npy").astype(bool)
similar_fast = np.load(f"../../../Data/Off/processed_data/Fast_similar.npy").astype(bool)
slow_idx = np.where(np.hstack((similar_slow[id, 1, :, :].flatten(), similar_slow[6, 0, :, :].flatten())))[0]
fast_idx = np.where(np.hstack((similar_fast[id, 1, :, :].flatten(), similar_fast[6, 0, :, :].flatten())))[0]
no_stim_idx = np.hstack((fast_idx, slow_idx))

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1.7, baseline=None, reject_by_annotation=True)

# Get fast and slow stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
stim_idx = np.where(stim)[0]

# Get the array of peak speeds
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

#print(f"{np.mean(stim_idx)} Stim Idx Mean")
#print(f"{np.mean(no_stim_idx)} No Stim Idx Mean")

# Compute tfr
tfr = epochs.compute_tfr(method="morlet", freqs=frequencies, picks=target_chan_name, average=False, n_cycles=4)

# Crop
tfr = tfr.crop(tmin=tmin, tmax=tmax)

# Apply baseline
tfr = tfr.apply_baseline(baseline=baseline, mode=mode)

# Replace the artifact regions with None
time_artifact = np.load("artifact_sec_epochs.npy")
if "ECOG" in target_chan_name:
    buffer_t = 0.065
    if freq_min >=40:
        buffer_t = 0.03
elif "LFP" in target_chan_name:
    buffer_t = 0.09
for i, (tfr_stim, t_artifact) in enumerate(zip(tfr[stim_idx], time_artifact)):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(tfr_stim.squeeze(), aspect="auto", origin="lower", cmap="jet",
               extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)),
               vmin=-3, vmax=3)
    plt.axvline(t_artifact[0]-buffer_t, color="red")
    plt.axvline(t_artifact[0]+buffer_t, color="red")
    plt.axvline(t_artifact[1]-buffer_t, color="red")
    plt.axvline(t_artifact[1]+buffer_t, color="red")

    # Replace with None
    tfr_stim_new = tfr_stim.copy()
    for t in t_artifact:
        tmin_art = int(np.abs(tmin - (t - buffer_t)) * sfreq)
        tmax_art = int(np.abs(tmin - (t + buffer_t)) * sfreq)
        tfr_stim_new[:, :, tmin_art:tmax_art] = None

    plt.subplot(1, 2, 2)
    plt.imshow(tfr_stim_new.squeeze(), aspect="auto", origin="lower", cmap="jet", vmin=-3, vmax=3)
    plt.axvline(int(np.abs(tmin - t_artifact[0] - buffer_t) * sfreq), color="red")
    plt.axvline(int(np.abs(tmin - t_artifact[0] + buffer_t) * sfreq), color="red")
    plt.axvline(int(np.abs(tmin - t_artifact[1] - buffer_t) * sfreq), color="red")
    plt.axvline(int(np.abs(tmin - t_artifact[1] + buffer_t) * sfreq), color="red")
    plt.close()

    tfr._data[stim_idx[i], :, :, :] = tfr_stim_new
mean_art_onset = np.mean(time_artifact, axis=0)
art_window_1 = [mean_art_onset[0]-buffer_t, mean_art_onset[0]+buffer_t]
art_window_2 = [mean_art_onset[1]-buffer_t, mean_art_onset[1]+buffer_t]

# Interpolate Nan values for plotting
def interpolate_nan(array):
    array = array.copy()
    nans = np.isnan(array)
    non_nans = ~nans
    array[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), array[non_nans])
    return array
data = tfr[stim_idx].get_data().copy()
data_interp = np.apply_along_axis(interpolate_nan, -1, data)
tfr_interp = tfr.copy()
tfr_interp._data[stim_idx, :, :, :] = data_interp

"""plt.figure()
plt.plot(data[0, 0, 0, :].flatten(), color="red", alpha=0.2)
plt.plot(data_interp[0, 0, 0, :].flatten(), color="green", alpha=0.2)
plt.show()"""

# If < 15 % are None replace with median of remaining distribution (necessary for cluster statistics)
data = tfr[stim_idx].get_data()
perc_none = np.count_nonzero(np.isnan(data[:, 0, 0, :]), axis=0) / len(data)
thres_exclude = 0.15
exclude = (perc_none > 0) & (perc_none < thres_exclude)
for t in range(data.shape[-1]):
    idx_none = np.where(np.isnan(data[:, 0, 0, t]))
    if 0 < np.sum(np.isnan(data[:, 0, 0, t]))/data.shape[0] < thres_exclude:
        replace_mean = np.nanmedian(data[:, :, :, t], axis=0)
        data[idx_none, :, :, t] = replace_mean
tfr._data[stim_idx, :, :, :] = data

"""plt.figure()
plt.plot(data.mean(axis=(0,1,2)).flatten())
data = tfr[stim_idx].get_data()
plt.plot(data.mean(axis=(0,1,2)).flatten())
plt.show()"""


# Prepare plotting
labels = ["Stim", "No Stim"]
colors = np.array(["#860008", "darkgrey"])
colors_op = np.array(["#860008", "darkgrey"])
line_style = ["-", "-"]
fig, axes = plt.subplots(4, 1, figsize=(3, 2.5), height_ratios=[0.5, 0.5, 2, 2])
times = tfr.times

# Loop over stim/no stim
for i, idx in enumerate([stim_idx, no_stim_idx]):

    # Add raw trace of exemplar stimulated and not stimulated movement
    raw_no_stim = np.mean(epochs[idx[0]].get_data(target_chan_name, tmax=tmax, tmin=tmin), axis=0).squeeze()
    axes[[1, 0][i]].plot(times[:len(raw_no_stim)], raw_no_stim, color=colors[i], linewidth=1, alpha=0.8, linestyle=line_style[i])

    # Add speed
    mean_behav = np.mean(epochs[idx].get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    std_behav = np.std(epochs[idx].get_data(["SPEED_MEAN"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    ax = axes[3]
    ax.plot(times[:len(mean_behav)], mean_behav, color=colors[i], linewidth=1, alpha=0.8, linestyle=line_style[i])
    ax.fill_between(times[:len(mean_behav)], mean_behav - std_behav, mean_behav + std_behav, color=colors[i], alpha=0.1)

    # Add power
    mean_power = np.mean(tfr_interp[idx].get_data(picks=target_chan_name), axis=(0, 1, 2)) #* 100
    std_power = scipy.stats.sem(tfr_interp[idx].get_data(picks=target_chan_name), axis=(0, 1, 2)) #* 100
    ax = axes[2]
    ax.plot(times, mean_power, linestyle=line_style[i], label=labels[i], color=colors[i], linewidth=1, alpha=0.8)
    ax.fill_between(times, mean_power - std_power, mean_power + std_power, color=colors_op[i], alpha=0.1)

# Do cluster statistics
# Remove the baseline window as no difference is expected here
tfr_short = tfr.copy().crop(tmin=baseline[-1])
power_stim = np.nanmean(tfr_short[stim_idx].get_data(picks=target_chan_name), axis=(1, 2))
power_no_stim = np.nanmean(tfr_short[no_stim_idx].get_data(picks=target_chan_name), axis=(1, 2))
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test([power_stim, power_no_stim],
                                                                           n_permutations=1000, tail=0, threshold=None, seed=40)
print(cluster_p_values)
for i, cluster in enumerate(clusters):
    if cluster_p_values[i] < 0.05:  # Adjust threshold as needed
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
ax.set_title(f"{target_chan_name}", fontsize=fontsize)

ax = axes[2]
ymin, ymax = ax.get_ylim()
ax.fill_between(baseline, ymin, ymax, color="grey", alpha=0.3)
ax.fill_between(art_window_1, ymin, ymax, color="white", alpha=0.5, hatch='/', zorder=2)
ax.fill_between(art_window_2, ymin, ymax, color="white", alpha=0.5, hatch='/', zorder=2)
sig_window1 = [art_window_1[1], art_window_2[0]]
sig_window2 = [art_window_2[1], tmax]
#ax.fill_between(sig_window1, ymin, ymax, color="red", alpha=0.5, hatch='/', zorder=2)
#ax.fill_between(sig_window2, ymin, ymax, color="red", alpha=0.5, hatch='/', zorder=2)
ax.axvline(0, color="black", linewidth=1)
ax.axhline(0, linewidth=0.5, color="black", linestyle="dashed")
ax.set_ylabel("High $\\beta$ (20-35 Hz) \n[normalized zscore]", fontsize=fontsize, rotation=0, ha="right")
ax.set_xlabel("Time aligned to peak speed [sec]", fontsize=fontsize)
ax.xaxis.set_tick_params(labelsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize)
ax.legend(loc="lower left")
ax.spines[["right", "top"]].set_visible(False)

ax = axes[3]
ax.set_ylabel("Average speed", fontsize=fontsize, rotation=0, ha="right")
ax.set_xlabel("Time aligned to peak speed [sec]", fontsize=fontsize)
ax.xaxis.set_tick_params(labelsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize)
ax.spines[["right", "top"]].set_visible(False)

plt.subplots_adjust(hspace=0.1, left=0.25)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}_{freq_min}_{freq_max}.pdf", format="pdf", bbox_inches="tight",
            transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}_{freq_min}_{freq_max}.png", format="png", bbox_inches="tight",
            transparent=False)
#plt.show()


# Add statistics (stim vs no_stim boxplot)
sig_window1 = [art_window_1[1], art_window_2[0]]
sig_window2 = [art_window_2[1], 0.8]
for i, sig_window in enumerate([sig_window1, sig_window2]):
    fig, ax = plt.subplots(1, 1, figsize=(1,1))
    bar_pos = [0, 1]
    box_width = 0.5
    power_all = []
    for j, idx in enumerate([stim_idx, no_stim_idx]):
        power = np.nanmean(epochs[idx].get_data(picks=["SPEED_MEAN"], tmin=sig_window[0], tmax=sig_window[1]),
                                axis=(1, 2))
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
        ax.scatter(np.repeat(bar_pos[j], len(power)) + jitter, power, s=0.5, c="dimgray", marker='o',
                   zorder=2)
        power_all.append(power)
    # Compare distributions
    res = scipy.stats.permutation_test(data=(power_all[0], power_all[1]),
                                       statistic=u.diff_mean_statistic,
                                       n_resamples=100000, permutation_type="independent")
    text = u.get_sig_text(res.pvalue)
    print(res.pvalue)
    cap_length = (ymax - ymin)/30
    ymax += cap_length
    ax.plot([bar_pos[0], bar_pos[0], bar_pos[1], bar_pos[1]], [ymax - cap_length, ymax, ymax, ymax - cap_length], color="black",
            linewidth=0.8)
    ax.text(np.mean(bar_pos), ymax + 0.1, text, fontsize=fontsize, horizontalalignment='center')

    # Adjust plot
    ax.set_ylim([ymin, ymax+cap_length*3])
    ax.set_ylabel("Average speed", fontsize=fontsize)
    ax.set_xticks(ticks=[0, 1], labels=labels, fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize - 1)
    ax.set_title(f"{np.round(sig_window[0], 2)}-{np.round(sig_window[1], 2)} sec", fontsize=fontsize)
    u.despine()

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}_{np.round(sig_window[0], 2)}-{np.round(sig_window[1], 2)} sec.pdf", format="pdf", bbox_inches="tight",
                transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}_{np.round(sig_window[0], 2)}-{np.round(sig_window[1], 2)} sec.png", format="png", bbox_inches="tight",
                transparent=False)
plt.show()