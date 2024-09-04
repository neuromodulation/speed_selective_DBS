# Compare the tfr stimulation vs no stimulation vs fast vs slow

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr, ttest_ind
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')
import matplotlib as mpl

# Export text as regular text instead of paths or using svgfonts
mpl.rcParams['svg.fonttype'] = 'none'

# Set font to a widely-available but non-default font to show that Affinity
# ignores font-family
mpl.rcParams['font.family'] = 'Arial'
#mpl.rcParams['font.sans-serif'] = 'Helvetica'

# Set parameters
tmin = -1
tmax = 1
baseline = (None, None)
mode = "percent"
cmap = "jet"
freq_min = 15
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)
fontsize=7

# Load the data
sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\{sub}_cleaned.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_name = ch_names[-1]

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.5, tmax=1.5, baseline=None, reject_by_annotation=True)

# Get fast and slow stimulated movements
stim_idx = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.5))], axis=-1).squeeze()
stim_idx = np.where(stim_idx)[0]

# Get not stimulated movements from the recovery block
not_stim_idx = np.where([True if (x in [2, 4]) else False for x in epochs.get_data(["BLOCK"])[:,:, epochs.times == 0].squeeze().astype(int)])[0]
# Take only fastest movements (same length as stimulated movements)
epochs_recovery = epochs.copy()[not_stim_idx]
peak_speed = epochs_recovery.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
fast_slow = np.argsort(peak_speed)
not_stim_idx = not_stim_idx[fast_slow[-len(stim_idx):]]

# Compare the peak speed values
peak_speed1 = epochs[stim_idx].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
peak_speed2 = epochs[not_stim_idx].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
# Plot as box plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
ax.boxplot([peak_speed1, peak_speed2])
ax.set_ylabel("Peak speed", fontsize=14)
ax.yaxis.set_tick_params(labelsize=12)
# Add the p value from an independent t-test
t, p = ttest_ind(peak_speed1, peak_speed2)
ax.text(0.5, 0.9, f"p = {p:.4f}", fontsize=fontsize, ha="center", va="center", transform=ax.transAxes)


# Compute tfr
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)
# Crop tfr
tfr.crop(tmin=tmin, tmax=tmax)
# Smooth tfr
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)
# Decimate tfr
tfr.decimate(decim=40)
tfr_data = tfr.data.squeeze()

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5, 1.5))

# Plot difference in power
idx1 = stim_idx
idx2 = not_stim_idx
tfr1 = tfr[idx1].average().apply_baseline(baseline=(None, None), mode="percent")
tfr2 = tfr[idx2].average().apply_baseline(baseline=(None, None), mode="percent")
tfr_diff = tfr1.copy()
tfr_diff.data = tfr1.data - tfr2.data
power = tfr_diff.data.squeeze()
im = ax.imshow(power, aspect="auto", origin="lower", cmap=cmap, extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
# Add colorbar
cbar = fig.colorbar(im, ax=ax, location="top")
cbar.outline.set_visible(False)
cbar.set_label('Power stimulation - no stimulation in [%]', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize-2)
# Adjust plot
ax.set_xlabel("Time(seconds)", fontsize=fontsize)
ax.set_ylabel("Frequency(Hz)", fontsize=fontsize)
ax.set_ylim(freq_min, freq_max-1)
ax.yaxis.get_label().set_fontsize(fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize-2)
ax.xaxis.set_tick_params(labelsize=fontsize-2)
ax.spines[['top', 'right']].set_visible(False)

# Add average speed
color= "dimgrey"
color_op = "lightgrey"
ax_twin = ax.twinx()
for j, idx in enumerate([stim_idx, not_stim_idx]):
    # Add speed
    behav_mean = np.mean(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    behav_std = np.std(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
    ax_twin.plot(times, behav_mean, color=color, linewidth=1, alpha=0.5)
    ax_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=color_op, alpha=0.5)
    # Adjust plot
    ax_twin.set_yticks([])
    ax_twin.spines[['top', 'right']].set_visible(False)
    ax_twin.set_ylabel("Speed", fontsize=fontsize)

# Compute difference between tfr of slow and fast movements
n_epochs, n_freqs, n_times = tfr_data.shape
res = np.zeros((2, n_freqs, n_times))
for j, freq in enumerate(frequencies):
    print(f"Freq: {freq}")
    for k, time in enumerate(tfr.times):
        t, p = ttest_ind(a=tfr_data[idx1, j, k], b=tfr_data[idx2, j, k])
        res[0, j, k] = t
        res[1, j, k] = p

# Plot statistics
# Delete values that are not significant
p = res[1, :, :].copy()
alphas = p.copy()
p[res[1, :, :] > 0] = 1
alphas[res[1, :, :] < 0.05] = 0
alphas[res[1, :, :] >= 0.05] = 0.8
im = ax.imshow(p, aspect="auto", origin="lower", cmap="binary",
                    extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)), alpha=alphas)

# Mark average stimulation onset
onset_stim = [np.where(epochs[i].get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.5))].squeeze())[0][0] for i in stim_idx]
times = epochs.times[(epochs.times < 0.5) & (epochs.times > -0.5)]
onset_times = times[onset_stim]
mean_onset = np.mean(onset_times)
# Add to plot
ax.axvline(x=mean_onset, color="#E02F2F", linewidth=1)
#ax.text(mean_onset-0.1, int(np.mean(frequencies)), "Stimulation \nonset", color="red", fontsize=fontsize-2, rotation=270)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}.pdf",
        format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}.png",
        format="png", bbox_inches="tight", transparent=False)



plt.show(block=True)