

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


# Set parameters
tmin = -1
tmax = 1
baseline = (None, None)
mode = "percent"
cmap = "jet"
freq_min = 15
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]

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
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.5))], axis=-1).squeeze()
slow_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 0
fast_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 1
fast_stim_idx = np.where(stim & fast_block_idx)[0]
slow_stim_idx = np.where(stim & slow_block_idx)[0]
not_stim_idx = np.where(~stim)[0]

# Get the fast and slow NOT stimulated movements
idx_recovery = np.where([True if (x in [2, 4]) else False for x in epochs.get_data(["BLOCK"])[:,:, epochs.times == 0].squeeze().astype(int)])[0]
epochs_recovery = epochs.copy()[idx_recovery]
peak_speed = epochs_recovery.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
slow_idx = []
fast_idx = []
for i, ps in enumerate(peak_speed):
    if i > 1 and np.all(ps < peak_speed[i-2:i]):
        slow_idx.append(i)
    elif i > 1 and np.all(ps > peak_speed[i-2:i]):
        fast_idx.append(i)
slow_idx = idx_recovery[slow_idx]
fast_idx = idx_recovery[fast_idx]

# Compute tfr
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)
# Crop tfr
tfr.crop(tmin=tmin, tmax=tmax)
# Smooth tfr
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)
# Decimate tfr
tfr.decimate(decim=40)
tfr_data = tfr.data.squeeze()

# Compute difference fast_stim vs fast
tfr1 = tfr[slow_idx].average()
tfr2 = tfr[fast_idx].average()
tfr_diff_fast = tfr1.copy()
tfr_diff_fast.data = tfr2.data - tfr1.data

# Compute difference slow_stim vs slow
tfr1 = tfr[slow_stim_idx].average()
tfr2 = tfr[fast_stim_idx].average()
tfr_diff_slow = tfr1.copy()
tfr_diff_slow.data = tfr2.data - tfr1.data

# Compute difference
tfr_diff = tfr1.copy()
tfr_diff.data = tfr_diff_fast.data - tfr_diff_slow.data
tfr_diff = tfr_diff.apply_baseline(baseline=baseline, mode=mode)

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
tfr_diff.plot(axes=ax, colorbar=True, cmap=cmap, show=False)
# Add colorbar
im = ax.images
cbar = im[-1].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Difference in power in %', fontsize=14)

# Add average speed
ax0_twin = ax.twinx()
behav_mean = np.mean(epochs[not_stim_idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
behav_std = np.std(epochs[not_stim_idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
ax0_twin.plot(times, behav_mean, color="black", linewidth=2, alpha=0.7)
ax0_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color="grey", alpha=0.5)
ax0_twin.get_yaxis().set_visible(False)

# Adjust plot
ax.set_ylim(freq_min, freq_max - 1)
ax.set_xlabel("Time(seconds)", fontsize=14)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

# Adjust figure
plt.suptitle("Fast stim-Fast vs Slow stim-Slow", fontsize=20)

# Save figure
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/diff_stim_no_stim_{sub}.svg",
            format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/diff_stim_no_stim_{sub}.png",
            format="png", bbox_inches="tight", transparent=False)
plt.show(block=True)
