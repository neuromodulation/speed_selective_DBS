# Results Figure 4: Power difference between stimulated and not stimulated movements

import numpy as np
import matplotlib.pyplot as plt
import mne
import sys
sys.path.insert(1, "../Code")
import utils as u
import os
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib
from scipy.ndimage import uniform_filter1d
matplotlib.use('Qt5Agg')

# Set parameters
tmin = -0.6
tmax = 0.6
baseline = (None, None)
mode = "percent"
cmap = "viridis"
freq_min = 15
freq_max = 150
frequencies = np.arange(freq_min, freq_max, 2)
fontsize = 7

# Load the data
sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_123_CAR.fif"
raw = mne.io.read_raw_fif(path).load_data()
# Add re-references raw channels
ecog_names = ["ECOG_R_1_CAR_raw", "ECOG_R_2_CAR_raw", "ECOG_R_3_CAR_raw"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_name = "ECOG_R_1_CAR_raw"

# Load index of similar movements
id = 6
similar_slow = np.load(f"../../../Data/Off/processed_data/Slow_similar.npy").astype(bool)
similar_fast = np.load(f"../../../Data/Off/processed_data/Fast_similar.npy").astype(bool)
fast_idx = np.where(np.hstack((similar_fast[id, 1, :, :].flatten(), similar_fast[6, 0, :, :].flatten())))[0]
slow_idx = np.where(np.hstack((similar_slow[id, 1, :, :].flatten(), similar_slow[6, 0, :, :].flatten())))[0]
no_stim_idx = np.hstack((slow_idx, fast_idx))

# Filter out line noise
#raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1, baseline=None, reject_by_annotation=True)

# Get fast and slow stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
stim_idx = np.where(stim)[0]

# Compare the peak speed values
peak_speed_stim = epochs[stim_idx].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
peak_speed_no_stim = epochs[no_stim_idx].get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
# Plot as box plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
ax.boxplot([peak_speed_stim, peak_speed_no_stim])
ax.set_ylabel("Peak speed", fontsize=14)
ax.yaxis.set_tick_params(labelsize=12)
# Add the p value from an independent t-test
t, p = ttest_ind(peak_speed_stim, peak_speed_no_stim)
ax.text(0.5, 0.9, f"p = {p:.4f}", fontsize=fontsize, ha="center", va="center", transform=ax.transAxes)

# Compute tfr
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)
# Crop tfr
tfr.crop(tmin=tmin, tmax=tmax)
# Smooth tfr
#tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)
# Decimate tfr
tfr.decimate(decim=40)
tfr_data = tfr.data.squeeze()

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.5, 1.8))

# Plot difference in power
idx1 = stim_idx
idx2 = no_stim_idx
tfr1 = tfr[idx1].average().apply_baseline(baseline=baseline, mode=mode)
tfr2 = tfr[idx2].average().apply_baseline(baseline=baseline, mode=mode)
tfr_diff = tfr1.copy()
tfr_diff.data = tfr1.data - tfr2.data
power = tfr_diff.data.squeeze()
im = ax.imshow(power, aspect="auto", origin="lower", cmap=cmap, extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
# Add colorbar
cbar = fig.colorbar(im, ax=ax, location="top")
cbar.outline.set_visible(False)
cbar.set_label('Stimulated vs not stimulated\n Normalized power difference[%]', fontsize=fontsize-1)
cbar.ax.tick_params(labelsize=fontsize-1)
# Adjust plot
ax.set_xlabel("Time(seconds)", fontsize=fontsize-1)
ax.set_ylabel("Frequency(Hz)", fontsize=fontsize-1)
ax.set_ylim(freq_min, freq_max-1)
ax.yaxis.get_label().set_fontsize(fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize-2)
ax.xaxis.set_tick_params(labelsize=fontsize-2)
ax.spines[['top', 'right']].set_visible(False)

# Add average speed
color= "lightgrey"
color_op = "lightgrey"
line_style = ["-", "--"]
ax_twin = ax.twinx()
for j, idx in enumerate([stim_idx, no_stim_idx]):
    # Add speed
    behav_mean = np.mean(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    behav_std = np.std(epochs[idx].get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
    ax_twin.plot(times, behav_mean, color=color, linewidth=2, alpha=0.3, linestyle=line_style[j])
    #ax_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=color_op, alpha=0.1)
    # Adjust plot
    ax_twin.set_yticks([])
    ax_twin.spines[['top', 'right']].set_visible(False)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}.pdf",
        format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}.png",
        format="png", bbox_inches="tight", transparent=False)

plt.show(block=True)