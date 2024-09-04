# Correlate peak speed and power values in the time frequency plot

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
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

# Set the dataset
sub = "EL012"
med = "Off"
chan = "ECOG_bipolar"

# Set parameters
tmin = -1
tmax = 1
baseline = (None, None)
cmap = "jet"
freq_min = 15
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)
fontsize = 7

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                    extensions=".vhdr",
                                    subjects=sub,
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                              f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])

# Load dataset
raw = read_raw_bids(bids_path=bids_path[0])
raw.load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]

if sub == "EL008":
    target_chan_name = "ECOG_L_05_SMC_AT"
    target_chan_name = "ECOG_L_06_SMC_AT"
elif sub == "EL012":
    target_chan_name = "ECOG_R_02_SMC_AT"
    target_chan_name2 = "ECOG_R_03_SMC_AT"

# Rereference to ecog channel
raw._data[ch_names.index(target_chan_name), :] = raw.get_data(target_chan_name) - raw.get_data(["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]).mean(axis=0) #raw.get_data(target_chan_name2)
#raw._data[ch_names.index(target_chan_name), :] = raw.get_data(target_chan_name) - raw.get_data(target_chan_name2)

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Annotate periods with stimulation
sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
n_stim = len(sample_stim)
onset = (sample_stim / sfreq) - 0.1
duration = np.repeat(0.5, n_stim)
stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
raw.set_annotations(stim_annot)

# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1.5, tmax=1.5, baseline=None, reject_by_annotation=True)
epochs.drop_bad()

# Extract the peak speed for each epoch
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute the power over time
tfr = epochs.compute_tfr(method="morlet", freqs=frequencies, picks=[target_chan_name], average=False)

# Compute correlation for every time-frequency value (baseline corrected)
tfr.crop(tmin=tmin, tmax=tmax)

# Smooth the power
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)

# Decimate
tfr.decimate(decim=40)

tfr_data = tfr.data.squeeze()
n_epochs, n_freqs, n_times = tfr_data.shape
corr_p = np.zeros((2, n_freqs, n_times))
for i, freq in enumerate(frequencies):
    print(f"Freq: {freq}")
    for j, time in enumerate(tfr.times):
        corr, p = pearsonr(peak_speed, tfr_data[:, i, j])
        corr_p[0, i, j] = corr
        corr_p[1, i, j] = p

# Plot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(1.5, 2.7))

# Plot power
tfr_average = tfr.average().apply_baseline(baseline=baseline, mode="zscore")
power = tfr_average.data.squeeze()
im = axes[0].imshow(power, aspect="auto", origin="lower", cmap=cmap, extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
#tfr_average = tfr.average()
#tfr_average.plot(axes=axes[0], baseline=(None,None), mode="percent", show=False, colorbar=True, cmap=cmap)
# Adjust plot
axes[0].set_xticks([])
axes[0].set_xlabel("")
# Add colorbar
"""im = axes[0].images
cbar = im[-1].colorbar
cbar.location = "top"
cbar.ax.tick_params(labelsize=fontsize - 2)
cbar.set_label('Power [%]', fontsize=fontsize)
cbar.outline.set_visible(False)"""
# Add colorbar
cbar = fig.colorbar(im, ax=axes[0], location="top")
cbar.ax.tick_params(labelsize=fontsize-2)
cbar.outline.set_visible(False)
cbar.set_label('Power [%]', fontsize=fontsize)

# Plot R values
# Delete correlation values that are not significant
sig = corr_p[1, :, :] > 0.05
corr_sig = corr_p[0, :, :]
corr_sig[sig] = 0
im = axes[1].imshow(corr_sig, aspect="auto", origin="lower", cmap="bwr", extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))
# Adjust plot
axes[1].set_xlabel("Time(seconds)", fontsize=fontsize)
# Add colorbar
cbar = fig.colorbar(im, ax=axes[1], location="top")
cbar.ax.tick_params(labelsize=fontsize-2)
cbar.outline.set_visible(False)
cbar.set_label('Correlation with peak speed [R]', fontsize=fontsize)

# Add average speed and ajust plots
color= "dimgrey"
color_op = "lightgrey"
for i in range(2):

    ax_twin = axes[i].twinx()
    ax = axes[i]

    # Add speed
    behav_mean = np.mean(epochs.get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    behav_std = np.std(epochs.get_data(["SPEED"], tmax=tmax, tmin=tmin), axis=0).squeeze()
    times = epochs.times[(epochs.times >= tmin) & (epochs.times < tmax)]
    ax_twin.plot(times, behav_mean, color=color, linewidth=1, alpha=0.5)
    ax_twin.fill_between(times, behav_mean - behav_std, behav_mean + behav_std, color=color_op, alpha=0.5)

    # Adjust plot
    ax_twin.set_yticks([])
    ax_twin.set_ylabel("Speed", fontsize=fontsize)
    ax.set_ylabel("Frequency(Hz)", fontsize=fontsize)
    # Adjust plot
    ax.set_ylim(freq_min, freq_max-1)
    ax.yaxis.get_label().set_fontsize(fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize-2)
    ax.xaxis.set_tick_params(labelsize=fontsize-2)
    ax.spines[['top', 'right']].set_visible(False)
    ax_twin.spines[['top', 'right']].set_visible(False)


# Adjust figure
plt.subplots_adjust(hspace=0.55)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{chan}.pdf",
        format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{chan}.png",
        format="png", bbox_inches="tight", transparent=False)
plt.show(block=True)
