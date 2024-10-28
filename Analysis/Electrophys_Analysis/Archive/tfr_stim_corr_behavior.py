# Correlate stimulation-induced changes in the TFR with changes in behavior

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
import seaborn as sb
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
tmin = -0.75
tmax = 0.75
baseline = (None, None)
mode = "percent"
cmap = "jet"
freq_min = 15
freq_max = 70
frequencies = np.arange(freq_min, freq_max, 2)
fontsize = 7
bands = [[15, 25], [40, 60]]
band_names = [f"{x[0]}-{x[1]}" for x in bands]

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
slow_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 0
fast_block_idx = epochs.get_data(["STIM_CONDITION"])[:,:, epochs.times == 0].squeeze().astype(int) == 1
fast_stim_idx = np.where(stim_idx & fast_block_idx)[0]
slow_stim_idx = np.where(stim_idx & slow_block_idx)[0]
not_stim_idx = np.where(~stim_idx)[0]
stim_idx = np.where(stim_idx)[0]

# Get the peak speed values
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute tfr
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=[target_chan_name], average=False)
# Crop tfr
tfr.crop(tmin=tmin, tmax=tmax)
# Smooth tfr
tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)
# Decimate tfr
tfr.decimate(decim=40)
tfr = tfr.apply_baseline(baseline=(None, None), mode="zscore")

# Loop over slow amd fast
names = ["Slow", "Fast", "No stim", "Stim"]
for i, idx_cond in enumerate([slow_stim_idx, fast_stim_idx, not_stim_idx, stim_idx]):

    # Get the average change in speed in the nexxt-next movement
    idx_next = idx_cond+2
    remove_idx = idx_next > len(peak_speed) - 1
    idx_next = idx_next[~remove_idx]
    idx_cond = idx_cond[~remove_idx]
    peak_speed_next = ((peak_speed[idx_next] - peak_speed[idx_cond]) / peak_speed[idx_cond]) * 100
    #peak_speed_next = peak_speed[idx_next] - peak_speed[idx_cond]
    #peak_speed_next = peak_speed[idx_cond]

    # Loop over epochs
    res = np.zeros((2, len(idx_cond)))
    for j, idx in enumerate(idx_cond):

        # Get the time of stimulation onset
        if i < 2:
            onset_stim = np.where(epochs[idx].get_data(["STIMULATION"])[:, :, ((epochs.times < 0.5) & (epochs.times > -0.5))].squeeze())[0][0]
            times = epochs.times[(epochs.times < 0.5) & (epochs.times > -0.5)]
            onset_time = times[onset_stim]
        else:
            onset_stim = 0.06

        for k, band in enumerate(bands):
            # Get the average power
            res[k, j] = tfr[idx].get_data(fmin=band[0], fmax=band[1], tmin=onset_time+0.25, tmax=onset_time+0.5).mean()

    # Correlate changes in average power with changes in behavior and plot
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    for k, band in enumerate(bands):
        ax = axes[k, 0]
        corr, p = pearsonr(res[k, :], peak_speed_next)
        label = f" R = {np.round(corr, 2)} p = {np.round(p, 4)}"
        sb.regplot(x=res[k, :], y=peak_speed_next, label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"}, ax=ax)
        ax.legend()
        ax.set_title(band_names[k])
        ax.set_xlabel("power change [%]")
        ax.set_ylabel("speed change [%]")

    # Plot the TFR
    tfr_cond = tfr[idx_cond].copy()
    tfr_average = tfr_cond.average()#.apply_baseline(baseline=baseline, mode="percent")
    power = tfr_average.data.squeeze()
    ax = axes[0, 1]
    im = ax.imshow(power, aspect="auto", origin="lower", cmap=cmap,
                        extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))

    # Compute the correlation for each TFR entry
    tfr_data = tfr_cond.data.squeeze()
    n_epochs, n_freqs, n_times = tfr_data.shape
    corr_p = np.zeros((2, n_freqs, n_times))
    for k, freq in enumerate(frequencies):
        print(f"Freq: {freq}")
        for l, time in enumerate(tfr.times):
            corr, p = pearsonr(peak_speed_next, tfr_data[:, k, l])
            corr_p[0, k, l] = corr
            corr_p[1, k, l] = p

    # Plot the correlation values
    sig = corr_p[1, :, :] > 0.05
    corr_sig = corr_p[0, :, :]
    corr_sig[sig] = 0
    ax = axes[1, 1]
    im = ax.imshow(corr_sig, aspect="auto", origin="lower", cmap="bwr",
                        extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)))

    # Adjust plot
    plt.suptitle(names[i])
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{names[i]}.pdf",
                format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{names[i]}.png",
                format="png", bbox_inches="tight", transparent=False)
plt.show(block=True)




