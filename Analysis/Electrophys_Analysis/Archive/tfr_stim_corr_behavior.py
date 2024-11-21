# Correlate stimulation-induced changes in the TFR with changes in behavior

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from scipy.stats import pearsonr, spearmanr, ttest_ind, zscore
import seaborn as sb
sys.path.insert(1, "../../../Code")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Set parameters
tmin = -0.2
tmax = 0.8
baseline = (-0.4, -0.1)
mode = "percent"
cmap = "bwr"
freq_min = 20
freq_max = 35
frequencies = np.arange(freq_min, freq_max, 1)
fontsize = 7
band = [20, 35]
band_names = f"{band[0]}-{band[1]}"

# Load the data
sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_all_CAR.fif"
raw = mne.io.read_raw_fif(path).load_data()
# Add re-references raw channels
ecog_names = ["ECOG_R_1_CAR_raw", "ECOG_R_2_CAR_raw", "ECOG_R_3_CAR_raw", "ECOG_R_4_CAR_raw", "ECOG_R_5_CAR_raw"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT", "ECOG_R_04_SMC_AT", "ECOG_R_05_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_names = [f"ECOG_R_{x}_CAR_raw" for x in range(1,6)]
target_chan_name = target_chan_names[4]

# Filter out line noise
#raw.notch_filter(50)

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
tfr = epochs.compute_tfr(method="morlet", freqs=frequencies, picks=[target_chan_name], n_cycles=5, average=False)
# Smooth tfr
#tfr.data = uniform_filter1d(tfr.data, size=int(sfreq*0.1), axis=-1)
# Decimate tfr
tfr.decimate(decim=5)
tfr = tfr.apply_baseline(baseline=baseline, mode=mode)

# Loop over slow amd fast
names = ["Fast", "Slow", "No stim", "Stim"]
for i, idx_cond in enumerate([fast_stim_idx, slow_stim_idx, not_stim_idx, stim_idx]):

    # Get the average change in speed in the next-next movement
    idx_next = idx_cond+2
    remove_idx = idx_next > len(peak_speed) - 1
    idx_next = idx_next[~remove_idx]
    idx_cond = idx_cond[~remove_idx]
    peak_speed_next = ((peak_speed[idx_next] - peak_speed[idx_cond]) / peak_speed[idx_cond]) * 100
    #peak_speed_next = peak_speed[idx_next] - peak_speed[idx_cond]
    #peak_speed_next = peak_speed[idx_cond]

    # Loop over epochs
    res = tfr[idx_cond].get_data(fmin=band[0], fmax=band[1], tmin=0.57, tmax=0.67).mean(axis=(2,3)).squeeze()

    # Correlate changes in average power with changes in behavior and plot
    # Remove outlier
    idx_out = np.abs(zscore(res)) > 3
    res = res[~idx_out]
    idx_cond = idx_cond[~idx_out]
    peak_speed_next = peak_speed_next[~idx_out]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5))
    corr, p = pearsonr(res, peak_speed_next)
    label = f" R = {np.round(corr, 2)} p = {np.round(p, 4)}"
    sb.regplot(x=res, y=peak_speed_next, label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"}, ax=ax1)
    ax1.legend()
    #ax1.set_title(band_names[k])
    ax1.set_xlabel("power change [%]")
    ax1.set_ylabel("speed change [%]")

    # Plot the TFR
    tfr_cond = tfr[idx_cond].copy()
    tfr_average = tfr_cond.average()
    power = tfr_average.data.squeeze()
    im = ax2.imshow(power, aspect="auto", origin="lower", cmap=cmap,
                        extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)), vmax=2, vmin=-2)

    # Compute the correlation for each TFR entry
    tfr_data = tfr_cond.get_data(tmin=tmin,tmax=tmax).squeeze()
    n_epochs, n_freqs, n_times = tfr_data.shape
    times = tfr.times[(tfr.times >= tmin) & (tfr.times < tmax)]
    corr_p = np.zeros((2, n_freqs, n_times))
    for k, freq in enumerate(frequencies):
        print(f"Freq: {freq}")
        for l, time in enumerate(times):
            corr, p = pearsonr(peak_speed_next, tfr_data[:, k, l])
            corr_p[0, k, l] = corr
            corr_p[1, k, l] = p

    # Plot the correlation values
    sig = corr_p[1, :, :] > 0.05
    corr_sig = corr_p[0, :, :]
    corr_sig[sig] = 0
    im = ax2.imshow(corr_sig, aspect="auto", origin="lower", cmap="bwr",
                        extent=(tmin, tmax, np.min(frequencies), np.max(frequencies)), alpha=0.5)

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




