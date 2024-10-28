# Compute the correlation between the power and the feature over trials
# Power computed on the -400-100 ms before peak speed (without stimulation)
# Use the LFPs and ECoG contacts and compute all correlations and write in table

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import scipy
import matplotlib
import seaborn as sb
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set the parameters
med = "Off"
feature_name = "mean_speed_300"
n_norm = 5
n_cutoff = 5

# Load the excel sheet containing the phenotype data
df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
subject_list = list(df["ID Berlin_Neurophys"][1:25])
subject_list.remove("L003")  # NO neurophys data available

# Load behavioral data
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape
# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)
# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

for i, sub in enumerate(subject_list):

    # Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
           f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
           f'rawdata/'

    bids_paths = find_matching_paths(root, tasks=["VigorStimR", "VigorStimL"],
                                        extensions=".vhdr",
                                        subjects=sub,
                                        sessions=[f"LfpMed{med}01", f"EcogLfpMed{med}01",
                                                  f"LfpMed{med}02", f"EcogLfpMed{med}02", f"LfpMed{med}Dys01"])
    raw = read_raw_bids(bids_path=bids_paths[0])
    raw.load_data()
    sfreq = raw.info["sfreq"]

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

    # Add bipolar channel LFP and ECoG
    # Average one segmented level
    # Subtract from level on the other side of the stimulation contact (if possible, otherwise adjacent one)
    # Add channel ="Ipsi/Contralateral LFP/ECoG"

    # Cut into epochs

    # Calculate correlation between power in different frequency bands
    # Save values in table
    # Save values to compute group level things afterwards


# Add bipolar motor cortex channel
if sub == "EL008":
    channels = ["ECOG_L_05_SMC_AT", "ECOG_L_06_SMC_AT"]
elif sub == "EL012":
    channels = ["ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]
new_chan = np.diff(raw.get_data(channels), axis=0)
# Create new name and info
target_chan_name = "bipolar_motor_cortex"
info = mne.create_info([target_chan_name], raw.info['sfreq'], ["ecog"])
# Add channel to raw object
new_chan_raw = mne.io.RawArray(new_chan, info)
raw.add_channels([new_chan_raw], force_update_info=True)



# Plot time-frequency spectrum for all events
event_id = 10003
event_names = ["Movement start", "Peak speed", "Movement end"]

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-0.4, tmax=-0.1, baseline=None, reject_by_annotation=True)
epochs_samples = list(epochs.selection.copy())
epochs.drop_bad()
drop_sample = np.where(list(map(lambda x: x == ('bad stim',), epochs.drop_log)))[0]
drop_idx = np.where(epochs_samples == drop_sample)[0]
drop_idx = np.array([epochs_samples.index(samp) for samp in drop_sample])

# Compute the correlation between power and feature in different frequency bands
#θ(4-8 Hz), α(8–12Hz), β(13–35Hz), low β(13–20Hz), high β(20–35Hz), all γ(60–200Hz), low γ(60–80Hz) (90–20Hz)
# Keep only target channel
epochs.load_data()
epochs = epochs.pick([target_chan_name, 'STIM_CONDITION'])
bands = [[4, 8], [8, 12], [13, 35], [13, 20], [20, 35], [60, 200], [60, 80], [90, 200]]
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]
box_width = 0.3
for i, band in enumerate(bands):

    # Compute power in frequency band
    psds, freqs = epochs.compute_psd(fmin=band[0], fmax=band[-1], method='multitaper').get_data(return_freqs=True)
    power = np.mean(psds, axis=-1).flatten()

    plt.figure(figsize=(10,4))

    # Calculate change in power from start
    for cond in range(2):

        # Get all trials for one condition
        cond_id = np.unique(epochs.get_data(["STIM_CONDITION"]), axis=-1).flatten()
        cond_idx = np.where(cond_id == cond)[0]
        power_tmp = power[cond_idx]

        # Remove outliers
        np.apply_along_axis(lambda m: u.fill_outliers_nan_ephys(m), axis=0, arr=power_tmp)
        # Cut away first 5
        power_tmp = power_tmp[n_cutoff:]
        # Normalize in % to next 5 trials
        power_perc = ((power_tmp - power_tmp[:n_norm].mean()) / power_tmp[:n_norm].mean()) * 100

        # Behavior: Remove the same trials, cutoff and normalize
        # Remove trials that were dropped (because of stimulation artifact)
        feature_cond = feature[cond, :]
        if np.max(cond_idx) > 96*3:
            if len(drop_idx[drop_idx > 96*2]) > 0:
                drop_idx_new = np.array([idx - 96*2 for idx in drop_idx if idx > 96*2])
                feature_cond = np.delete(feature_cond, drop_idx_new)
        else:
            if len(drop_idx[drop_idx < 96*2]) > 0:
                drop_idx_new = np.array([idx for idx in drop_idx if idx < 96*2])
                feature_cond = np.delete(feature_cond, drop_idx_new)
        feature_cond = feature_cond[n_cutoff:]
        feature_cond = u.norm_perc(feature_cond, n_norm=n_norm)

        # Calculate correlation
        not_nan = ~np.isnan(feature_cond) & ~np.isnan(power_perc)
        try:
            corr, p = spearmanr(power_perc[not_nan], feature_cond[not_nan])
            # corr, p = u.permutation_correlation(x, y, n_perm=100000, method='pearson')
            p = np.round(p, 3)
            if p < 0.05:
                label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
            else:
                label = f" R = {np.round(corr, 2)} p = {p}"
            plt.subplot(1, 2, cond+1)
            sb.regplot(x=power_perc[not_nan], y=feature_cond[not_nan], label=label, scatter_kws={"color": "dimgrey"}, line_kws={"color": "indianred"})

            # Adjust plot
            plt.legend(loc="upper right", fontsize=11)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel(f"Change in {feature_name} %", fontsize=14)
            plt.xlabel(f"Change in power %", fontsize=14)
            u.despine()
            plt.title(labels[cond], fontsize=13)
        except:
            print(f"{band[0]}-{band[-1]} only nan")

    # Adjust plot
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.8, wspace=0.4)
    plt.suptitle(f"{band[0]}-{band[-1]} Hz Power", fontsize=13)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{feature_name}_{band[0]}_{band[-1]}.svg",
                format="svg", bbox_inches="tight", transparent=True)
    plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{sub}_{feature_name}_{band[0]}_{band[-1]}.png",
                format="png", bbox_inches="tight", transparent=True)

plt.show()
