# Investigate whether there are significant differences between the power during the fast and the slow condition

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
import pandas as pd
import scipy
import matplotlib
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set parameters
########################################################################################################################
med = "Off"
tmin = -0.2
tmax = -0.1
#baseline = (-0.75, -0.5)
baseline = None
mode = "zscore"
cmap = "jet"
fmin = 30
fmax = 100
n_cycles = 4
freqs = np.arange(fmin, fmax, 2)

# Read the list of the datasets
df = pd.read_excel(f'../../Data/Dataset_list.xlsx', sheet_name=med)

# Loop through the subjects
subject_list = list(df["ID Berlin_Neurophys"][1:21])
subject_list.remove("L003")  # NO neurophys data available
for sub in subject_list:

    # Load the data
    ########################################################################################################################
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

    # Filter out line noise
    #raw.notch_filter(50)

    # Drop bad and not interesting channels
    ########################################################################################################################
    raw.drop_channels(raw.info["bads"])
    raw.pick_channels(raw.copy().pick_types(ecog=True, dbs=True, eeg=True).info["ch_names"] +
                      ["STIMULATION", "STIM_CONDITION", "BLOCK"])
    ch_names = raw.info["ch_names"]

    # Extract relevant epochs
    ########################################################################################################################
    # Extract events
    events = mne.events_from_annotations(raw)[0]

    # Cut into epochs aligned to the peak speed of each movement
    event_id = 10003
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2,
                        baseline=None, reject_by_annotation=False)

    # Loop over epochs and check how far the stimulation is from the window of interest
    times = epochs.times
    is_stim = []
    for i, stim_epoch in enumerate(epochs.get_data("STIMULATION").squeeze()):
        # Get the samples at which stimulation is applied in this block
        idx_stim = np.where(stim_epoch == 1)[0]
        # Check how close the stimulation sample is to thw window of interest
        if len(idx_stim) > 0:
            # Time during which the artifact influences the power in the target window due to the wavelets
            t_extra = (n_cycles-1)/fmin
            if np.any((times[idx_stim] < tmax+t_extra) & (times[idx_stim] > tmin-t_extra)):
                is_stim.append(i)

    # Remove bad epochs (stimulation during movement)
    epochs.load_data()
    epochs.drop(is_stim)

    # Extract the block number and condition for each epoch
    epochs_tmp = epochs.copy().crop(tmin=tmin, tmax=tmax)
    block = np.unique(epochs_tmp.get_data("BLOCK"), axis=-1).flatten()
    cond = np.unique(epochs_tmp.get_data("STIM_CONDITION"), axis=-1).flatten()

    # Compute power using morlet wavelets
    ########################################################################################################################
    power = mne.time_frequency.tfr_morlet(epochs, n_cycles=n_cycles, return_itc=False,
                                          picks=np.arange(len(ch_names)-3),
                                          freqs=freqs, average=False, use_fft=True)

    # Apply baseline correction
    #power.apply_baseline(baseline=baseline, mode=mode)

    # Average the power in the determined window
    power.crop(tmin=tmin, tmax=tmax)
    power.average(dim="times")
    power_data = power.data.squeeze()

    # For each frequency and channel, determine whether there is a significant difference between
    # the power in the stim slow and fast condition
    ########################################################################################################################

    # Initialize figure which contains subplot for each channel
    ch_names = power.info["ch_names"]
    n_cols = np.ceil(len(ch_names)/6).astype(int)
    n_rows = 6
    cond_names = ["Slow", "Fast"]
    fig = plt.figure()
    for i, ch in enumerate(ch_names):

        # Create one figure for each channel
        plt.subplot(n_rows, n_cols, i + 1)

        # Loop over the conditions
        power_change_conds = []
        for c in range(2):

            # Get the power values for all epochs in the first block
            cond_power = power_data[(cond == c) & ((block == 1) | (block == 3)), i, :]

            # Delete first 5 moves
            cond_power = cond_power[5:, :]

            # Normalize power to average power of next 5 moves
            power_change = u.norm_perc(cond_power.T)

            # Compute the average power
            mean_power = power_change.mean(axis=-1)

            # Plot
            plt.plot(freqs, mean_power, label=cond_names[c])

            # Save change in power
            power_change_conds.append(power_change)

        # Add legend
        plt.legend()

        # Calculate difference for each frequency
        p_s = np.zeros(len(freqs))
        for i_f, f in enumerate(freqs):
            z, p = scipy.stats.ttest_ind(power_change_conds[0][i_f], power_change_conds[1][i_f])
            p_s[i_f] = p
            if p < 0.01:
                plt.axvline(f, linewidth=2, alpha=0.2, color="grey")

        # Add title
        plt.title(ch)
        u.despine()

    # Adjust plot
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.suptitle(sub)
    plt.show()
