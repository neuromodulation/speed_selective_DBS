# Extract power in specific frequency band for later analysis

import numpy as np
import matplotlib
import os
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import mne
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
import pandas as pd
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import logging

# Set the logging level to a higher level (e.g., WARNING) to suppress MNE output
logging.getLogger('mne').setLevel(logging.WARNING)

# Set parameters
med = "Off"
bands = [[4, 8], [8, 12], [13, 35], [13, 20], [20, 35], [60, 200], [60, 80], [90, 200]]
n_trials = 96

# Load the excel sheet containing the phenotype data
df = pd.read_excel(f'../../Data/Dataset_list.xlsx', sheet_name=med)
subject_list = list(df["ID Berlin_Neurophys"][1:25])
hand = list(df["Hand"][1:25])
remove_idx = subject_list.index("L003")
subject_list.pop(remove_idx)
hand.pop(remove_idx)  # NO neurophys data available

# Loop over subjects
for i, sub in enumerate(subject_list):

    print(sub)

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
    #raw.notch_filter(50)

    # Drop bad channels
    raw.drop_channels(raw.info["bads"])
    ch_names = raw.info["ch_names"]

    # Extract events
    events = mne.events_from_annotations(raw)[0]

    # Annotate periods with stimulation
    sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
    n_stim = len(sample_stim)
    onset = (sample_stim / sfreq) - 0.1
    duration = np.repeat(0.5, n_stim)
    stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
    raw.set_annotations(stim_annot)

    #raw.plot()

    # Compute LFP contact of interest
    # ?? Make LFP bipolar ??
    for loc in ["LFP_L", "LFP_R"]:
        print([ch for ch in ch_names if (loc in ch)])
        target_chs = [ch for ch in ch_names if (loc in ch) and (not "01" in ch) and (not "08" in ch)]
        if hand[i] == "R" and loc == "LFP_L" or hand[i] == "L" and loc == "LFP_R":
            target_ch = "contra_LFP"
        else:
            target_ch = "ipsi_LFP"
        new_ch = raw.get_data(target_chs).mean(axis=0)

        # Compute bipolar channel
        lower = any([any(idx in ch for idx in ["02", "03", "04"]) for ch in target_chs])
        upper = any([any(idx in ch for idx in ["05", "06", "07"]) for ch in target_chs])
        if f"{loc}_08_STN_MT" in ch_names or f"{loc}_01_STN_MT" in ch_names:
            if lower and f"{loc}_08_STN_MT" in ch_names:
                new_ch = new_ch - raw.get_data([f"{loc}_08_STN_MT"])
            elif lower and f"{loc}_01_STN_MT" in ch_names:
                new_ch = new_ch - raw.get_data([f"{loc}_01_STN_MT"])
            elif upper and f"{loc}_01_STN_MT" in ch_names:
                new_ch = new_ch - raw.get_data([f"{loc}_01_STN_MT"])
            elif upper and f"{loc}_08_STN_MT" in ch_names:
                new_ch = new_ch - raw.get_data([f"{loc}_08_STN_MT"])
            # Add channel
            u.add_new_channel(raw, new_ch, target_ch, type="dbs")

    # Compute the LFP contact of interest
    ECOG_target_1 = df.loc[df["ID Berlin_Neurophys"] == sub]["ECOG_target_1"].iloc[0]
    ECOG_target_2 = df.loc[df["ID Berlin_Neurophys"] == sub]["ECOG_target_2"].iloc[0]
    if "E" in sub:
        # Compute bipolar channel
        new_ch = np.diff(raw.get_data([ECOG_target_1, ECOG_target_2]), axis=0)
        if "R" in ECOG_target_1 and hand[i] == "R" or "L" in ECOG_target_1 and hand[i] == "L":
            target_ch = "ipsi_ECOG"
        else:
            target_ch = "contra_ECOG"
        u.add_new_channel(raw, new_ch, target_ch, type="dbs")

    # Cut into epochs
    event_id = 10003
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-0.55, tmax=-0.3, baseline=None,
                        reject_by_annotation=False)
    #epochs.plot(block=True, picks=raw.info["ch_names"][-5:])

    # Save index of bad epochs
    """epochs_samples = list(epochs.selection.copy())
    epochs.drop_bad()
    drop_sample = np.where(list(map(lambda x: x == ('bad stim',), epochs.drop_log)))[0]
    drop_idx = np.where(epochs_samples == drop_sample)[0]
    drop_idx = np.array([epochs_samples.index(samp) for samp in drop_sample])"""

    # Loop over bands of interest
    epochs.load_data()
    ch_names = raw.info["ch_names"]
    for band in bands:
        for channel in ["contra_LFP", "ipsi_LFP", "contra_ECOG", "ipsi_ECOG"]:
            if channel in ch_names:
                # Compute one matrix for each band and channel
                res = np.zeros((2, 2, n_trials))

                # Compute power in frequency band
                psds, freqs = epochs.compute_psd(fmin=band[0], fmax=band[-1], picks=channel, method='multitaper').get_data(
                    return_freqs=True)
                power = np.mean(psds, axis=-1).flatten()

                # Cut into conditions and blocks
                cond_id = np.unique(epochs.get_data(["STIM_CONDITION"]), axis=-1).flatten()
                if cond_id[0] == 0:
                    res[0, 0, :] = power[:n_trials]
                    res[0, 1, :] = power[n_trials:n_trials*2]
                    res[1, 0, :] = power[n_trials*2:n_trials*3]
                    res[1, 1, :] = power[-n_trials:]
                else:
                    res[1, 0, :] = power[:n_trials]
                    res[1, 1, :] = power[n_trials:n_trials*2]
                    res[0, 0, :] = power[n_trials*2:n_trials*3]
                    res[0, 1, :] = power[-n_trials:]
                """if channel == "contra_LFP" and band[0] == 60 and band[-1] == 80:
                    plt.plot(power)
                    plt.show()"""

                # Save in file
                np.save(f"../../Data/{med}/processed_data/ephys/power_{sub}_{band[0]}_{band[-1]}_{channel}.npy", res)

# Load and add together in order to get one matrix containing the data for all subjects
subject_list = list(df["ID Berlin_Neurophys"][1:25])
for band in bands:
    for channel in ["contra_LFP", "ipsi_LFP", "contra_ECOG", "ipsi_ECOG"]:
        res = np.zeros((len(subject_list), 2, 2, n_trials))
        for i, sub in enumerate(subject_list):
            try:
                res[i, :, :, :] = np.load(f"../../Data/{med}/processed_data/ephys/power_{sub}_{band[0]}_{band[-1]}_{channel}.npy")
            except:
                res[i, :, :, :] = np.zeros((2, 2, n_trials)) * np.nan

        np.save(f"../../Data/{med}/processed_data/power_{band[0]}_{band[-1]}_{channel}.npy", res)