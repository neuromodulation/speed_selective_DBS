# Extract features using py_neuromodulation

import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
import py_neuromodulation as pn
import seaborn as sb
from py_neuromodulation import (
    nm_analysis,
    nm_define_nmchannels,
    nm_plots
)
import matplotlib
#matplotlib.use('Qt5Agg')

sys.path.insert(1, "C:/CODE/ac_toolbox/")

# Specify the medication group
med = "Off"

# Read the list of the datasets
df = pd.read_excel(f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'
       f'Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/'
       f'Tablet_task/Data/Dataset_list.xlsx', sheet_name=med)

# Define target folder fpr features
output_root = 'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'\
            f'Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/'\
            f'Tablet_task/Data/OFF/processed_data/'

# Loop through the subjects
for sub in df["ID Berlin_Neurophys"][1:21]:
    print(sub)

    if sub == "L003":
        pass
    else:

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

        # Drop bad channels
        raw.drop_channels(raw.info["bads"])

        # Add bipolar channels
        # TO-DO

        # Extract events
        events = mne.events_from_annotations(raw)[0]

        # Annotate periods with stimulation
        sample_stim = np.where(events[:, 2] == 10004)[0]
        # Get first stim in a block
        stim_onset_1 = events[sample_stim[0], 0]
        stim_onset_2 = events[sample_stim[np.argmax(np.diff(sample_stim)) + 1], 0]
        # Get last stim of block
        stim_offset_1 = events[sample_stim[np.argmax(np.diff(sample_stim))], 0]
        stim_offset_2 = events[sample_stim[-1], 0]

        # Get onset of break
        offset = events[np.where(events[:, 2] == 10001)[0], :]
        diff_break = np.diff(offset[:, 0])
        breaks_onset = []
        breaks_dur = []
        plt.figure()
        plt.plot(raw.get_data(picks=["SPEED_MEAN"]).flatten())
        for i, diff in enumerate(diff_break):
            if diff > 5000:
                break_onset = offset[i, 0]
                break_offset = offset[i+1, 0]
                plt.axvline(break_onset)
                plt.axvline(break_offset)
                breaks_onset.append(break_onset)
                breaks_dur.append(break_offset-break_onset)

        # Set annotation
        n_stim = 2 + len(breaks_onset)
        onset = np.array([stim_onset_1-2, stim_onset_2-2] + breaks_onset) / sfreq
        duration = np.array([stim_offset_1+10 - stim_onset_1, stim_offset_2+10 - stim_onset_2] + breaks_dur) / sfreq
        stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
        raw.set_annotations(stim_annot)

        # Inspect
        """
        plt.figure()
        plt.plot(raw.get_data(picks="dbs").mean(axis=0)[:-1000], color="red")
        plt.axvline(onset[0]*sfreq)
        plt.axvline(onset[1]*sfreq)
        plt.axvline((onset[0] + duration[0])*sfreq)
        plt.axvline((onset[1] + duration[1])*sfreq)
        """

        # Add channels
        # To-Do

        # Set channels
        # Get ephys channels
        ch_names_ephys = raw.copy().pick_types(ecog=True, dbs=True, eeg=True).info["ch_names"]
        ch_names_behavior = ["SPEED_MEAN"]
        ch_types = ["ecog"] * len(ch_names_ephys) + ["BEH"]
        nm_channels = nm_define_nmchannels.set_channels(ch_names=ch_names_ephys + ch_names_behavior, ch_types=ch_types, reference=None, target_keywords="SPEED")

        # Settings
        settings = pn.nm_settings.get_default_settings()
        settings = pn.nm_settings.reset_settings(settings)
        settings["features"]["fft"] = True

        stream = pn.Stream(
            settings=settings,
            nm_channels=nm_channels,
            verbose=True,
            sfreq=sfreq,
            line_noise=50
        )

        # Get the data of interest (without stimulation and only during task)
        task_start = offset = events[np.where(events[:, 2] == 10002)[0][0], 0] / sfreq
        task_end = offset = events[np.where(events[:, 2] == 10001)[0][-1], 0] / sfreq
        raw.crop(tmin=task_start, tmax=task_end)
        data = raw.get_data(picks=list(nm_channels["name"]), reject_by_annotation="omit")
        plt.figure()
        plt.plot(data[-1, :].flatten())
        #plt.show()

        # Compute features
        features = stream.run(data, folder_name=sub, out_path_root=output_root)