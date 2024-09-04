# Compute the time frequency plot for every subject and recording location
# Average over subjects
# Add cross-validation --> Gamma and beta as the most predictive markers

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
import matplotlib
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Specify the medication group
med = "Off"

# Specify if individual or only group plots are needed
plot_individual = False

# Specify the root folder
root_folder = 'C:/Users/ICN/Charité - Universitätsmedizin Berlin/Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/Tablet_task/'

# Set parameters for analysis
tmin = -0.3
tmax = 0
baseline = (-0.75, -0.5)
mode = "percent"
cmap = "jet"
freq_min = 20
freq_max = 100
frequencies = np.arange(freq_min, freq_max, 2)
target_names = ["STN Contralateral", "STN Ipsilateral", "ECOG Contralateral", "ECOG Ipsilateral"]

# Read the list of the datasets
df = pd.read_excel(f'{root_folder}Data/Dataset_list.xlsx', sheet_name=med)

# Loop through the subjects
subject_list = list(df["ID Berlin_Neurophys"][1:21])
subject_list.remove("L003")  # NO neurophys data available

power_all_sub = []
behav_all_sub = []
ps = np.zeros((len(subject_list), 4, 2))
for i_sub, sub in enumerate(subject_list):
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

    # Drop bad channels
    raw.drop_channels(raw.info["bads"])
    ch_names = raw.info["ch_names"]

    # Filter out line noise
    raw.notch_filter(50)

    # Add average LFP channels
    for loc in ["LFP_L", "LFP_R"]:
        target_chs = [ch for ch in ch_names if (loc in ch) and (not "01" in ch) and (not "08" in ch)]
        target_ch = f"av_{loc}"
        new_ch = raw.get_data(target_chs).mean(axis=0)
        u.add_new_channel(raw, new_ch[np.newaxis, :], target_ch, type="dbs")
    # Select the ecog target channel
    ECOG_target = df.loc[df["ID Berlin_Neurophys"] == sub]["ECOG_target"].iloc[0]
    if "E" in sub:
        if "R" in ECOG_target:
            target_ECOG_R = ECOG_target
            target_ECOG_L = ""
        elif "L" in ECOG_target:
            target_ECOG_L = ECOG_target
            target_ECOG_R = ""
    else:
        target_ECOG_R = ""
        target_ECOG_L = ""

    # ipsi/contralateral target channels
    if sub == "EL012" or sub == "L013":  # Left
        target_chs = ["av_LFP_R", "av_LFP_L", target_ECOG_R, target_ECOG_L]
    else:  # right
        target_chs = ["av_LFP_L", "av_LFP_R", target_ECOG_L, target_ECOG_R]

    # Extract events
    events = mne.events_from_annotations(raw)[0]

    # Annotate periods with stimulation
    sample_stim = events[np.where(events[:, 2] == 10004)[0], 0]
    n_stim = len(sample_stim)
    onset = (sample_stim / sfreq) - 0.1
    duration = np.repeat(0.5, n_stim)
    stim_annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim, orig_time=raw.info['meas_date'])
    raw.set_annotations(stim_annot)

    # Compute time-frequency spectrum aligned to peak speed
    event_id = 10003

    # Cut into epochs
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2, baseline=None, reject_by_annotation=True)
    epochs.drop_bad()

    # Extract peak speed
    peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

    # Exclude outliers
    peak_speed = u.fill_outliers_nan(peak_speed)

    for i_ch, target_ch in enumerate(target_chs[2:3]):
        if target_ch != "":
            # Compute the tfr
            power = mne.time_frequency.tfr_morlet(epochs, n_cycles=7,  picks=[raw.info["ch_names"].index(target_ch)],
                                              return_itc=False, freqs=frequencies, average=False, use_fft=True)
            # Apply baseline correction using the defined method
            #power.apply_baseline(baseline=baseline, mode=mode)
            # Crop in window of interest
            power.crop(tmin=tmin, tmax=tmax)

            """# Reduce in the frequency domain
            power_matrix = power.data.copy()
            #power_matrix = np.apply_along_axis(u.average_windows, axis=2, arr=power_matrix, n_steps=4)

            # Reduce in the time domain
            #power_matrix = np.apply_along_axis(u.average_windows, axis=-1, arr=power_matrix, n_steps=8)

            # Applying PCA
            n_trials, _, n_freq, n_t = power.data.shape
            power_matrix = power.data.copy().reshape((n_trials, n_freq * n_t))
            pca = PCA(n_components=30)
            X = pca.fit_transform(power_matrix)"""

            # Extract features (maximum synchronization in the gamma and desynchronization in the beta band)
            beta = np.percentile(power.copy().crop(fmin=30, fmax=45).data, 10, axis=(-2, -1)).squeeze()
            gamma = np.percentile(power.copy().crop(fmin=75, fmax=95).data, 90, axis=(-2, -1)).squeeze()
            #beta = np.median(power.copy().crop(fmin=20, fmax=40).data, axis=(-2, -1)).squeeze()
            #gamma = np.median(power.copy().crop(fmin=70, fmax=90).data, axis=(-2, -1)).squeeze()
            X = np.vstack((gamma, gamma)).T

            # Train linear model to predict peak speed from tfr
            y = peak_speed

            # Delete Nan values (outliers)
            X = X[~np.isnan(y), :]
            y = y[~np.isnan(y)]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            # Train
            y_pred = model.predict(X_train)
            r2 = sklearn.metrics.r2_score(y_train, y_pred)
            plt.scatter(y_pred, y_train)
            corr, p = spearmanr(y_train, y_pred)
            ps[i_sub, i_ch, 0] = p
            # Test
            y_pred = model.predict(X_test)
            r2 = sklearn.metrics.r2_score(y_test, y_pred)
            #plt.scatter(y_pred, y_test)
            #plt.title(spearmanr(y_test, y_pred))
            #mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
            corr, p = spearmanr(y_test, y_pred)
            ps[i_sub, i_ch, 1] = p
            #plt.show()
# Analyze
p_test = ps[:, :, 1]
print("END")

