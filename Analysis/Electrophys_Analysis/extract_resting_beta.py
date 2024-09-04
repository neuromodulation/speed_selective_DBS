# Extract the average resting beta activity recorded on the same day

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

# Load the excel sheet containing the phenotype data
med = "Off"
fmin = 15
fmax = 40
df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
subject_list = list(df["ID Berlin_Neurophys"][1:25])
day_rest_recording = list(df["Rest"][1:25])
stim_contact_right = list(df["Stimulation contact right"][1:25])
stim_contact_left = list(df["Stimulation contact left"][1:25])
power_all = []

for i, sub in enumerate(subject_list):

    acquisitions = "StimOff"
    if day_rest_recording[i] == 1:
        sessions = [f"LfpMed{med}01", f"EcogLfpMed{med}01"]
    elif day_rest_recording[i] == 2:
        sessions = [f"LfpMed{med}02", f"EcogLfpMed{med}02"]
    elif day_rest_recording[i] == 3:
        sessions = [f"LfpMed{med}Dys01"]
        acquisitions = "StimOffDopaPre"
    else:
        continue

    # Load the electrophysiological data converted to BIDS (brainvision) from the raw data folder
    root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
           f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
           f'rawdata/'

    bids_paths = find_matching_paths(root, tasks=["Rest"],
                                        extensions=".vhdr",
                                        acquisitions=acquisitions,
                                        subjects=sub,
                                        sessions=sessions)
    if sub == "L014":
        raw = read_raw_bids(bids_path=bids_paths[-1])
    else:
        raw = read_raw_bids(bids_path=bids_paths[0])
    raw.load_data()
    sfreq = raw.info["sfreq"]

    # Select the channels used for stimulation
    if stim_contact_right[i] == '2,3,4':
        picks_right = ['LFP_R_02_STN_MT', 'LFP_R_03_STN_MT', 'LFP_R_04_STN_MT']
    elif stim_contact_right[i] == '5,6,7':
        picks_right = ['LFP_R_05_STN_MT', 'LFP_R_06_STN_MT', 'LFP_R_07_STN_MT']
    elif stim_contact_right[i] == '7,8,9':
        picks_right = ['LFP_R_07_STN_BS', 'LFP_R_08_STN_BS', 'LFP_R_09_STN_BS']
    if stim_contact_left[i] == '2,3,4':
        picks_left = ['LFP_L_02_STN_MT', 'LFP_L_03_STN_MT', 'LFP_L_04_STN_MT']
    elif stim_contact_left[i] == '5,6,7':
        picks_left = ['LFP_L_05_STN_MT', 'LFP_L_06_STN_MT', 'LFP_L_07_STN_MT']
    elif stim_contact_left[i] == '7,8,9':
        picks_left = ['LFP_L_07_STN_BS', 'LFP_L_08_STN_BS', 'LFP_L_09_STN_BS']

    picks = picks_right + picks_left
    print(picks)
    raw.pick_channels(picks)

    # Crop
    raw.crop(10, raw.tmax-10)

    # Compute power in beta band
    psds, freqs = raw.compute_psd(fmin=fmin, fmax=fmax, method='multitaper').get_data(return_freqs=True)
    power = np.mean(psds, axis=-1).flatten()
    print(power)
    power_all.append(np.vstack((power[:3], power[3:])))

# Save the extracted power values
power_all = np.array(power_all)

# Save the extracted power values
np.save(f'../../../Data/{med}/processed_data/power_{fmin}_{fmax}.npy', power_all)

# Correlate with behavioral effect
feature_name = "peak_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5
# Load matrix containing the outcome measure
x = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")

# entry for missing subject
x = np.delete(x, 10, axis=0)

# Average power over contacts
y = np.mean(power_all, axis=(1, 2))

# Correlate
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    corr, p = pearsonr(x[:, i], y)
    p = np.round(p, 3)
    if p < 0.05:
        label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
    else:
        label = f" R = {np.round(corr, 2)} p = {p}"
    plt.subplot(1, 2, i+1)
    sb.regplot(x=x[:, i], y=y, label=label, scatter_kws={"color": "indianred"}, line_kws={"color": "teal"})