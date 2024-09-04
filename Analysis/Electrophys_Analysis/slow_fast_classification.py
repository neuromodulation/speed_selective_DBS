# Classify a trial either as slow or fast

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
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import minmax_scale
matplotlib.use('Qt5Agg')

# Set the parameters
sub = "EL012"
med = "Off"

# Load the electrophysiological data
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
new_ch = raw.get_data("ECOG_R_02_SMC_AT") - raw.get_data("ECOG_R_03_SMC_AT")
target_chan_name = "ECOG_bipolar"
u.add_new_channel(raw, new_ch, target_chan_name, type="ecog")
ch_names = raw.info["ch_names"]

# Filter out line noise
raw.notch_filter(50)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Plot time-frequency spectrum for all events
event_id = 10003

# Cut into epochs
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=0.5, baseline=None, reject_by_annotation=False)
epochs.load_data()

# Get the fast and slow movements indexes
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()
labels = np.zeros(len(peak_speed))
for i, ps in enumerate(peak_speed):
    if i > 1 and np.all(ps < peak_speed[i-2:i]):
        labels[i] = 1
    elif i > 1 and np.all(ps > peak_speed[i-2:i]):
        labels[i] = 2


# Calculate the features
freq_min = 15
freq_max = 90
frequencies = np.arange(freq_min, freq_max, 2)
tfr = epochs.compute_tfr(method="multitaper", freqs=frequencies, picks=target_chan_name, average=False)
bands = [(15, 35), (35, 60), (60, 90)]
features = []
for i, band in enumerate(bands):

    # Compute power in frequency band
    power = tfr.get_data(fmin=band[0], fmax=band[-1], tmin=-0.5, tmax=0).mean(axis=(2, 3)).flatten()
    feature_1 = power[2:] - power[1:-1]
    feature_2 = power[2:] - power[:-2]
    features.extend(np.vstack((feature_1, feature_2)))

# Train a classifier
X = np.array(features).T * 1e9
y = np.array(labels[2:])

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(multi_class="multinomial", random_state=0, class_weight="balanced").fit(X, y)
pred = clf.predict(X)
prob = clf.predict_proba(X)
print(np.sum(y == pred) / len(y))

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
import random
random.shuffle(y)
cross_val_score(clf, X, y, cv=10)

print("what is wrong here?")

