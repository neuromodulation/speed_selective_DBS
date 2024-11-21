# Train and save the model (for plotting of the predicted test traces)

import os
from joblib import Parallel, delayed
import numpy as np
import mne
import py_neuromodulation as nm
from bayes_opt import BayesianOptimization
from sklearn import metrics, model_selection
from sklearn.model_selection import KFold
import pandas as pd
from catboost import CatBoostRegressor
from openpyxl import Workbook
from openpyxl import load_workbook
import random
import shutil
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, "../../../Code")
import utils as u
random.seed(420)
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
setting = "LFP_combined"
fold = 0
n_folds = 8

# Load the optimal parameters
filename = f"results_4/feature_model_optimization_{setting}_{fold}.xlsx"
df = pd.read_excel(filename, sheet_name=f"Fold {fold}")
samp_freq, seg_ms, n_stack, depth, learning_rate, _ = df.loc[len(df)-1]

# Load the data
path = "EL012_ECoG_CAR_LFP_BIP_small.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
ecog_names = [name for name in ch_names if "ECOG" in name]
lfp_names = [name for name in ch_names if "LFP" in name]
events = mne.events_from_annotations(raw)[0]

# Set channels
if "combined" not in setting:
    ch_names = [setting, "SPEED_MEAN"]
    ch_types = ["ecog", "BEH"]

elif setting == "LFP_combined":
    ch_names = lfp_names + ["SPEED_MEAN"]
    ch_types = ["ecog"] * len(lfp_names) + ["BEH"]

elif setting == "ECoG_LFP_combined":
    ch_names = ecog_names + lfp_names + ["SPEED_MEAN"]
    ch_types = ["ecog"] * len(ecog_names + lfp_names) + ["BEH"]

elif setting == "ECoG_combined":
    ch_names = ecog_names + ["SPEED_MEAN"]
    ch_types = ["ecog"] * len(ecog_names) + ["BEH"]

channels = nm.utils.set_channels(
    ch_names=ch_names,
    ch_types=ch_types,
    reference=None,
    used_types=ch_types[:-1],
    target_keywords=["SPEED_MEAN"],
)

# Attach the blocks together
blocks = []
for i in range(4):
    tmin = events[np.where((events[:, 2] == 2) | (events[:, 2] == 10002))[0], 0][int(96 * i)] / sfreq
    tmax = events[np.where((events[:, 2] == 1) | (events[:, 2] == 10001))[0], 0][int(96 * (i + 1)) - 1] / sfreq
    block = raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=ch_names)
    blocks.append(block)
blocks_all = np.hstack((blocks[0], blocks[1], blocks[2], blocks[3]))

# Set analysis parameters
samp_freq = int(samp_freq)
seg_ms = int(seg_ms)
settings = nm.NMSettings.get_fast_compute()
settings.features.fft = True
settings.features.return_raw = True
settings.sampling_rate_features_hz = samp_freq
settings.segment_length_features_ms = seg_ms
settings.fft_settings.windowlength_ms = seg_ms
del settings.frequency_ranges_hz["theta"]
settings.postprocessing.feature_normalization = True
settings.feature_normalization_settings.normalization_time_s = 1
settings.feature_normalization_settings.normalization_method = "zscore"
settings.preprocessing = settings.preprocessing[:2]

# Compute features
stream = nm.Stream(
    settings=settings,
    channels=channels,
    verbose=False,
    sfreq=sfreq,
    line_noise=50
)

data = blocks_all

stream.run(data=data, out_dir="", experiment_name=f"optimization_{setting}_vis")
feature_reader = nm.analysis.FeatureReader(
    feature_dir="",
    feature_file=f"optimization_{setting}_vis"
)

# Set the label
feature_reader.label_name = "SPEED_MEAN"
feature_reader.label = feature_reader.feature_arr[feature_reader.label_name]

# Setup the model and train
model = CatBoostRegressor(iterations=30,
                          depth=int(depth),
                          learning_rate=learning_rate
                          )
feature_reader.decoder = nm.analysis.Decoder(
                    features=feature_reader.feature_arr,
                    label=feature_reader.label,
                    label_name=feature_reader.label_name,
                    used_chs=feature_reader.used_chs,
                    STACK_FEATURES_N_SAMPLES=True,
                    time_stack_n_samples=int(n_stack),
                    model=model,
                    eval_method=metrics.r2_score,
                    cv_method=model_selection.KFold(n_splits=n_folds),
                    VERBOSE=True
                )
estimate_combined = True if "combined" in setting else False
estimate_channels = False if "combined" in setting else True
performances = feature_reader.run_ML_model(
    estimate_channels=estimate_channels,
    estimate_gridpoints=False,
    estimate_all_channels_combined=estimate_combined,
    save_results=True,
)
df_per = feature_reader.get_dataframe_performances(performances)
perf = np.array(df_per["performance_test"])[-1]