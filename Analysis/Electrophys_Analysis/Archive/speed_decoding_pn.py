# Decode speed using py_neuromodulation

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
import random
import pandas as pd
from sklearn import metrics, model_selection, linear_model
import py_neuromodulation as nm
from catboost import CatBoostRegressor, Pool
from xgboost import XGBClassifier
from scipy.stats import zscore
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
)
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import matplotlib
matplotlib.use('Qt5Agg')
random.seed(420)

# Set analysis parameters
samp_freq = 37
seg_ms = 354

# Loop over blocks (break in between during which behavior is not recorded but ephys recording continues)
feature_reader = nm_analysis.FeatureReader(
    feature_dir="..\\..\\..\\Data\\Off\\processed_data\\",
    feature_file=f"{samp_freq}_{seg_ms}",
)

feature_reader.plot_all_features(ch_used="ECOG_R_01_SMC_AT", clim_high=4, clim_low=-4)

# Set the label
feature_reader.label_name = "SPEED_MEAN"
feature_reader.label = feature_reader.feature_arr[feature_reader.label_name]

# Normalize between 0 and 1
#feature_reader.label = (feature_reader.label - np.min(feature_reader.label)) / (np.max(feature_reader.label) - np.min(feature_reader.label))
#feature_reader.label = zscore(feature_reader.label)

#model = linear_model.LinearRegression()
#model = linear_model.Ridge()
#model = XGBClassifier()
model = CatBoostRegressor(iterations=50,
                          depth=5,
                          loss_function='RMSE', learning_rate=0.46)
#model = linear_model.LinearRegression()

feature_reader.decoder = nm_decode.Decoder(
    features=feature_reader.feature_arr,
    label=feature_reader.label,
    label_name=feature_reader.label_name,
    used_chs=feature_reader.used_chs,
    STACK_FEATURES_N_SAMPLES=True,
    time_stack_n_samples=20,
    model=model,
    eval_method=metrics.r2_score,
    cv_method=model_selection.KFold(n_splits=4),
    VERBOSE=True
)

performances = feature_reader.run_ML_model(
    estimate_channels=True,
    estimate_gridpoints=False,
    estimate_all_channels_combined=True,
    save_results=True,
)

df_per = feature_reader.get_dataframe_performances(performances)
print(df_per)

# Plot the predicted speed
channel = "all_ch_combined"
for i in range(3):
    y_test = feature_reader.decoder.all_ch_results["y_test"][i]
    y_test_pr = feature_reader.decoder.all_ch_results["y_test_pr"][i]
    y_train = feature_reader.decoder.all_ch_results["y_train"][i]
    y_train_pr = feature_reader.decoder.all_ch_results["y_train_pr"][i]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(y_test), color="black", label="speed")
    ax1.plot(y_test_pr, color="red", label="predicted speed")
    ax1.set_title("Test")
    ax1.legend()
    ax2.plot(np.array(y_train), color="black", label="speed")
    ax2.plot(y_train_pr, color="red", label="predicted speed")
    ax2.set_title("Training")
    ax2.legend()
    plt.suptitle(f"Cross-validation run {i}")
plt.show(block=True)

print("test")
