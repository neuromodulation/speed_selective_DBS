# Decode speed using py_neuromodulation

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
import py_neuromodulation as nm
from sklearn import metrics, model_selection, linear_model
import py_neuromodulation as nm
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
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Loop over blocks (break in between during which behavior is not recorded but ephys recording continues)
for i_block in range(4):

    feature_reader = nm_analysis.FeatureReader(
        feature_dir="../../../Data/Off/processed_data\\",
        feature_file=f"block_{i_block}",
    )

    feature_reader.plot_all_features(ch_used="ECOG_bipolar", clim_high=4, clim_low=-4)
    #plt.show()

    feature_reader.label_name = "SPEED_MEAN"
    feature_reader.label = feature_reader.feature_arr["SPEED_MEAN"]
    #feature_reader.sfreq = 50

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(feature_reader.label).flatten(), color="red")
    ax2.hist(np.array(feature_reader.label).flatten(), bins=100, alpha=0.5)
    plt.close()

    # Transform/adjust the features
    # Replace 0 values with nan
    #idx_0 = np.where(feature_reader.label == 0)[0]
    use_idx = np.where(feature_reader.label != 0)[0][100:-100]
    feature_reader.label = feature_reader.label[use_idx].reset_index(drop=True)
    #np.random.shuffle(feature_reader.label)
    # Normalize between 0 and 1
    #feature_reader.label = (feature_reader.label - np.min(feature_reader.label)) / (np.max(feature_reader.label) - np.min(feature_reader.label))
    feature_reader.label = zscore(feature_reader.label)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(feature_reader.label).flatten())
    ax2.hist(np.array(feature_reader.label).flatten(), bins=100, alpha=0.5)
    #plt.show()
    plt.close()

    # Delete the 0 value for all features
    feature_reader.feature_arr = feature_reader.feature_arr.loc[use_idx, :].reset_index(drop=True)

    model = linear_model.LinearRegression()
    #model = linear_model.Lasso()
    #model = XGBClassifier(max_depth=6)

    feature_reader.decoder = nm_decode.Decoder(
        features=feature_reader.feature_arr,
        label=feature_reader.label,
        label_name=feature_reader.label_name,
        used_chs=feature_reader.used_chs,
        model=model,
        eval_method=metrics.r2_score,
        cv_method=model_selection.KFold(n_splits=10),
    )

    performances = feature_reader.run_ML_model(
        estimate_channels=True,
        estimate_gridpoints=False,
        estimate_all_channels_combined=False,
        save_results=True,
    )

    df_per = feature_reader.get_dataframe_performances(performances)
    print(df_per)

    # Plot the predicted speed
    for i in range(3):
        y_test = feature_reader.decoder.ch_ind_results["ECOG_bipolar"]["y_test"][i]
        y_test_pr = feature_reader.decoder.ch_ind_results["ECOG_bipolar"]["y_test_pr"][i]
        y_train = feature_reader.decoder.ch_ind_results["ECOG_bipolar"]["y_train"][i]
        y_train_pr = feature_reader.decoder.ch_ind_results["ECOG_bipolar"]["y_train_pr"][i]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(np.array(y_test), color="black", label="speed")
        ax1.plot(y_test_pr, color="red", label="predicted speed")
        ax1.set_title("Test")
        ax1.legend()
        ax2.plot(np.array(y_train), color="black", label="speed")
        ax2.plot(y_train_pr, color="red", label="predicted speed")
        ax2.set_title("Training")
        ax2.legend()
        plt.suptitle(f"block {i_block} Cross-validation run {i}")
        plt.close()
plt.show(block=True)

print("test")
