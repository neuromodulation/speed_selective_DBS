# Check outcome measure and make sure no outliers are contained anymore

# Import useful libraries
import os
import sys
sys.path.insert(1, "../Code")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "mean_speed"

# Set normalization window and cut-off of interest
cutoff = 5
n_norm = 5

# set medication state
med = "Off"
# set normalization
method = "normalized"
# Set averaging
av = "mean"
# set smoothing
smooth = True

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
labels = ["Slow", "Fast"]

# Load matrix containing the feature values and the stimulated trials
feature_all = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature_all.shape

# Reshape matrix such that blocks from one condition are concatenated
feature_all = np.reshape(feature_all, (n_datasets, 2, n_trials * 2))
feature_all_outlier = u.fill_outliers_nan(feature_all.copy())

# Loop over patients
for i in range(n_datasets):

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(feature_all[i, :, :].T, label="raw", color="red", alpha=0.5)

    feature = feature_all_outlier[i, :, :]
    plt.plot(feature.T, label="outliers", color="black", alpha=0.5)

    if method == "mean":
        # Delete the first 5 movements
        feature = feature[:, cutoff:]
        # Average over the next 5 movements
        feature = np.nanmean(feature, axis=1)
        plt.plot(feature.T, label="mean", color="orange", alpha=0.5)

    if method == "median":
        # Delete the first 5 movements
        feature = feature[:, cutoff:]
        # Median over the next 5 movements
        feature = np.nanmedian(feature, axis=1)
        plt.plot(feature.T, label="median", color="orange", alpha=0.5)

    if method == "normalized":
        # Delete the first 5 movements
        feature = feature[:, cutoff:]
        # Normalize to average of the next 5 movements
        feature = u.norm_perc(feature, n_norm=n_norm)

        plt.subplot(1, 2, 2)
        plt.plot(feature.T, label="normalized", color="blue", alpha=0.5)

    # Smooth over 5 consecutive movements for plotting
    if smooth:
        feature = u.smooth_moving_average(feature, window_size=5, axis=1)
        plt.plot(feature.T, label="smoothed", color="green", alpha=0.5)

    plt.legend()
plt.show()