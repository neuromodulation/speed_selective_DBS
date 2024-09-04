# Save the outcome measure (e.g. difference in peak speed between slow and fast stimulation) for correaltion analysis

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
from scipy.io import savemat
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
feature_name = "mean_speed"
mode = "diff"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
feature = u.fill_outliers_nan(feature)

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))

# Delete the first n_cutoff movements
feature = feature[:, :, n_cutoff:]

if n_norm != 0:
    # Normalize to average of the first n_norm movements
    feature = u.norm_perc(feature, n_norm=n_norm)

# Compute the difference
res = np.zeros((n_datasets, 2))

for block in range(2):

    # Get the movements of the stimulation/recovery block
    if block == 0:
        feature_block = feature[:, :, n_norm:n_trials-n_cutoff]
    else:
        feature_block = feature[:, :, -n_trials:]

    # Apply method (mean/median..) to summarize the feature in the block
    if method == "median":
        feature_tmp = np.nanmedian(feature_block, axis=-1)
    elif method == "mean":
        feature_tmp = np.nanmean(feature_block, axis=-1)

    # Combine the conditions
    if mode == "diff" or mode == "mean":
        res[:, block] = feature_tmp[:, 1] - feature_tmp[:, 0]
    elif mode == "all":
        res[:, block] = np.nanmean(feature_tmp, axis=1)
    elif mode == "slow":
        res[:, block] = feature_tmp[:, 0]
    elif mode == "fast":
        res[:, block] = feature_tmp[:, 1]

# Average stimulation and recovery
if mode == "mean":
    res_mean = np.nanmean(res, axis=-1)
    res[:, 0] = res_mean
    res[:, 1] = res_mean

# Save matrix
np.save(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy", res)
# As mat file for imaging analysis
savemat(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.mat", {"res": res})


