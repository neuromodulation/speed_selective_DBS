# Script for calculating the rquired sample size (power analysis)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
def statistic(x, y):
    return np.mean(x) - np.mean(y)

# Set analysis parameters
feature_name = "peak_speed"
med = "OFF"  # "on", "off", "all"

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")
feature_matrix = feature_matrix[1:, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Detect and fill outliers (e.g. when subject did not touch the screen)
np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature_matrix)

# Reshape matrix such that blocks from one condition are concatenated
feature_matrix = np.reshape(feature_matrix, (n_datasets, 2, n_trials*2))

# Delete the first 5 movements
feature_matrix = feature_matrix[:, :, 5:]

# Normalize to average of first 5 movements
feature_matrix = u.norm_perc(feature_matrix)

# Median over all movements in that period
feature_matrix_mean = np.nanmean(feature_matrix[:, :, :91], axis=2)

plt.figure()
plt.hist(feature_matrix_mean, alpha=0.5)

# Create sampling function
def sample(array):
    # Draw random sample from data array
    i = np.random.randint(0, len(array))
    val = array[i]
    # Sample from gaussian bump located on sample
    return np.random.normal(val, 3)

# Specify power, alpha level
power_level = 0.8
alpha = 0.05

# Test
x = []
x2 = []
for i in range(1000):
    x.append(sample(feature_matrix_mean[:, 0]))
    x2.append(sample(feature_matrix_mean[:, 1]))
plt.hist(x, alpha=0.5)
plt.hist(x2, alpha=0.5)
#plt.show()

# Compute necessary sample size
n_start = 5
n_sample = n_start
power = 0
n_sim = 100
power_at = []
while power < power_level:
    n_sample += 1
    p_h0 = np.zeros(n_sim)
    for sim in range(n_sim):
        x = [sample(feature_matrix_mean[:, 0]) for i in range(n_sample)]
        y = [sample(feature_matrix_mean[:, 1]) for i in range(n_sample)]
        res = scipy.stats.permutation_test(data=(x, y), statistic=statistic,
                                       permutation_type="samples")
        p_h0[sim] = res.pvalue
    power = np.sum(p_h0 < alpha)/n_sim
    power_at.append(power)
    print(f"{n_sample} with power {power}")

plt.figure()
plt.plot(power_at)
print("END")


