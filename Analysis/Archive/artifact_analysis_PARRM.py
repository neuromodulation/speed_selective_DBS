# Check whether the artifact can be removed with PARRM

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from pyparrm import get_example_data_paths, PARRM
from pyparrm._utils._power import compute_psd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Qt5Agg')

# Load data
data = np.load("burst_stim.npy")

# Crop data such that the array contains artifact only
data = data[675:1875]
fig, axes = plt.subplots(3, 1, figsize=(15, 10))
axes[0].plot(data)

# PARRM needs a two-dimensional input
data = data[np.newaxis, :]

# Define parameters
sampling_freq = 4000
artefact_freq = 130

# Initialize PARRM
parrm = PARRM(
    data=data,
    sampling_freq=sampling_freq,
    artefact_freq=artefact_freq,
    verbose=False,
)
parrm.find_period()

# Explore the filter parameters
#parrm.explore_filter_params()

# Create and apply filter
parrm.create_filter(period_half_width=0.036, filter_half_width=599)
filtered_data = parrm.filter_data()

# Plot filtered data
axes[1].plot(filtered_data[0])

def bandpass_filter(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    # Design the filter
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')

    # Apply the filter to the data
    filtered_data = scipy.signal.sosfiltfilt(sos, data)

    return filtered_data


# Example usage
sample_rate = 4000  # Replace with your actual sample rate
edges = [2, 90]  # Specify the frequency range (in Hz)

bandpass_data = bandpass_filter(filtered_data[0], edges, sample_rate)
axes[2].plot(bandpass_data)
plt.show(block=True)

