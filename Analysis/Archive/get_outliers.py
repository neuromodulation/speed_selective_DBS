# Detect/define all outliers and save index as numpy file

# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import mplcursors
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')

# Set feature to analyze
feature_name = "median_speed_raw"
threshold = 3
med = "Off"

# Load matrix containing the feature values and the stimulated trials
feature_all = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
n_datasets, _, _, n_trials = feature_all.shape

# Reshape matrix such that blocks from one condition are concatenated
feature_all = np.reshape(feature_all, (n_datasets, 2, n_trials * 2))

outliers_all = []
# Loop over subject and condition
for i in range(n_datasets):
    outliers_dataset = []
    for j in range(2):
        # Get the data
        array = feature_all[i, j, :]

        # Preselect outliers based on z-score
        idx_outlier = np.where(np.abs(zscore(array, nan_policy='omit')) > threshold)[0]

        # plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(array)
        for idx in idx_outlier:
            ax1.text(idx, array[idx], "*", color="red", fontsize=18, ha="center", va="center")

        # Plot to select
        scatter = ax2.scatter(np.arange(len(array)), array)
        # List to store selected point indices
        selected_points = []

        # Function to handle click events
        def on_click(event):
            if event.button == 1:  # Check for left mouse button click
                if scatter.contains(event)[0]:
                    ind = scatter.contains(event)[1]["ind"][0]
                    selected_points.append(ind)
                    print(f"Selected point: {ind}")

        fig.canvas.mpl_connect('button_press_event', on_click)
        mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Point {sel.target.index}'))

        plt.show()

        print("Indices of selected points:", selected_points)

        # Append outliers
        outliers_dataset.append(selected_points)

    # Append outliers
    outliers_all.append(outliers_dataset)

# Save outliers
np.save(f"../../../Data/{med}/processed_data/{feature_name}_outliers.npy", outliers_all,  allow_pickle=True)