# Plot the feature over time (to get an exemplar trace showing the paradigm)

# Import useful libraries
import os
import sys
sys.path.append('../Code')
import utils as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')

# Set parameters
feature_name = "peak_speed"
med = "Off"
method = "mean"
n_norm = 0
n_cutoff = 0

# Load matrix containing the feature values and the stimulated trials
feature = np.load(f"../../../Data/{med}/processed_data/{feature_name}.npy")
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
n_datasets, _, _, n_trials = feature.shape

# Get the patient information
path_table = "C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - " \
             "PROJECT ReinforceVigor\\vigor_stim_task\\Data\\Dataset_list.xlsx"
df_info = pd.read_excel(path_table)
IDs = df_info["ID Berlin_Neurophys"][1:25].to_numpy()
cond = df_info["Block order"][1:25].to_numpy()

# Reshape matrix such that blocks from one condition are concatenated
feature = np.reshape(feature, (n_datasets, 2, n_trials * 2))
stim = np.reshape(stim, (n_datasets, 2, n_trials * 2))

# Delete the first n_cutoff movements
feature = feature[:, :, n_cutoff:]
stim = stim[:, :, n_cutoff:]

if n_norm != 0:
    # Normalize to average of the first n_norm movements
    feature = u.norm_perc(feature, n_norm=n_norm)

# Prepare plotting
colors = ["#00863b", "#3b0086"]
colors_op = ["#b2dac4", "#b099ce"]
conditions = ["Slow", "Fast"]

# Loop over patients
for i in range(n_datasets):

    # Prepare plotting
    fig = plt.figure(figsize=(5, 2.5))

    # Loop over conditions (In the order they were performed)
    order = [0, 1] if cond[i] == "Slow" else [1, 0]
    for x, j in enumerate(order):

        plt.subplot(1, 2, x+1)

        # Plot feature over time
        plt.plot(feature[i, j, :], color=colors[j], linewidth=1)

        # Add stimulated trials
        stim_idx = np.where(stim[i, j, :])[0]
        for idx in stim_idx:
            plt.plot(idx, feature[i, j, idx], marker="d", color="#990040", ms=3)#, fontsize=8, ha="center", va="center")

        # Adjust subplot
        feature_name_plot = feature_name.replace("_", " ")
        plt.ylabel(f"Raw {feature_name_plot} ", fontsize=13)
        if j == 1: plt.xlabel("Movement number", fontsize=13)
        plt.ylim([1000, 4000])
        plt.axvline(n_trials-n_cutoff)
        u.despine()

    # Adjust plot
    plt.subplots_adjust(bottom=0.15, left=0.1, top=0.85, hspace=0.4, right=0.95)
    plt.suptitle(IDs[i], fontsize=16, y=0.95)

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(f"../../../Figures/{dir_name}/Individual/stimtrace_{IDs[i]}_{feature_name}_{med}.svg", format="svg",
                bbox_inches="tight", transparent=False)
    plt.savefig(f"../../../Figures/{dir_name}/Individual/stimtrace_{IDs[i]}_{feature_name}_{med}.png", format="png",
                bbox_inches="tight", transparent=False)

plt.show()

