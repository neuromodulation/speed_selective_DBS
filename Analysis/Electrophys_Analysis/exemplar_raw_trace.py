# Plot the raw data during one trial (exemplar trace)

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
import sys
import random
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings("ignore")
random.seed(420)

# Load the data
path = f"EL012_ECoG_CAR_LFP_BIP.fif"
raw = mne.io.read_raw_fif(path).load_data()
raw.filter(2, None)

sfreq = raw.info["sfreq"]
target_chan_name = raw.info["ch_names"][-1]
events = mne.events_from_annotations(raw)[0]

# Get the raw data for the new channels
channel_names = raw.info["ch_names"][-7:]
data = raw.get_data(channel_names)
speed = raw.get_data(["SPEED_MEAN"]).flatten()
speed = np.convolve(speed, np.ones(100)/100, mode='same')

# Get the peak speed index
peaks_idx = events[np.where((events[:, 2] == 3)), 0].flatten()

# Plot for every movement (centering on peak speed +-1 sec)
fontsize=6
tmin = int(0.5 * sfreq)
tmax = int(0.6 * sfreq)
names = ["ECoG 1", "ECoG 2", "ECoG 3", "ECoG 4", "ECoG 5", "LFP 1", "LFP 2"]
for idx in peaks_idx[11:]:

    data_tmp = data[:, idx-tmin:idx+tmax]
    speed_tmp = speed[idx-tmin:idx+tmax]

    fig, axes = plt.subplots(len(data)+1, 1, figsize=(1.5, 1.5), constrained_layout=True)
    axes[0].plot(speed_tmp, color="#1182C6")
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[0].xaxis.set_tick_params(labelsize=fontsize-2)
    axes[0].set_ylabel("Speed", rotation=0, fontsize=fontsize, labelpad=22)
    for i, ax in enumerate(axes[1:]):
        ax.plot(data_tmp[i, :].flatten(), linewidth=0.5, color="#404040")
        ax.spines[["right", "top", "bottom"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(names[i], rotation=0, fontsize=fontsize, labelpad=22)
    plt.subplots_adjust(left=0.1, hspace=0)

    # Save figure
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(
        f"../../../Figures/{dir_name}/{plot_name}_{idx}.pdf",
        format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(
        f"../../../Figures/{dir_name}/{plot_name}_{idx}.png",
        format="png", bbox_inches="tight", transparent=False)
    plt.show()

