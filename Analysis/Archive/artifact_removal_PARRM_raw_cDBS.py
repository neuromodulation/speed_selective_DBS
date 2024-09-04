# Remove the stimulation artifact using PARRM of the constant stimulation rest recording

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import zscore
from scipy import signal
from pyparrm import PARRM
from mne_bids import read_raw_bids, find_matching_paths
import matplotlib
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
matplotlib.use('Qt5Agg')

# Load the data
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks="Rest",
                                    extensions=".vhdr",
                                    subjects="EL012",
                                    acquisitions="StimOnB",
                                    sessions=[f"LfpMedOff01", f"EcogLfpMedOff01",
                                              f"LfpMedOff02", f"EcogLfpMedOff02", f"LfpMedOffDys01"])
raw = read_raw_bids(bids_path=bids_path[0])
sfreq = raw.info["sfreq"]

# Apply bipolar re-reference for ecog channel
raw.load_data()
raw.crop(tmax=raw.times[-1] - 0.5)
new_ch = raw.get_data("ECOG_R_02_SMC_AT") - raw.get_data("ECOG_R_03_SMC_AT")
target_chan_name = "ECOG_bipolar"
u.add_new_channel(raw, new_ch, target_chan_name, type="ecog")
ch_names = raw.info["ch_names"]

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 7))
data_artefact = raw.get_data(target_chan_name)
axes[0].plot(data_artefact.flatten())
axes[0].set_title("Raw")

# Apply PARRM to remove the artefact
parrm = PARRM(data=data_artefact, sampling_freq=sfreq, artefact_freq=130, verbose=False)
parrm.find_period()
#parrm.explore_filter_params()
parrm.create_filter(period_half_width=0.02, filter_half_width=8000)
data_artefact_clean = parrm.filter_data().flatten()
axes[1].plot(data_artefact_clean)
axes[1].set_title("PARRM cleaned")
raw_stim = raw.copy()
raw_stim._data[ch_names.index(target_chan_name), :] = data_artefact_clean

# Add data without stimulation
root = f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
       f'Interventional Cognitive Neuromodulation - BIDS_01_Berlin_Neurophys/' \
       f'rawdata/'
bids_path = find_matching_paths(root, tasks="Rest",
                                    extensions=".vhdr",
                                    subjects="EL012",
                                    acquisitions="StimOff",
                                    sessions=[f"LfpMedOff01", f"EcogLfpMedOff01",
                                              f"LfpMedOff02", f"EcogLfpMedOff02", f"LfpMedOffDys01"])
raw_no_stim = read_raw_bids(bids_path=bids_path[0])
sfreq = raw_no_stim.info["sfreq"]
raw_no_stim.load_data()
raw_no_stim.crop(tmax=raw_no_stim.times[-1] - 0.5)
new_ch = raw_no_stim.get_data("ECOG_R_02_SMC_AT") - raw_no_stim.get_data("ECOG_R_03_SMC_AT")
target_chan_name = "ECOG_bipolar"
u.add_new_channel(raw_no_stim, new_ch, target_chan_name, type="ecog")


# Plot the psd for cleaned/stimulated and not stimulated data
# Use epochs of 1 second length
colors = ["b", "r"]
labels = ["Stimulated", "Not Stimulated"]
for r, raw in enumerate([raw_stim, raw_no_stim]):

    # Cut into epochs of 1 second length
    events = []
    n_events = np.floor(np.max(raw.times))
    for i in range(int(n_events)):
        events.append([int(i*sfreq), 0, 1])

    events = np.array(events)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1, baseline=None, preload=True)
    # Plot the PSD
    psd = epochs.compute_psd(picks=[target_chan_name], fmin=0.1, fmax=200).average()
    data = np.log10(psd.data.flatten())
    axes[2].plot(data, color=colors[r], label=labels[r])
axes[2].legend()

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.suptitle(target_chan_name)

plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}.svg",
            format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{target_chan_name}.png",
            format="png", bbox_inches="tight", transparent=False)
plt.show(block=True)