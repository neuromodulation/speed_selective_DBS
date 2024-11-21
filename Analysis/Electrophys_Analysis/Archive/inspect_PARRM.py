# Inspect the data, ensure that modulation is not due to artifact
# Plot raw data

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
import sys
import random
sys.path.insert(1, "../Code")
import utils as u
matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings("ignore")
random.seed(420)

# Load the data
sub = "EL012"
path = f"..\\..\\..\\Data\\Off\\Neurophys\\Artifact_removal\\EL012_cleaned_123_CAR.fif"
raw = mne.io.read_raw_fif(path).load_data()
# Add re-references raw channels
ecog_names = ["ECOG_R_1_CAR_raw", "ECOG_R_2_CAR_raw", "ECOG_R_3_CAR_raw"]
og_chan_names = ["ECOG_R_01_SMC_AT", "ECOG_R_02_SMC_AT", "ECOG_R_03_SMC_AT"]
for i, chan in enumerate(og_chan_names):
    new_ch = raw.get_data(chan) - raw.get_data(og_chan_names).mean(axis=0)
    u.add_new_channel(raw, new_ch, ecog_names[i], type="ecog")
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_name ='ECOG_R_1_CAR_raw'
raw.filter(None, 100)

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1, baseline=None, reject_by_annotation=True, picks=[target_chan_name, "STIMULATION"])
epochs.load_data()

# Get fast and slow stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
stim_idx = np.where(stim)[0]
epochs = epochs[stim_idx].pick(target_chan_name)

# Plot for every movement (centering on peak speed +-1 sec)
tmin = 0.15
tmax = 0.25
epochs.crop(tmin=tmin, tmax=tmax)
fig, ax = plt.subplots(nrows=len(epochs), ncols=1, figsize=(6.5, 20.8))
for i, epoch in enumerate(epochs):
    # Get data
    data = epoch.flatten()
    ax[i].plot(data)
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    ax[i].spines[["right", "top", "bottom"]].set_visible(False)
    ax[i].axvline((0.06-tmin)*sfreq, color="red", linewidth=0.5)
    ax[i].axvline((0.36-tmin)*sfreq, color="red", linewidth=0.5)

#plt.subplots_adjust(left=0.1, hspace=0)
plt.show()

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
    f"../../../Figures/{dir_name}/{plot_name}.pdf",
    format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(
    f"../../../Figures/{dir_name}/{plot_name}.png",
    format="png", bbox_inches="tight", transparent=False)
plt.show()

