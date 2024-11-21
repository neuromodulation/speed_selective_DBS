# Get the time point of the artifact onset for each stimulated epoch
# Save time points as array

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import zscore
import matplotlib
matplotlib.use('Qt5Agg')

# Load the data
path = f"EL012_ECoG_CAR_LFP_BIP.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
target_chan_name = "ECOG_R_1_CAR_12345"

# Crop dataset
raw.crop(tmin=0, tmax=raw.times[-1]-2)

# Get epochs aligned to peak speed
events = mne.events_from_annotations(raw)[0]
tmin = -0.2
tmax = 0.5
epochs = mne.Epochs(raw, events=events, event_id=3, tmin=tmin, tmax=tmax, baseline=None)
epochs.load_data()

# Get index of stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
stim_idx = np.where(stim)[0]
epochs = epochs[stim_idx].pick(target_chan_name)

# Get time point in aligned epoch at which stimulation starts and stops
idx_stim = np.zeros((len(epochs), 2))
for i, (data_epoch, event) in enumerate(zip(epochs.get_data(), epochs.events)):

    data_epoch = data_epoch.flatten()
    # First sample of the epoch that is above 3 standard deviations (from both sides)
    idx_stim_on = np.where(np.abs(zscore(data_epoch)) > 3)[0][0] - 10
    idx_stim_off = len(data_epoch) - np.where(np.abs(zscore(np.flip(data_epoch))) > 3)[0][0] + 15

    # Plot for visual inspection
    """plt.figure()
    plt.plot(data_epoch)
    plt.axvline(idx_stim_on, color="red")
    plt.axvline(idx_stim_off, color="red")
    plt.show()"""

    # Save
    idx_stim[i] = [(tmin*sfreq) + idx_stim_on, (tmin*sfreq) + idx_stim_off]

# Save array
idx_stim = np.array(idx_stim) / sfreq
np.save("artifact_sec_epochs.npy", idx_stim)
