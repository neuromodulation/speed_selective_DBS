# Load and look at neurophysiological data
import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report

# Define dataset
ID = "08"
med = "OFF"
file_path = f"..\\..\\Data\\{med}\\raw_data\\{ID}\\neurophys_behavior.vhdr"

# Load dataset
raw = mne.io.read_raw_brainvision(file_path, preload=True)
sfreq = raw.info["sfreq"]

# Create epochs centered on the peak speed


"""# Add bipolar channel
new_chan = np.diff(raw.get_data(["ECOG_R_04_SMC_AT", "ECOG_R_05_SMC_AT"]), axis=0)
# Create new name and info
new_chan_name = "bipolar_4_5"
info = mne.create_info([new_chan_name], raw.info['sfreq'], ["dbs"])
# Add channel to raw object
new_chan_raw = mne.io.RawArray(new_chan, info)
raw.add_channels([new_chan_raw], force_update_info=True)

# Filter the data
raw.notch_filter(130)
# Remove line noise
raw.notch_filter(50)
raw.filter(l_freq=4, h_freq=100)

# Inspect raw data
raw.plot(block=True)

# Inspect power spectral density
#raw.compute_psd(tmin=raw.tmax-130, tmax=raw.tmax-30, fmin=1, fmax=80, n_fft=int(sfreq*1)).plot()
#plt.show()

# Inspect the behavioral data
plt.subplot(3, 1, 1)
spead_mean = raw.get_data(["SPEED_MEAN"])
plt.plot(spead_mean.flatten()[:])
plt.subplot(3, 1, 2)
stim_cond = raw.get_data(["STIM_CONDITION"])
plt.plot(stim_cond.flatten()[:])
plt.subplot(3, 1, 3)
stim = raw.get_data(["STIMULATION"])
plt.plot(stim.flatten()[:]*0.005)
plt.subplot(3, 1, 3)
lfp = raw.get_data(["LFP_L_01_STN_MT"])
plt.plot(lfp.flatten()[:])
plt.close()

# Get onset, offset, peak events
onset_idx = utils.get_onset_idx(raw)
offset_idx = utils.get_offset_idx(raw)
peak_idx = utils.get_peak_idx(raw)
stim_idx = utils.get_stim_idx(raw)

# Extract whether it was slow/fast or fast/slow
slow_first = 1 if stim_cond[0, 0] == 0 else 0

# Get onset of stimulation and recovery blocks
slow_stim = [onset_idx[0], offset_idx[96]]
slow_recovery = [onset_idx[96], offset_idx[96*2]]
fast_stim = [onset_idx[96*2], offset_idx[96*3]]
fast_recovery = [onset_idx[96*3], offset_idx[96*4-1]]

# Add stimulation periods to raw object as annotation

n_stim = len(stim_idx)
onset = (stim_idx / sfreq) - 0.4
duration = np.repeat(1.1, n_stim)
annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim,
                                  orig_time=raw.info['meas_date'])

# Add stimulation periods to raw object as annotation
n_stim = len(onset_idx)
onset = (np.array(onset_idx) / sfreq) - 0.2
duration = np.repeat(0.4, n_stim)
annot = mne.Annotations(onset, duration, ['bad stim'] * n_stim,
                                  orig_time=raw.info['meas_date'])
raw_inspect = raw.copy()
raw_inspect.set_annotations(annot)
print(raw_inspect.annotations)
#raw_inspect.plot(block=True)
#plt.show()

#for i in stim_idx:
#    plt.axvline(i, color="red")

# Compute beta power at start, peak and end of movements
# Use only not stimulated movements

# Check

plt.figure()
lfp = raw.get_data(["bipolar_4_5"])
plt.plot(lfp.flatten()[:])
for i in onset_idx[-190:]:
    plt.axvline(i, color="red")
    plt.axvline(i+100, color="blue")
    plt.axvline(i-100, color="green")
plt.figure()
for i in range(190):
    plt.plot(x[i,:,:].flatten())   
    

# Plot epoch data around movement onset, peak and offset
channel = "bipolar_4_5"
fig, axes1 = plt.subplots(nrows=3, ncols=3)
fig2, axes2 = plt.subplots(nrows=1, ncols=3)
for i, idx in enumerate([onset_idx, peak_idx, offset_idx]):
    n_moves = len(idx)
    events = np.stack((idx, np.zeros(n_moves), np.ones(n_moves))).T
    epochs = mne.Epochs(raw, events=events.astype(int), event_id=1, tmin=-0.2, tmax=0.2)

    epochs.plot_image(picks=[channel], axes=axes1[:, i], show=False)

    # Inspect power
    epochs.plot_psd(fmin=10, fmax=40, average=True)

    frequencies = np.arange(15, 30, 1)
    power = mne.time_frequency.tfr_morlet(epochs, n_cycles=2, return_itc=False,
                                          freqs=frequencies, decim=3)
    power.plot([channel], axes=axes2[i], show=False, baseline=(-0.1, 0))

# Inspect power around onset (without artifacts)
n_moves = len(onset_idx)
events = np.stack((onset_idx, np.zeros(n_moves), np.ones(n_moves))).T
epochs_onset = mne.Epochs(raw, events=events.astype(int), event_id=1, tmin=0, tmax=0.3, baseline=None)
frequencies = np.arange(13, 35, 1)
power = mne.time_frequency.tfr_morlet(epochs_onset, n_cycles=2, return_itc=False,
                                          freqs=frequencies, decim=3, average=False)
# Plot slow vs fast
av_power = np.median(power._data[:, -1, :, ], axis=[1, 2])
power_slow = utils.norm_perc(av_power[96:96*2])
power_fast = utils.norm_perc(av_power[96*3:])
power_slow = av_power[96:96*2]
power_fast = av_power[96*3:]

# Plot
my_pal = {"Slow": "#00863b", "Fast": "#3b0086", "All": "grey"}
my_pal_trans = {"Slow": "#80c39d", "Fast": "#9c80c2", "All": "lightgrey"}
plt.figure()
plt.plot(power_slow.flatten(), color=my_pal["Slow"])
plt.plot(power_fast.flatten(), color=my_pal["Fast"])
plt.show()
"""


