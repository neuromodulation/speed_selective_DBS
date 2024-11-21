# Compute cortico-subcortical coherence modulated by STN-DBS

import os
import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import scipy
import sys
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from scipy.stats import pearsonr, spearmanr, ttest_ind
sys.path.insert(1, "../../../Code")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')


# Set parameters
method = "coh"
indices = (np.array([1]), np.array([0]))
#indices = (np.array([[0]]), np.array([[1]]))
fmin = 60
fmax = 80
frequencies = np.arange(fmin, fmax, 1)
tmin = 0.5
tmax = 1
fontsize = 10

# Load the data
path = f"EL012_ECoG_CAR_LFP_BIP.fif"
raw = mne.io.read_raw_fif(path).load_data()
sfreq = raw.info["sfreq"]
ch_names = raw.info["ch_names"]
target_chan_names = ["ECOG_R_1_CAR_12345", "LFP_R_2_BIP_234_8"]

# Load index of similar movements
id = 6
similar_slow = np.load(f"../../../Data/Off/processed_data/Slow_similar.npy").astype(bool)
similar_fast = np.load(f"../../../Data/Off/processed_data/Fast_similar.npy").astype(bool)
slow_idx = np.where(np.hstack((similar_slow[id, 1, :, :].flatten(), similar_slow[6, 0, :, :].flatten())))[0]
fast_idx = np.where(np.hstack((similar_fast[id, 1, :, :].flatten(), similar_fast[6, 0, :, :].flatten())))[0]
no_stim_idx = np.hstack((fast_idx, slow_idx))

# Extract events
events = mne.events_from_annotations(raw)[0]

# Cut into epochs
event_id = 3
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=1, baseline=None, reject_by_annotation=True)

# Get stimulated movements
stim = np.any(epochs.get_data(["STIMULATION"])[:,:, ((epochs.times < 0.5) & (epochs.times > -0.2))], axis=-1).squeeze()
stim_idx = np.where(stim)[0]

# Get the array of peak speeds
peak_speed = epochs.get_data(["SPEED_MEAN"])[:, :, epochs.times == 0].squeeze()

# Compute coherence for stimulated and not stimulated movements
plt.figure(figsize=(3, 4))
labels = ["Stim", "No Stim"]
colors = ["red", "green"]
conns = []
for i, idx in enumerate([stim_idx, no_stim_idx]):
    data = epochs[idx].get_data(picks=target_chan_names, tmin=tmin, tmax=tmax).squeeze()
    """conn = spectral_connectivity_epochs(data=data, sfreq=sfreq, mode='cwt_morlet', fmin=fmin, fmax=fmax,
                                        faverage=True, cwt_freqs=frequencies, cwt_n_cycles=4,
                                        method=method, indices=indices, gc_n_lags=27)"""
    conn = spectral_connectivity_time(data=data, sfreq=sfreq, freqs=frequencies, method=method, faverage=True, n_cycles=4,
                                      indices=indices, padding=0.05, gc_n_lags=27)
    conns.append(conn.get_data().flatten())

    # Plot
    plt.boxplot(conn.get_data().flatten(), positions=[i], label=labels[i], showfliers=False,
                boxprops=dict(color=colors[i]),
                capprops=dict(color=colors[i]),
                whiskerprops=dict(color=colors[i]),
                medianprops=dict(color=colors[i], linewidth=1)
                )
plt.ylabel(f"Connectivity {method}")
plt.xticks([])
plt.legend()
plt.subplots_adjust(left=0.2)

# Statistics
z, p = scipy.stats.ttest_ind(conns[0], conns[1])
print(p)
res = scipy.stats.permutation_test(data=(conns[0], conns[1]),
                                   statistic=u.diff_mean_statistic,
                                   n_resamples=100000, permutation_type="independent")
p = res.pvalue
text = u.get_sig_text(p)
plt.title(f"p = {text}")
print(p)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{method}.pdf", format="pdf", bbox_inches="tight",
            transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_{method}.png", format="png", bbox_inches="tight",
            transparent=False)

plt.show()