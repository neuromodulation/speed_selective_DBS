# Plot the decoded trace and a correlation between (true and predicted peak speed)

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
import random
import pickle
from scipy.stats import pearsonr
import seaborn as sb
sys.path.insert(1, "../../../Code")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')
random.seed(420)

# Load the model
setting = "LFP_combined"
model_path = f"optimization_{setting}_vis/optimization_{setting}_vis_LM_ML_RES.p"
file = open(model_path,'rb')
decoder = pickle.load(file)

# Plot the predicted speed
channel = "all_ch_combined"
for i in range(0, 7):
    y_test = decoder.all_ch_results["y_test"][i]
    y_test_pr = decoder.all_ch_results["y_test_pr"][i]

    # Smooth the prediction
    #y_test_pr = np.convolve(y_test_pr, np.ones(3) / 3, mode='same')

    fig, ax = plt.subplots()  # figsize=(1.3, 1.3))
    if "ECoG_combined":
        ax.plot(np.array(y_test)[2060:2240], color="black", label="speed")
        ax.plot(y_test_pr[2060:2240], color="red", label="predicted speed")
    elif "LFP_combined":
        ax.plot(np.array(y_test)[990:1170], color="black", label="speed")
        ax.plot(y_test_pr[990:1170], color="red", label="predicted speed")
    ax.legend()

    # Save
    plot_name = os.path.basename(__file__).split(".")[0]
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(
            f"../../../Figures/{dir_name}/{plot_name}_{setting}.pdf",
            format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(
        f"../../../Figures/{dir_name}/{plot_name}_{setting}.png",
        format="png", bbox_inches="tight", transparent=False)

plt.show()


# Plot the predicted speed
channel = "all_ch_combined"
for i in range(4):

    y_test = decoder.all_ch_results["y_test"][i]
    y_test_pr = decoder.all_ch_results["y_test_pr"][i]

    # Smooth the prediction
    y_test_pr = np.convolve(y_test_pr, np.ones(6)/6, mode='same')

    fig, ax = plt.subplots()#figsize=(1.3, 1.3))
    ax.plot(np.array(y_test)[200:400], color="black", label="speed")
    ax.plot(y_test_pr[200:400], color="red", label="predicted speed")
    ax.legend()
    #plt.close()
plt.show()

# Plot the correlation between true and predicted speed
speed_true = []
speed_decoded = []
peaks = np.array(decoder.features.PEAKS)
speed = np.array(decoder.features.SPEED_MEAN)
for i in range(4):
    y_test = decoder.all_ch_results["y_test"][i]
    y_test_pr = decoder.all_ch_results["y_test_pr"][i]
    idx_all = np.where(peaks[y_test.index[0]:y_test.index[-1]] == 1)[0]
    idx_all = idx_all[idx_all >= 10]

    speed_true.extend(np.array(y_test)[idx_all])
    speed_decoded_tmp = np.array([np.max(np.array(y_test_pr)[idx-10:idx+10]) for idx in idx_all])
    speed_decoded.extend(np.array(y_test_pr)[idx_all])
    #speed_decoded.extend(speed_decoded_tmp)

    """plt.plot(np.array(y_test))
    plt.plot(np.array(y_test_pr))
    for idx in idx_all:
        plt.axvline(idx, color="r")
    plt.show(block=True)"""

# Calculate whether movement would be fast or not
fast = np.zeros((len(speed_decoded)-2, 2))
for i in range(2, len(speed_decoded)):
    if np.all(speed_true[i] < speed_true[i-2:i]):
        fast[i-2, 0] = 1
    if np.all(speed_decoded[i] < speed_decoded[i - 2:i]):
        fast[i-2, 1] = 1

fast_1 = fast[:, 0]
fast_2 = fast[:, 1]
np.random.shuffle(fast_2)
#fast = np.stack((fast_1, fast_2), axis=1)

perf = np.sum(fast[:, 0] == fast[:, 1]) / len(fast)
sensitivity = np.sum((fast[:, 0] == 1) & (fast[:, 1] == 1)) / np.sum(fast[:, 0] == 1)
specificity = np.sum((fast[:, 0] == 0) & (fast[:, 1] == 0)) / np.sum(fast[:, 0] == 0)
print(f"Performance = {perf} Sensitivity = {sensitivity} Specificity = {specificity}")
n_fast_decoded = np.sum(fast[:, 1])/len(fast)
n_fast_true = np.sum(fast[:, 0])/len(fast)
print(f"n_fast_decoded = {n_fast_decoded} n_fast_true = {n_fast_true}")

# Plot
plt.figure(figsize=(1, 1))
corr_res, p = pearsonr(speed_true, speed_decoded)
p = np.round(p, 3)
sb.regplot(x=speed_true, y=speed_decoded,
     scatter_kws={"color": "grey", 's': 0.5}, line_kws={"color": "red", 'linewidth':1})
# Adjust plot
fontsize=8
plt.xticks(fontsize=fontsize-2)
plt.yticks(fontsize=fontsize-2)
plt.xlabel("True peak speed", fontsize=fontsize, labelpad=0.5)
plt.ylabel("Decoded peak speed", fontsize=fontsize, labelpad=0.5)
plt.title(f"R = {np.round(corr_res, 2)} p = {p}", fontsize=fontsize-1, pad=0.5)
u.despine()

# Save figure
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
f"../../../Figures/{dir_name}/corr_peak_speed.pdf",
format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(
f"../../../Figures/{dir_name}/corr_peak_speed.png",
format="png", bbox_inches="tight", transparent=False)

plt.show()

