# Compute and plot the correlation between the outcome measure and a feature e.g. peak speed of simulated movements

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from scipy.stats import percentileofscore
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
feature_name = "perc_speed_during_stim"
om_feature_name = "mean_speed"
mode = "diff"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load excel sheet containing the information of the block order
df = pd.read_excel(f'../../../Data/Dataset_list.xlsx', sheet_name=med)
slow_first = df["Block order"].to_numpy()[1:24 + 1]
slow_fast = np.where(slow_first == "Slow")[0]
fast_slow = np.where(slow_first == "Fast")[0]

# Load matrix containing the outcome measure
x = np.load(f"../../../Data/{med}/processed_data/res_{om_feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
#x = np.load(f"../../../Data/{med}/processed_data/res_inst_{om_feature_name}_{method}.npy")
#x = np.vstack((x, x)).T
#x = x[np.delete(np.arange(24), 3), :]

# Compute feature to correlate
# Difference in percentile of peak speed of stimulated movements in the overall distribution between slow and fast
if feature_name == "percentile_peak_speed_stim":
    feature = np.load(f"../../../Data/{med}/processed_data/peak_speed.npy")
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    n_datasets,_, _, n_trials = feature.shape
    np.apply_along_axis(lambda m: u.fill_outliers_nan_peak_speed(m), axis=3, arr=feature)
    feature = feature[:, :, 0, :]
    stim = stim[:, :, 0, :]
    y = np.zeros(len(stim))
    for i in range(stim.shape[0]):
        y[i] = np.nanmean([percentileofscore(feature[i, 0, :], x, nan_policy='omit') for x in feature[i, 0, :][stim[i, 0, :] == 1]]) - \
                             np.nanmean([percentileofscore(feature[i, 1, :], x, nan_policy='omit') for x in feature[i, 1, :][stim[i, 1, :] == 1]])

if feature_name == "percentile_mean_speed_stim":
    feature = np.load(f"../../../Data/{med}/processed_data/mean_speed.npy")
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    n_datasets,_, _, n_trials = feature.shape
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=3, arr=feature)
    feature = feature[:, :, 0, :]
    stim = stim[:, :, 0, :]
    y = np.zeros(len(stim))
    for i in range(stim.shape[0]):
        y[i] = np.nanmean([percentileofscore(feature[i, 0, :], x, nan_policy='omit') for x in feature[i, 0, :][stim[i, 0, :] == 1]]) - \
                             np.nanmean([percentileofscore(feature[i, 1, :], x, nan_policy='omit') for x in feature[i, 1, :][stim[i, 1, :] == 1]])

elif feature_name == "peak_speed_stim":
    feature = np.load(f"../../../Data/{med}/processed_data/peak_speed.npy")
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    n_datasets,_, _, n_trials = feature.shape
    np.apply_along_axis(lambda m: u.fill_outliers_nan_peak_speed(m), axis=3, arr=feature)
    feature = feature[:, :, 0, :]
    stim = stim[:, :, 0, :]
    y = np.zeros(len(stim))
    for i in range(stim.shape[0]):
        y[i] = np.nanmean([x for x in feature[i, 0, :][stim[i, 0, :] == 1]]) - \
                             np.nanmean([x for x in feature[i, 1, :][stim[i, 1, :] == 1]])

# Difference between mean speed during stimulation
elif feature_name == "speed_during_stim":
    feature = np.load(f"../../../Data/{med}/processed_data/speed_during_stim.npy")
    feature_av = np.nanmean(feature, axis=(2, 3, 4))
    #feature_av = np.nanpercentile(feature, 80, axis=(2, 3, 4))
    y = feature_av[:, 0] - feature_av[:, 1]

# Difference between mean dec during stimulation
elif feature_name == "dec_during_stim":
    feature = np.load(f"../../../Data/{med}/processed_data/speed_during_stim.npy")
    feature = np.diff(feature, axis=-1)
    feature_av = np.nanmean(feature, axis=(2, 3, 4))
    y = feature_av[:, 0] - feature_av[:, 1]

# Difference between percentile of mean speed during stimulation
elif feature_name == "perc_speed_during_stim":
    feature_stim = np.load(f"../../../Data/{med}/processed_data/speed_during_stim.npy")
    feature_all = np.load(f"../../../Data/{med}/processed_data/speed_after_peak.npy")
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    stim = stim[:, :, 0, 3:]
    feature_stim = np.nanmean(feature_stim[:, :, 0, 3:, :], axis=-1)
    feature_all = np.nanmean(feature_all[:, :, 0, 3:, :], axis=-1)
    y = np.zeros(len(feature_stim))
    for i in range(feature_stim.shape[0]):
        y[i] = np.nanmean([percentileofscore(feature_all[i, 0, :], x, nan_policy='omit') for x in feature_stim[i, 0, :][stim[i, 0, :] == 1]]) - \
                             np.nanmean([percentileofscore(feature_all[i, 1, :], x, nan_policy='omit') for x in feature_stim[i, 1, :][stim[i, 1, :] == 1]])

# Difference between percentile of mean speed during stimulation
elif feature_name == "perc_dec_during_stim":
    feature_stim = np.load(f"../../../Data/{med}/processed_data/speed_during_stim.npy")
    feature_all = np.load(f"../../../Data/{med}/processed_data/speed_after_peak.npy")
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    stim = stim[:, :, 0, 3:]
    feature_stim = np.nanmean(np.diff(feature_stim[:, :, 0, 3:, :], axis=-1), axis=-1)
    feature_all = np.nanmean(np.diff(feature_all[:, :, 0, 3:, :], axis=-1), axis=-1)
    y = np.zeros(len(feature_stim))
    for i in range(feature_stim.shape[0]):
        y[i] = np.nanmean([percentileofscore(feature_all[i, 0, :], x, nan_policy='omit') for x in feature_stim[i, 0, :][stim[i, 0, :] == 1]]) - \
                             np.nanmean([percentileofscore(feature_all[i, 1, :], x, nan_policy='omit') for x in feature_stim[i, 1, :][stim[i, 1, :] == 1]])

# Number of stimulated movements during slow or fast
elif feature_name == "n_stim":
    stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")
    feature_sum = np.sum(stim[:, :, 0, :], axis=(1, 2))
    y = feature_sum

# Compute correlation and plot for both blocks
plt.figure(figsize=(8, 3.5))
#x = x[fast_slow, :]
#y = y[fast_slow]
#y = y[np.delete(np.arange(24), 3)]
for i, block in enumerate(["Stimulation", "Recovery"]):

    corr, p = pearsonr(x[:, i], y)
    p = np.round(p, 3)
    if p < 0.05:
        label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
    else:
        label = f" R = {np.round(corr, 2)} p = {p}"
    plt.subplot(1, 2, i+1)
    sb.regplot(x=x[:, i], y=y, label=label, scatter_kws={"color": "indianred"}, line_kws={"color": "teal"})

    # Adjust plot
    plt.legend(loc="upper right", fontsize=11)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(f"{method} {om_feature_name} \n difference Slow-Fast", fontsize=12)
    plt.title(block, fontsize=12)
    u.despine()
    plt.ylabel(feature_name, fontsize=13)

# Adjust figure
plt.subplots_adjust(wspace=0.3, top=0.8, bottom=0.2)
#plt.suptitle(f"{med} {cutoff_name} {norm_name} {smooth_name}", fontsize=13, y=0.97)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/corr_{om_feature_name}_{feature_name}.svg",
            format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/corr_{om_feature_name}_{feature_name}.png",
            format="png", bbox_inches="tight", transparent=True)
plt.show()