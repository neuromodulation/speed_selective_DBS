# Inspect predicition accuracy using connectivity to different regions

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
import scipy
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib
from mycolorpy import colorlist as mcp
from matplotlib import cm
matplotlib.use('TkAgg')

# Load the excel sheet with the performance metrics
ecog_names = ["ECOG_R_1_CAR_12345", "ECOG_R_2_CAR_12345", "ECOG_R_3_CAR_12345", "ECOG_R_4_CAR_12345", "ECOG_R_5_CAR_12345"]
lfp_names = ["LFP_R_2_BIP_234_8", "LFP_R_1_BIP_234_1"]
settings_decoding = ecog_names + ["ECoG_combined"] + lfp_names + ["LFP_combined", "ECoG_LFP_combined"]
labels = ["Best ECoG channel", "All ECoG channels", "Best LFP channel", "All LFP channels", "All ECoG & LFP channels"]
n_folds = 8
res_all = np.zeros((len(settings_decoding), n_folds))

for i, setting in enumerate(settings_decoding):
    for j in range(n_folds):

        filename = f"results_4/feature_model_optimization_{setting}_{j}.xlsx"
        df = pd.read_excel(filename, sheet_name=f"Fold {j}")
        try:
            res_all[i, j] = df["all"].iloc[33]
        except:
            res_all[i, j] = 0

# Get the performance of the best channel
res_mean = res_all.mean(axis=-1)
keep_ecog = res_mean[:5].argmax()
keep_lfp = res_mean[6:8].argmax()
res = np.vstack((res_all[keep_ecog, :], res_all[5, :], res_all[6+keep_lfp, :], res_all[-2:, :]))
res_mean = res.mean(axis=-1)
res_std = np.std(res, axis=-1)

# Order according to the values
new_order = np.argsort(res_mean)
res = res[new_order, :]
labels = np.array(labels)[new_order]
res_mean = res_mean[new_order]
res_std = res_std[new_order]

# Plot as bar plot
fontsize = 6
fig, ax = plt.subplots(figsize=(1.5, 1.3))
colors = mcp.gen_color_normalized(cmap="GnBu",data_arr=res_mean, vmin=0, vmax=0.7)
ax.barh(labels, res_mean, height=0.6, color=colors, xerr=res_std, error_kw=dict(lw=0.6))
for i, res_setting in enumerate(res):
    print(np.round(res_mean[i], 4))
    print(np.round(res_std[i], 3))
    jitter = np.random.uniform(-0.05, 0.05, size=len(res_setting))
    ax.scatter(res_setting, np.repeat(i, len(res_setting)) + jitter, s=0.5, c="dimgray",
                       zorder=2)

# Add statistics
"""comparisons = [[0, 2], [1, 2], [2, 3], [2, 4]]
xmax = 0.6
caps_length = 0.02
for comp in comparisons:
    z, p = scipy.stats.ttest_ind(res[comp[0], :], res[comp[1], :])
    text = u.get_sig_text(p)
    ax.plot([xmax - caps_length, xmax, xmax, xmax - caps_length], [comp[0], comp[0], comp[1], comp[1]], color="black",
            linewidth=0.5)
    ax.text(xmax + 0.005, np.array(comp).mean(), text, fontsize=fontsize-2, verticalalignment='center', rotation=270)
    xmax += 0.08"""

plt.xticks(fontsize=fontsize)
ax.xaxis.set_tick_params(length=0)
plt.yticks(fontsize=fontsize, rotation=0)
plt.xlabel(f"Decoding accuracy [$R^2$]", fontsize=fontsize)
#plt.subplots_adjust(left=0.2)
u.despine()


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