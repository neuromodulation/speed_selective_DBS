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
matplotlib.use('TkAgg')

# Load the excel sheet with the performance metrics
ecog_names = ["ECOG_R_1_CAR", "ECOG_R_2_CAR", "ECOG_R_3_CAR"]
lfp_names = ["LFP_1", "LFP_2"]
settings_decoding = ecog_names + lfp_names + ["ECoG_combined", "LFP_combined", "ECoG_LFP_combined"]
labels = ["Best ECoG", "Best LFP", "All ECoG", "All LFP", "All ECoG & LFP"]
n_folds = 8
res_all = np.zeros((len(settings_decoding), n_folds))

for i, setting in enumerate(settings_decoding):

    filename = f"decoding_results/feature_model_optimization_{setting}.xlsx"

    for j in range(n_folds):
        df = pd.read_excel(filename, sheet_name=f"Fold {j}")

        res_all[i, j] = df["all"].iloc[-1]

# Get the performance of the best channel
res_mean = res_all.mean(axis=-1)
keep_ecog = res_mean[:3].argmax()
keep_lfp = res_mean[3:5].argmax()
res = np.vstack((res_all[keep_ecog, :], res_all[3+keep_lfp, :], res_all[-3:, :]))
res_mean = res.mean(axis=-1)
res_std = np.std(res, axis=-1)

# Plot as bar plot
fontsize = 8
fig, ax = plt.subplots(figsize=(1.8, 1.6))
ax.barh(labels, res_mean, height=0.5, color="#D24A4A", xerr=res_std, error_kw=dict(lw=0.6))
for i, res_setting in enumerate(res):
    jitter = np.random.uniform(-0.05, 0.05, size=len(res_setting))
    ax.scatter(res_setting, np.repeat(i, len(res_setting)) + jitter, s=0.5, c="dimgray",
                       zorder=2)

# Add statistics
comparisons = [[0, 2], [1, 2], [2, 3], [2, 4]]
xmax = 1
caps_length = 0.02
for comp in comparisons:
    z, p = scipy.stats.ttest_ind(res[comp[0], :], res[comp[1], :])
    text = u.get_sig_text(p)
    ax.plot([xmax - caps_length, xmax, xmax, xmax - caps_length], [comp[0], comp[0], comp[1], comp[1]], color="black",
            linewidth=0.5)
    print(text)
    ax.text(xmax + 0.01, np.array(comp).mean(), text, fontsize=fontsize-2, verticalalignment='center', rotation=270)
    xmax += 0.1

plt.xticks(fontsize=fontsize)
ax.xaxis.set_tick_params(length=0)
plt.yticks(fontsize=fontsize, rotation=0)
plt.xlabel(f"$R^2$", fontsize=fontsize+2)
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