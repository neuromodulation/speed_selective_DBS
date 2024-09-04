# Inspect predicition accuracy using connectivity to different regions

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Load the excel sheet with the performance metrics
ecog_names = ["ECOG_R_1_CAR", "ECOG_R_2_CAR", "ECOG_R_3_CAR"]
lfp_name = ["LFP_1, LFP_2"]
labels = ["Best ECoG", "LFP", "All ECoG & LFP", "All ECoG"]
settings = ["ECOG_individual", "LFP", "ECoG_LFP_combined", "ECoG_combined"]
perf_file = f'feature_model_optimization_all_old.xlsx'

res = np.zeros(len(labels))
for i, setting in enumerate(settings):

    if setting == "ECOG_individual":
        res_tmp = np.zeros(len(ecog_names))
        for j in range(len(ecog_names)):
            df = pd.read_excel(perf_file, sheet_name=ecog_names[j], header=None)
            # Get the highest decoding performance
            res_tmp[j] = df[max(df.columns)].max()
        res[i] = res_tmp.max()
    else:
        df = pd.read_excel(perf_file, sheet_name=setting, header=None)
        res[i] = df[max(df.columns)].max()

# Plot as bar plot
colors = []
cmap = get_cmap('Reds')
denominator = (max(res) - min(res))*1.2
scaled_data = [(datum-min(res))/denominator for datum in res]
for decimal in scaled_data:
    colors.append(cmap(decimal))
fig, ax = plt.subplots(figsize=(1.3, 1.3))
ax.barh(labels, res, height=0.5, color="#AC9090")
plt.xticks(fontsize=8)
ax.xaxis.set_tick_params(length=0)
plt.yticks(fontsize=8, rotation=0)
plt.xlabel(f"$R^2$", fontsize=10)
#plt.ylabel("Used channels", fontsize=9, labelpad=2)
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