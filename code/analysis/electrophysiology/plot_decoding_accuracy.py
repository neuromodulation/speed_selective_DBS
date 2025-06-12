# Results Figure 4: Plot decoding accuracy achieved using different combinations of channels 

# Prepare environment
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mycolorpy import colorlist as mcp
matplotlib.use('TkAgg')

# Prepare plotting
labels = ["Best ECoG channel", "All ECoG channels", "Best LFP channel", "All LFP channels", "All ECoG & LFP channels"]
fontsize = 12
fig, ax = plt.subplots(figsize=(8, 5))

# Load the excel sheet with the accuracies for every fold and channel combination
ecog_names = ["ECOG_R_1_CAR_12345", "ECOG_R_2_CAR_12345", "ECOG_R_3_CAR_12345", "ECOG_R_4_CAR_12345", "ECOG_R_5_CAR_12345"]
lfp_names = ["LFP_R_2_BIP_234_8", "LFP_R_1_BIP_234_1"]
settings_decoding = ecog_names + ["ECoG_combined"] + lfp_names + ["LFP_combined", "ECoG_LFP_combined"]
n_folds = 8
accuracy_all = np.zeros((len(settings_decoding), n_folds))
for i, setting in enumerate(settings_decoding):
    for j in range(n_folds):

        filename = f"../data/electrophysiology/decoding_results/feature_model_optimization_{setting}_{j}.xlsx"
        df = pd.read_excel(filename, sheet_name=f"Fold {j}")
        try:
            accuracy_all[i, j] = df["all"].iloc[33]
        except:
            accuracy_all[i, j] = 0

# Get the accuracy of the best channel
accuracy_mean = accuracy_all.mean(axis=-1) # Average over outer cross-validation folds
best_ecog = accuracy_mean[:5].argmax()
best_lfp = accuracy_mean[6:8].argmax()
res = np.vstack((accuracy_all[best_ecog, :], accuracy_all[5, :], accuracy_all[6+best_lfp, :], accuracy_all[-2:, :]))
res_mean = res.mean(axis=-1)
res_std = np.std(res, axis=-1)

# Order according to the values
new_order = np.argsort(res_mean)
res = res[new_order, :]
labels = np.array(labels)[new_order]
res_mean = res_mean[new_order]
res_std = res_std[new_order]

# Plot accuracy values as bar plot
colors = mcp.gen_color_normalized(cmap="GnBu",data_arr=res_mean, vmin=0, vmax=0.7)
ax.barh(labels, res_mean, height=0.6, color=colors, xerr=res_std, error_kw=dict(lw=0.6))
for i, res_setting in enumerate(res):
    print(np.round(res_mean[i], 4))
    print(np.round(res_std[i], 3))
    jitter = np.random.uniform(-0.05, 0.05, size=len(res_setting))
    ax.scatter(res_setting, np.repeat(i, len(res_setting)) + jitter, s=0.5, c="dimgray",
                       zorder=2)

# Adjust plot
ax.xaxis.set_tick_params(labelsize=fontsize, length=0)
ax.yaxis.set_tick_params(labelsize=fontsize, rotation=0)
ax.set_xlabel(f"Decoding accuracy [$R^2$]", fontsize=fontsize)
ax.spines[["right", "top"]].set_visible(False)
plt.subplots_adjust(left = 0.3)


# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
plt.savefig(f"../figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../figures/{plot_name}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()