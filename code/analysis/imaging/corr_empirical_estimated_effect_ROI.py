# Results Figure 3: Correlation empricial and estimated stimulation effect based on functional connectivity to different Regions of Interest (ROI)

# Prepare environment
import os
from scipy.stats import pearsonr
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.stats.multitest as smm
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Prepare plotting 
f, ax = plt.subplots(1, 1, figsize=(5, 5))
fontsize = 12
plot_name = os.path.basename(__file__).split(".")[0]

# Load matrix containing the (empirical) stimulation effect
empirical_stimulation_effect = np.load("../data/imaging/stimulation_effect.npy")

# Load matrix containing average functional connectvity values to 12 ROIs
ROI_functional_connectivity = np.load("../data/imaging/ROI_functional_connectivity.npy")

# Load the array containing the name of the ROIs
ROI_names = np.load("../data/imaging/ROI_names.npy")

# Train a linear regression model for each ROI to estimate the stimulation effect
ROI_corr= np.zeros(len(ROI_names))
ROI_p_value = np.zeros(len(ROI_names))
for i, (functional_connectivity, name) in enumerate(zip(ROI_functional_connectivity, ROI_names)):

    model = LinearRegression()
    model.fit(functional_connectivity.reshape(-1, 1), empirical_stimulation_effect.reshape(-1, 1))
    estimated_stimulation_effect = model.predict(functional_connectivity.reshape(-1, 1))
    corr, p = pearsonr(estimated_stimulation_effect.squeeze(), empirical_stimulation_effect)
    ROI_corr[i] = corr * corr
    ROI_p_value[i] = p

    # Plot the correlation between empirical and estimated stimulation effect
    f, ax_ROI = plt.subplots(1, 1, figsize=(5, 5))
    sb.regplot(x=empirical_stimulation_effect, y=estimated_stimulation_effect, scatter_kws={"color": "grey", 's': 4}, line_kws={"color": 'dimgrey', 'linewidth': 2}, ax=ax_ROI)

    # Adjust plot
    ax_ROI.xaxis.set_tick_params(labelsize=fontsize-2)
    ax_ROI.yaxis.set_tick_params(labelsize=fontsize-2)
    ax_ROI.set_xlabel("Empirical effect", fontsize=fontsize)
    ax_ROI.set_ylabel(f"Estimated effect", fontsize=fontsize)
    ax_ROI.set_title(f" {name} $R^2$ = {np.round(ROI_corr[i], 2)}", fontsize=fontsize)
    ax_ROI.spines[['top', 'right']].set_visible(False)

    # Save figure
    plt.savefig(f"../figures/{plot_name}_{name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
    plt.savefig(f"../figures/{plot_name}_{name}.svg", format="svg", transparent=True, bbox_inches="tight")

# Correct the p values using FDR correction 
significant, ROI_p_value_corrected = smm.fdrcorrection(ROI_p_value, alpha=0.05, method='poscorr')
ROI_p_value_corrected[np.where(~significant)[0]] = 1

# Sort by highest correlation value
idx_sorted = np.argsort(ROI_corr)
ROI_corr_sorted = ROI_corr[idx_sorted]
ROI_p_value_corrected_sorted = ROI_p_value_corrected[idx_sorted]
ROI_names_sorted = ROI_names[idx_sorted]
text = [f"{text} p = {np.round(p, 3)}" for (text, p) in zip(ROI_names_sorted, ROI_p_value_corrected_sorted)]

# Plot correction value as bar plot
ax.barh(text, ROI_corr_sorted, height=0.6, color="#777777")
ax.xaxis.set_tick_params(labelsize=fontsize-2)
ax.yaxis.set_tick_params(labelsize=fontsize-2)
ax.yaxis.set_tick_params(length=0)
ax.set_xlabel(f"$R^2$", fontsize=fontsize)
ax.set_ylabel("Regions of interest (ROI)", fontsize=fontsize, labelpad=1)
ax.spines[['top', 'right']].set_visible(False)
plt.subplots_adjust(left=0.2)

# Save figure
plt.savefig(f"../figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../figures/{plot_name}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()
