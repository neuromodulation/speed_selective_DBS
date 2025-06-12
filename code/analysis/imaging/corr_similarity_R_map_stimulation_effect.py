# Results Figure 3: Correlation "optimal R-map" with patient-individual functional connectivity-map

# Prepare environment
import os
import seaborn as sb
from scipy.stats import pearsonr
import numpy as np
import scipy.stats
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Prepare plotting 
f, ax = plt.subplots(1, 1, figsize=(5, 5))
fontsize = 12

# Load matrix containing the stimulation effect
stimulation_effect = np.load("../data/imaging/stimulation_effect.npy")


# Compute the spatial correlation (similarity) for each subject and each R map (computed without the subject)
similarity = np.zeros(len(stimulation_effect))
subjects = np.concatenate((np.arange(1, 4), np.arange(5, 25)))
for i, sub in enumerate(subjects):

    # Load the R map computed without the subject
    r_map = np.load(f"../data/imaging/r_maps/{sub}.npy")

    # Load the fcuntional connectivity map of the patient
    functional_connectivity = nib.load(f"../data/imaging/functional_connectivity_PPMI_patients/{sub}.nii").get_fdata()

    # Compute the spatial correlation between both maps
    valid_indices = ~np.isnan(r_map.flatten()) & ~np.isnan(functional_connectivity.flatten())
    similarity[i], _ = pearsonr(r_map.flatten()[valid_indices], functional_connectivity.flatten()[valid_indices])

# Compute and plot the correlation between the similarity to the r-map and the stimulation effect
corr, p = scipy.stats.pearsonr(stimulation_effect, similarity)
corr = corr * corr
p = np.round(p, 3)
sb.regplot(x=stimulation_effect, y=similarity, scatter_kws={"color": "grey", 's': 4}, line_kws={"color": 'dimgrey', 'linewidth': 2}, ax=ax)

# Adjust plot
ax.xaxis.set_tick_params(labelsize=fontsize-2)
ax.yaxis.set_tick_params(labelsize=fontsize-2)
ax.set_xlabel("Stimulation effect [%]", fontsize=fontsize)
ax.set_ylabel(f"Similarity to R-map", fontsize=fontsize)
ax.set_title(f" $R^2$ = {np.round(corr, 2)} p = {p}", fontsize=fontsize)
ax.spines[['top', 'right']].set_visible(False)
plt.subplots_adjust(left = 0.2)

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
plt.savefig(f"../figures/{plot_name}.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../figures/{plot_name}.svg", format="svg", transparent=True, bbox_inches="tight")

plt.show()
