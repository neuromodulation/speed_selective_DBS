# Validate the R-map by computing the network score

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import mat73
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import fdrcorrection
from sklearn.linear_model import LinearRegression
import scipy.stats
from scipy.io import savemat, loadmat
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
from openpyxl import load_workbook
mpl.use('TkAgg')
# Export text as regular text instead of paths or using svgfonts
mpl.rcParams['svg.fonttype'] = 'none'

# Set font to a widely-available but non-default font to show that Affinity
# ignores font-family
mpl.rcParams['font.family'] = 'Arial'

# Set parameters
med = "Off"
mask = "none"
Fz = True
corr_method = "spearman"
feature_name = "mean_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load matrix containing the outcome measure
outcome = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")

# Delete subject for whom the electrode type is different
outcome = outcome[np.delete(np.arange(24), 3), :]

# Load the mask
if mask != "none":
       mask_image = nib.load(f"C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\atlas\\{mask}.nii").get_fdata()

# Build up a matrix containing the spatial correlation coefficients for each subject and each R map
subjects = np.concatenate((np.arange(2, 5), np.arange(6, 26)))
res = np.zeros((len(subjects), 2))
for i, sub in enumerate(subjects):

       print(sub)

       # Load the R map computed without the subject
       X = loadmat(f"../../../Data/{med}/processed_data/vw_corr_{sub}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.mat")
       corr = X["corr"]
       p = X["p"]

       # Load the connectivity map of the patient
       subject_path = f"C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\LeadDBSDataset\\derivatives\\leaddbs\\sub-{sub}\\stimulations\\MNI152NLin2009bAsym\\gs_test\\"
       # Load nifti file
       if Fz:
              file_name = f"sub-{sub}_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmap.nii"
       else:
              file_name = f"sub-{sub}_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgR_funcmap.nii"
       image = nib.load(os.path.join(subject_path, file_name)).get_fdata()

       # Compute the spatial correlation between both maps
       if mask == "none":
              valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 0].flatten())
              res_stim, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 0].flatten()[valid_indices])
              res[i, 0] = res_stim
              valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 1].flatten())
              res_recov, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 1].flatten()[valid_indices])
              res[i, 1] = res_recov
       else:
              valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 0].flatten()) & (mask_image.flatten() > 0)
              res_stim, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 0].flatten()[valid_indices])
              res[i, 0] = res_stim
              valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 1].flatten()) & (mask_image.flatten() > 0)
              res_recov, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 1].flatten()[valid_indices])
              res[i, 1] = res_recov

# Plot the network score and the outcome measure
block_names = ["Stim", "Recovery"]
fig = plt.figure(figsize=(1.2, 1.2))
for i in range(1):
       plt.subplot(1, 1, i+1)
       x = outcome[:, 0]
       y = res[:, 0]
       corr_res, p = scipy.stats.pearsonr(x, y)#, nan_policy="omit")
       corr_res = corr_res * corr_res
       #corr_res, p = u.permutation_correlation(x, y, n_perm=100000)
       p = np.round(p, 3)
       sb.regplot(x=x, y=y, scatter_kws={"color": "grey", 's': 0.5}, line_kws={"color": 'dimgrey', 'linewidth':1})
       # Adjust plot
       fontsize=8
       plt.xticks(fontsize=fontsize-2)
       plt.yticks(fontsize=fontsize-2)
       #plt.xlabel(f"{method} {feature_name} {mode}", fontsize=12)
       plt.xlabel("Stimulation effect [%]", fontsize=fontsize)
       plt.ylabel(f"Similarity to R-map", fontsize=fontsize)
       plt.title(f" R = {np.round(corr_res, 2)} p = {p}", fontsize=fontsize)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
u.despine(["top", "right"])

# Save
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
       f"../../../Figures/{dir_name}/corr_ns_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{mask}.pdf",
       format="pdf", bbox_inches="tight", transparent=False)
plt.savefig(
       f"../../../Figures/{dir_name}/corr_ns_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{mask}.png",
       format="png", bbox_inches="tight", transparent=False)
plt.show()