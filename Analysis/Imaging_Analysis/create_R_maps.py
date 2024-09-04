# Create R-maps based on the computed correlation values


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
import scipy.stats
from statsmodels.stats.multitest import fdrcorrection, multipletests
from scipy.io import savemat, loadmat
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
Fz = True
corr_method = "spearman"
feature_name = "mean_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5

# Load the correlation values
X = loadmat(f"../../../Data/{med}/processed_data/vw_corr_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.mat")
corr = X["corr"]
p = X["p"]

# Load nifti file to use as mask
path = f"C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\LeadDBSDataset\\derivatives\\leaddbs\\" \
       f"sub-2\\stimulations\\MNI152NLin2009bAsym\\gs_test\\sub-2_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmap.nii"
nii = nib.load(path)

# Filter by p value and save stimulation and recovery as nifti files
alpha = 0.05
block_names = ["stim", "recov"]
plt.figure()
for i in range(2):
       p_tmp = p[:, :, :, i]
       corr_tmp = corr[:, :, :, i]

       # False Positive Correction
       idx_p_not_nan = ~np.isnan(p_tmp)
       p_not_nan = p_tmp[idx_p_not_nan]
       rejected, p_corrected = fdrcorrection(p_not_nan.flatten(), alpha=alpha, method='i')

       """rejected, p_corrected, _, _ = multipletests(p_not_nan.flatten(), alpha=0.1, method='fdr_tsbky', is_sorted=False,
                                                 returnsorted=False)"""
       p_tmp[idx_p_not_nan] = p_corrected

       # Plot
       plt.subplot(1, 2, i + 1)
       plt.hist(p_not_nan.flatten(), bins=100, alpha=0.5, color="grey")
       plt.hist(p_corrected, bins=100, alpha=0.5, color="red")
       plt.xlim([0, 0.1])

       #corr_tmp[p_tmp > alpha] = 0
       nii_new = nib.Nifti1Image(corr_tmp, nii.affine, nii.header)
       nib.save(nii_new, f'results\\{block_names[i]}_R_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}')

plt.title(f"{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}")

# Save
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/p_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.svg",
            format="svg", bbox_inches="tight", transparent=False)
plt.savefig(f"../../../Figures/{dir_name}/p_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.png",
            format="png", bbox_inches="tight", transparent=False)
plt.show()

