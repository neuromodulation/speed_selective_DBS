# Validate the correlation using leave-one-out crossvalidation

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
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
mask = "pathway"
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
res = np.zeros((len(subjects), len(subjects), 2))
for i, sub in enumerate(subjects):

       print(sub)

       # Load the R map computed without the subject
       X = loadmat(f"../../../Data/{med}/processed_data/vw_corr_{sub}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.mat")
       corr = X["corr"]

       for j, sub2 in enumerate(subjects):

              # Load the connectivity map of the patient
              subject_path = f"C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\LeadDBSDataset\\derivatives\\leaddbs\\sub-{sub2}\\stimulations\\MNI152NLin2009bAsym\\gs_test\\"
              # Load nifti file
              if Fz:
                     file_name = f"sub-{sub2}_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmap.nii"
              else:
                     file_name = f"sub-{sub2}_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgR_funcmap.nii"
              image = nib.load(os.path.join(subject_path, file_name)).get_fdata()

              # Compute the spatial correlation between both maps
              if mask == "none":
                     valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 0].flatten())
                     res_stim, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 0].flatten()[valid_indices])
                     res[i, j, 0] = res_stim
                     valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 1].flatten())
                     res_recov, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 1].flatten()[valid_indices])
                     res[i, j, 1] = res_recov
              else:
                     valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 0].flatten()) & (mask_image.flatten() > 0)
                     res_stim, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 0].flatten()[valid_indices])
                     res[i, j, 0] = res_stim

                     valid_indices = ~np.isnan(image.flatten()) & ~np.isnan(corr[:, :, :, 1].flatten()) & (mask_image.flatten() > 0)
                     res_recov, _ = scipy.stats.pearsonr(image.flatten()[valid_indices], corr[:, :, :, 1].flatten()[valid_indices])
                     res[i, j, 1] = res_recov


# Delete patients for which values are NaN
res[:, :, 1] = res[:, :, 0]
use_patients = ~np.isnan(res).any(axis=1).any(axis=1)
subjects = subjects[use_patients]
res = res[use_patients, :, :][:, use_patients,:]
# Fit a linear regression between the correlation coefficents of the remaining subjects and the outcome measure
pred_outcome = np.zeros((len(subjects), 2))
for block in range(2):
       for i, sub in enumerate(subjects):
              idx_train = np.delete(np.arange(len(subjects)), i)
              idx_train = np.arange(len(subjects))
              y = outcome[idx_train, block]
              x = res[i, idx_train, block][:, np.newaxis]
              # Create a linear regression model
              model = LinearRegression()
              model.fit(x, y)
              # Predict the outcome measure for left-out subject
              pred_outcome[i, block] = model.predict(np.array([res[i, i, block]])[:, np.newaxis])

# Plot the predicted outcome measure and correlate
block_names = ["Stim", "Recovery"]
fig = plt.figure(figsize=(3, 4.5))
for i in range(1):
       plt.subplot(1, 1, i+1)
       #corr_res, p = scipy.stats.spearmanr(outcome[:len(pred_outcome), i], pred_outcome[:, i])#, nan_policy="omit")
       corr_res, p = u.permutation_correlation(outcome[:len(pred_outcome), i], pred_outcome[:, i], n_perm=10000)
       p = np.round(p, 3)
       if p < 0.05:
              label = f" R = {np.round(corr_res, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
       else:
              label = f" R = {np.round(corr_res, 2)} p = {p}"
       corr_res, p = scipy.stats.spearmanr(outcome[:len(pred_outcome), i], pred_outcome[:, i])#, nan_policy="omit")
       sb.regplot(x=outcome[:len(pred_outcome), i], y=pred_outcome[:, i], label=label,
             scatter_kws={"color": "indianred"}, line_kws={"color": "teal"})
       # Adjust plot
       plt.legend(loc="upper right", fontsize=11)
       plt.xticks(fontsize=12)
       plt.yticks(fontsize=12)
       plt.xlabel(f"{method} {feature_name} {mode}", fontsize=12)
       plt.ylabel(f"Predicted {method} {feature_name} {mode}", fontsize=12)
       plt.title(f"{block_names[i]}", fontsize=12)

plt.subplots_adjust(wspace=0.5, hspace=0.5)

# Save
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
       f"../../../Figures/{dir_name}/corr_val_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{mask}.svg",
       format="svg", bbox_inches="tight", transparent=False)
plt.savefig(
       f"../../../Figures/{dir_name}/corr_val_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{mask}.png",
       format="png", bbox_inches="tight", transparent=False)
plt.show()