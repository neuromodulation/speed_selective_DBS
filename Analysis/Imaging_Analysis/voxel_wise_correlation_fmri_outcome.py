# Correlate the functional connectivity of each voxel (with the VTA) with an outcome measure
# Validate using leave-one-out crossvalidation (compute R map for leaving one patient out and compute distance from left out patient to computed R-map)


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
import scipy.stats as stats
from scipy.io import savemat
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

# Load matrix containing the outcome measure
x = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
# Delete subject for whom the electrode type is different
x = x[np.delete(np.arange(24), 3), :]
#x = x[np.delete(np.arange(23), 10), :]

# Loop over subjects
subjects = np.concatenate((np.arange(2, 5), np.arange(6, 26)))
#subjects = np.arange(17, 26)
#subjects = subjects[np.delete(np.arange(23), 10)]# IDs of the subjects in leaddbs folder
image_all = []
for i, sub in enumerate(subjects):

    subject_path = f"C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\LeadDBSDataset\\derivatives\\leaddbs\\sub-{sub}\\stimulations\\MNI152NLin2009bAsym\\gs_test\\"

    # Load nifti file
    if Fz:
        file_name = f"sub-{sub}_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmap.nii"
    else:
        file_name = f"sub-{sub}_sim-binary_model-simbio_conn-PPMI74P15CxPatients_desc-AvgR_funcmap.nii"

    image = nib.load(os.path.join(subject_path, file_name)).get_fdata()
    image_all.append(image)
image_all = np.array(image_all)

# Leave one patient out
for i in range(len(subjects)+1):
#for i in [len(subjects)]:

    print(f"Subject {i+1}")

    if i == len(subjects):
        idx_train = np.arange(len(subjects))
    else:
        # Delete the test subject from the training set
        idx_train = np.delete(np.arange(len(subjects)), i)

    # Compute correlation for each voxel
    corr = np.zeros((image.shape[0], image.shape[1], image.shape[2], 2))
    p = np.zeros((image.shape[0], image.shape[1], image.shape[2], 2))
    count = 0
    for j in range(image.shape[0]):
        for k in range(image.shape[1]):
            for l in range(image.shape[2]):

                count = count + 1
                progress = (count / (image.shape[0] * image.shape[1] * image.shape[2])) * 100
                # Print progress in the same line
                print(f"Progress: {progress:.2f}%", end='\r', flush=True)

                """if np.mean(image_all[idx_train, j, k, l]) != 0:
                    test = image_all[idx_train, j, k, l]
                    print(stats.shapiro(test))"""

                if corr_method == "pearson":
                    corr[j, k, l, 0], p[j, k, l, 0] = pearsonr(image_all[idx_train, j, k, l], x[idx_train, 0])
                    #corr[j, k, l, 1], p[j, k, l, 1] = pearsonr(image_all[idx_train, j, k, l], x[idx_train, 1])
                else:
                    corr[j, k, l, 0], p[j, k, l, 0] = spearmanr(image_all[idx_train, j, k, l], x[idx_train, 0])
                    #corr[j, k, l, 1], p[j, k, l, 1] = spearmanr(image_all[idx_train, j, k, l], x[idx_train, 1])


    # Save the correlation map for further analysis
    if i == len(subjects):
        savemat(f"../../../Data/{med}/processed_data/vw_corr_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.mat",
                {"corr":corr, "p":p})
    else:
        savemat(f"../../../Data/{med}/processed_data/vw_corr_{subjects[i]}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}.mat",
                {"corr":corr, "p":p})
