# Create nifit file of R-maps for plotting

# Import useful libraries
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

       nii_new = nib.Nifti1Image(corr_tmp, nii.affine, nii.header)
       nib.save(nii_new, f'C:\\Users\\ICN\\Documents\\Try_without_Sdrive\\{block_names[i]}_R_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_{corr_method}')

