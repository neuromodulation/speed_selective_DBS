# Results Figure 3: Prediction accuracy of stimulation effect using connectivity to a series of Regions of Interest (ROI)

# Import useful libraries
import os
import sys
from scipy.stats import pearsonr
sys.path.insert(1, "../Code")
import utils as u
import numpy as np
import mat73
from sklearn.linear_model import LinearRegression
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Set parameters
plot = True
med = "Off"
Fz = True
feature_name = "mean_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5

if mode == "inst":
    y = np.load(f"../../../Data/{med}/processed_data/res_inst_{feature_name}_{method}.npy")
    y = np.vstack((y, y)).T
else:
    # Load matrix containing the outcome measure
    y = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
# Delete subject 3 (different electrode type)
y = y[np.delete(np.arange(24), 3), 1]

# Load the mat files containing the connectivity values
if Fz:
    matrix_filename = f"../../../Data/{med}/processed_data/Cau-_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat"
else:
    matrix_filename = f"../../../Data/{med}/processed_data/Cau-_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat"
conn_mat_original = mat73.loadmat(matrix_filename)["X"]
seeds = mat73.loadmat(matrix_filename)["seeds"]
n_targets = len(conn_mat_original)-len(y)
targets = np.array([seed[0].split("\\")[-1].split(".")[0] for seed in seeds[:n_targets]])

# Compute the R2 value for each ROI
res = np.zeros(n_targets)
for i, target_name in enumerate(targets):

    # Get the connectivity values for the cortico-basal ganglia pathway
    conn_mat = conn_mat_original[i, n_targets:]

    # Compute correlations between targets
    corr_mat = np.corrcoef(conn_mat, rowvar=True)

    # Fit a linear regression between the correlation coefficients of the remaining subjects and the outcome measure
    pred_outcome = np.zeros(len(y))
    for j in range(len(y)):
        idx_train = np.delete(np.arange(len(y)), j)
        x_train = conn_mat[idx_train, np.newaxis]
        y_train = y[idx_train]
        # Create a linear regression model
        model = LinearRegression()
        model.fit(x_train, y_train)
        # Predict the outcome measure for left-out subject
        pred_outcome[j] = model.predict(np.array([conn_mat[j]])[:, np.newaxis])
    # Compute correlation between estimated and empirical outcome
    R, p = scipy.stats.pearsonr(y[:len(pred_outcome)], pred_outcome)
    # Save
    res[i] = R*R

# Delete the frontal eye field and the SNc
idx_SNc = np.where(targets == "Substantia_nigra_pars_reticulata")[0][0]
idx_inf = np.where(targets == "Frontal_Inf_")[0][0]
targets = np.delete(targets, [idx_SNc, idx_inf])
res = np.delete(res, [idx_SNc, idx_inf])

# Ordered by highest number
idx_res_ordered = np.argsort(res)
res_orderd = res[idx_res_ordered]
targets_ordered = targets[idx_res_ordered]
# Define names of targets
labels = np.array(["Caudate", "Crus II", "DN", "GPe", "GPi", "M1", "Putamen", "STN", "SNr", "SMA", "CMPf","VLa"])
labels_ordered = labels[idx_res_ordered]

# Plot as bar plot
fig, ax = plt.subplots(figsize=(1.3, 2.5))
ax.barh(labels_ordered, res_orderd, height=0.6, color=(119/255, 119/255, 119/255))
plt.yticks(fontsize=8)
ax.yaxis.set_tick_params(length=0)
plt.xticks(fontsize=8)
plt.xlabel(f"$R^2$", fontsize=10)
plt.ylabel("Regions of interest (ROI)", fontsize=9, labelpad=1)
plt.subplots_adjust(left=0.2)
u.despine()

# Save figure
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(
    f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}.pdf",
    format="pdf", bbox_inches="tight", transparent=True)
plt.savefig(
    f"../../../Figures/{dir_name}/{plot_name}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}.png",
    format="png", bbox_inches="tight", transparent=False)
plt.show()