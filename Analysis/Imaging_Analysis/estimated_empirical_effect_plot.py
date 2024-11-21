# Results Figure 3: Correlation between estimated and empirical stimulation effect
# SMA and putamen

# Import useful libraries
import os
import sys
import seaborn as sb
from scipy.stats import pearsonr
sys.path.insert(1, "../../../Code")
import utils as u
import numpy as np
import mat73
from sklearn.linear_model import LinearRegression
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

# Set parameters
plot = True
med = "Off"
Fz = True
feature_name = "mean_speed"
mode = "mean"
method = "mean"
n_norm = 5
n_cutoff = 5
target_names = ['Supp_Motor_Area', 'Put-']
names = ["SMA", "Putamen"]

# Load matrix containing the outcome measure
if mode == "inst":
    y = np.load(f"../../../Data/{med}/processed_data/res_inst_{feature_name}_{method}.npy")
    y = np.vstack((y, y)).T
else:
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
colors = ["#B3F2DD", "#f2b3c8"]

for t, target_name in enumerate(target_names):

    # Get the connectivity values for the cortico-basal ganglia pathway
    idx_targets = np.array([np.where(targets == target)[0][0] for target in [target_name]])
    conn_mat = conn_mat_original[idx_targets, n_targets:]

    # Compute correlations between targets
    corr_mat = np.corrcoef(conn_mat, rowvar=True)

    # Fit a linear regression between the correlation coefficients of the remaining subjects and the outcome measure
    pred_outcome = np.zeros(len(y))
    for i in range(len(y)):
          idx_train = np.delete(np.arange(len(y)), i)
          x_train = conn_mat[:, idx_train]
          y_train = y[idx_train]
          # Create a linear regression model
          model = LinearRegression()
          model.fit(x_train.T, y_train)
          # Predict the outcome measure for left-out subject
          pred_outcome[i] = model.predict(conn_mat[:, i][:, np.newaxis].T)

    # Plot
    plt.figure(figsize=(1, 1))
    corr_res, p = scipy.stats.pearsonr(y[:len(pred_outcome)], pred_outcome)
    #corr_res, p = u.permutation_correlation(y[:len(pred_outcome)], pred_outcome, n_perm=10000)
    p = np.round(p, 3)
    sb.regplot(x=y[:len(pred_outcome)], y=pred_outcome,
         scatter_kws={"color": "grey", 's': 1.2}, line_kws={"color": colors[t], 'linewidth':1})
    # Adjust plot
    fontsize=7
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    plt.xlabel("Empirical effect", fontsize=fontsize, labelpad=0.5)
    plt.ylabel("Estimated effect", fontsize=fontsize, labelpad=0.5)
    plt.title(f"R² = {np.round(corr_res*corr_res, 2)} p = {p}", fontsize=fontsize-1, pad=0.5)
    u.despine()

    # Save figure
    dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
    plt.savefig(
    f"../../../Figures/{dir_name}/estimate_{target_name[:10]}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}.pdf",
    format="pdf", bbox_inches="tight", transparent=True)
    plt.savefig(
    f"../../../Figures/{dir_name}/estimate_{target_name[:10]}_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}.png",
    format="png", bbox_inches="tight", transparent=False)

plt.show()