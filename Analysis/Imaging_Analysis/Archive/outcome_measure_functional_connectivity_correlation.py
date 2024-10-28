# Correlate the connectivity computed with Lead DBS/mapper with an outcome measure
# Save in excel sheet with correlation and p value for both, contra and ipsilateral

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
import matplotlib.pyplot as plt
import matplotlib
from openpyxl import load_workbook
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
    x = np.load(f"../../../Data/{med}/processed_data/res_inst_{feature_name}_{method}.npy")
    x = np.vstack((x, x)).T
else:
    # Load matrix containing the outcome measure
    x = np.load(f"../../../Data/{med}/processed_data/res_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}.npy")
# Delete subject 3 (different electrode type)
x = x[np.delete(np.arange(24), 3), :]

# Load the excel sheet containing the subjects info
path_table = "C:\\Users\\ICN\\Charité - Universitätsmedizin Berlin\\Interventional Cognitive Neuromodulation - " \
             "PROJECT ReinforceVigor\\vigor_stim_task\\Data\\Dataset_list_Off.xlsx"
df_info = pd.read_excel(path_table)
hand = df_info["Hand"][1:25].to_numpy()
hand = hand[np.delete(np.arange(24), 3)]

# Load the mat files containing the connectivity values
if Fz:
    matrix_filename = "results\\Cau-_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat"
    matrix_filename_L = "results\\Cau-_L_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat"
    matrix_filename_R = "results\\Cau-_R_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat"
else:
    matrix_filename = "results\\Cau-_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat"
    matrix_filename_L = "results\\Cau-_L_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat"
    matrix_filename_R = "results\\Cau-_R_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat"
conn_mat = mat73.loadmat(matrix_filename)["X"]
conn_mat_L = mat73.loadmat(matrix_filename_L)["X"]
conn_mat_R = mat73.loadmat(matrix_filename_R)["X"]

seeds = mat73.loadmat(matrix_filename)["seeds"]
n_targets = len(conn_mat)-len(x)
targets = np.array([seed[0].split("\\")[-1].split(".")[0] for seed in seeds[:n_targets]])

# Get a matrix for ipsilateral and contralateral connectivity
conn_mat_contra = np.zeros(conn_mat_R.shape)
conn_mat_ipsi = np.zeros(conn_mat_R.shape)
# Loop over subject and insert correct column
for s in range(len(x)):
    hand_s = hand[s]
    s_id = len(conn_mat)-len(x) + s
    if hand_s == "R":
        conn_mat_ipsi[:, s_id] = conn_mat_R[:, s_id]
        conn_mat_contra[:, s_id] = conn_mat_L[:, s_id]
    else:
        conn_mat_ipsi[:, s_id] = conn_mat_L[:, s_id]
        conn_mat_contra[:, s_id] = conn_mat_R[:, s_id]

# Loop over the structures and compute correlations
res = []
for j in range(2):
    res_tmp = []
    for i in range(n_targets):
        corr, p = spearmanr(x[:, j], conn_mat[i, n_targets:])
        corr_contra, p_all = spearmanr(x[:, j], conn_mat_contra[i, n_targets:])
        corr_ipsi, p_ipsi = spearmanr(x[:, j], conn_mat_ipsi[i, n_targets:])
        res_tmp.extend(np.array([[corr, p], [corr_contra, p_all], [corr_ipsi, p_ipsi]]))
    res.append(res_tmp)
res = np.hstack((res[0], res[1]))

# Plot the correlation
conn_mats = [conn_mat, conn_mat_contra, conn_mat_ipsi]
hemi_names = ["All", "Contra", "Ipsi"]
block_names = ["Stim", "Recovery"]
if plot:
    for i in range(n_targets):
        plt.figure(figsize=(10, 6))
        for j in range(2):
            for k in range(3):
                plt.subplot(2, 3, j * 3 + k + 1)
                corr, p = spearmanr(x[:, j], conn_mats[k][i, n_targets:])
                p = np.round(p, 3)
                if p < 0.05:
                    label = f" R = {np.round(corr, 2)} " + "$\\bf{p=}$" + f"$\\bf{p}$"
                else:
                    label = f" R = {np.round(corr, 2)} p = {p}"
                sb.regplot(x=x[:, j], y=conn_mats[k][i, n_targets:], label=label,
                           scatter_kws={"color": "indianred"}, line_kws={"color": "teal"})
                # Adjust plot
                plt.legend(loc="upper right", fontsize=11)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlabel(f"{method} {feature_name}", fontsize=12)
                plt.ylabel(f"Functional connectivity", fontsize=12)
                plt.title(f"{hemi_names[k]} {block_names[j]}", fontsize=12)
        # Adjust plot
        plt.suptitle(f"{targets[i]}", fontsize=14)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Save figure
        dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
        plt.savefig(
            f"../../../Figures/{dir_name}/corr_fmri_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}.svg",
            format="svg", bbox_inches="tight", transparent=False)
        plt.savefig(
            f"../../../Figures/{dir_name}/corr_fmri_{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}.png",
            format="png", bbox_inches="tight", transparent=False)
    plt.show()

# Save as excel sheet
targets = np.array([[f"{target} All", f"{target} Contra", f"{target} Ipsi"] for target in targets]).flatten()
res_final = np.hstack((targets[:, np.newaxis], np.round(res, 4)))
res_df = pd.DataFrame(res_final, columns=['Structure', 'Correlation Stim', 'P Stim', 'Correlation Recovery', 'P Recovery'])
# Save
try:
    with pd.ExcelWriter("results\\functional_connectivity_structures_results.xlsx", engine='openpyxl', mode='a') as writer:
        res_df.to_excel(writer, sheet_name=f"{feature_name}_{mode}_{method}_{n_norm}_{n_cutoff}_{Fz}_800")
except:
    print("ERROR:Sheet already exists")
