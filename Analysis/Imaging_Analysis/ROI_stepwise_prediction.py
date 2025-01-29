# Predict the behavioral effect using a stepwise regression

# Import useful libraries
import os
import sys
import pandas as pd
import seaborn as sb
from scipy.stats import pearsonr, spearmanr
sys.path.insert(1, "../../../Code")
import utils as u
import numpy as np
import mat73
from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, BayesianRidge
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
from openpyxl import load_workbook
matplotlib.use('TkAgg')

# Set parameters
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

# Delete subject 3 (different electrode type) and chose the recovery block
y = y[np.delete(np.arange(24), 3), 1]

# Load the mat files containing the connectivity values
if Fz:
    matrix_filename = f"../../../Data/{med}/processed_data/Cau-_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat"
else:
    matrix_filename = f"../../../Data/{med}/processed_data/Cau-_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat"
conn_mat_original = mat73.loadmat(matrix_filename)["X"]
seeds = np.array(mat73.loadmat(matrix_filename)["seeds"])
n_targets = len(conn_mat_original)-len(y)
targets = np.array([seed[0].split("\\")[-1].split(".")[0] for seed in seeds[:n_targets]])
X = conn_mat_original[:n_targets, n_targets:].T

# Remove 2 regions
idx_SNc = np.where(targets == "Substantia_nigra_pars_reticulata")[0][0]
idx_inf = np.where(targets == "Frontal_Inf_")[0][0]
X = np.delete(X, [idx_SNc, idx_inf], axis=1)
targets = np.delete(targets, [idx_SNc, idx_inf])

# Try stepwise regression
from sklearn.feature_selection import SequentialFeatureSelector
model = LinearRegression()
sfs2 = SequentialFeatureSelector(model, direction="backward", cv=len(y))
selected_features2 = sfs2.fit(X, y)
print("Stepwise regression")
print(targets[selected_features2.support_])