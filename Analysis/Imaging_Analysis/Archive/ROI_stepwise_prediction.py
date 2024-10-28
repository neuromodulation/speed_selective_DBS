# Predict the behavioral effect using a stepwise regression

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
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, BayesianRidge
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
from openpyxl import load_workbook
matplotlib.use('TkAgg')

# Set parameters
med = "Off"
Fz = True
feature_name = "peak_speed"
mode = "diff"
method = "median"
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
    matrix_filename = "results\\Cau-_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmatrix.mat"
else:
    matrix_filename = "results\\Cau-_conn-PPMI74P15CxPatients_desc-AvgR_funcmatrix.mat"
conn_mat_original = mat73.loadmat(matrix_filename)["X"]
seeds = np.array(mat73.loadmat(matrix_filename)["seeds"])
n_targets = len(conn_mat_original)-len(y)
X = conn_mat_original[:n_targets, n_targets:].T

# Train a regression model (LASSO)
for alpha in [0, 0.01, 0.1, 1]:
    clf = Lasso(alpha=alpha)
    clf.fit(X, y)
    coeffs = clf.coef_
    print(f"alpha = {alpha}")
    for i in range(len(coeffs)):
            print(seeds[i], coeffs[i])


# Elastic Net
for alpha in [0, 0.01, 0.1, 1]:
    clf = ElasticNet(alpha=alpha)
    clf.fit(X, y)
    coeffs = clf.coef_
    print(f"alpha = {alpha}")
    for i in range(len(coeffs)):
            print(seeds[i], coeffs[i])


# Try stepwise regression
from sklearn.feature_selection import SequentialFeatureSelector
model = LinearRegression()
sfs2 = SequentialFeatureSelector(model, n_features_to_select=1, direction="forward")
selected_features2 = sfs2.fit(X, y)
print("Stepwise regression")
print(seeds[:n_targets][selected_features2.support_])