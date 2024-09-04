# Script for statistical analysis of feature
# Correlate the stimulation effect (diff fast-slow) with the percentile of the stimulated movements

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
from scipy.stats import pearsonr, spearmanr, percentileofscore
import seaborn as sb
import pandas as pd
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
from pandas import *

# Set analysis parameters
feature_name = "peak_speed"
med = "Off"  # "on", "off", "all"

# Open sheet with UPDRS scores
df = pd.read_excel(f'../../Data/Dataset_list.xlsx', sheet_name=med)
UPDRS = df["UPDRS"]

# Load feature matrix
feature_matrix = np.load(f"../../Data/{med}/processed_data/{feature_name}.npy")

# Select the dataset of interest
feature_matrix = feature_matrix[:, :, :, :]
n_datasets, _,_, n_trials = feature_matrix.shape

# Delete outliers
np.apply_along_axis(lambda m: u.fill_outliers_nan(m, threshold=3), axis=3, arr=feature_matrix)

# Compute mean feature
feature_mean = np.nanmean(feature_matrix, axis=(1, 2, 3))[:-1]
# Compute the correlation
corr, p = spearmanr(UPDRS, feature_mean, nan_policy='omit')
p = np.round(p, 3)
sb.regplot(x=UPDRS, y=feature_mean, label=f" R = {np.round(corr, 2)} "+"$\\bf{p=}$"+f"$\\bf{p}$")
plt.legend()

# Save figure on group basis
plt.savefig(f"../../Figures/corr_UPDRS_{feature_name}_{med}.svg", format="svg", bbox_inches="tight", transparent=True)

plt.show()