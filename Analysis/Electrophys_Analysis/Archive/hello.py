import os
from joblib import Parallel, delayed
import mne_bids
import numpy as np
import mne
from bayes_opt import BayesianOptimization
from sklearn import metrics, model_selection, linear_model
from sklearn.model_selection import KFold
import pandas as pd
from scipy.stats import zscore

import sys
input1 = sys.argv[1]
input2 = sys.argv[2]
print(f"input 1: {input1} input2: {input2}")