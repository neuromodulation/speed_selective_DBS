# Analyze features extracted with py_neuromodulation

import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
import os
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import sys
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths
from scipy.stats import pearsonr, spearmanr
import py_neuromodulation as pn
import seaborn as sb
from sklearn import metrics, model_selection, linear_model
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_plots,
)
import matplotlib
#matplotlib.use('Qt5Agg')

sys.path.insert(1, "C:/CODE/ac_toolbox/")

# Specify the medication group
med = "Off"

# Read the list of the datasets
df = pd.read_excel(f'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'
       f'Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/'
       f'Tablet_task/Data/Dataset_list.xlsx', sheet_name=med)

output_root = 'C:/Users/ICN/Charité - Universitätsmedizin Berlin/' \
              'Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/' \
              'Tablet_task/Code/Analysis/'
# Define target folder fpr features
output_root = 'C:/Users/ICN/Charité - Universitätsmedizin Berlin/'\
            f'Interventional Cognitive Neuromodulation - PROJECT ReinforceVigor/'\
            f'Tablet_task/Data/OFF/processed_data/'

# Get the decoding performance for each

# Loop through the subjects
for i_sub, sub in enumerate(df["ID Berlin_Neurophys"][1:21]):
    print(sub)

    if sub == "L003":
        pass
    else:
        """
        # Load features
        features = pd.read_csv(f"{output_root}{sub}/{sub}_FEATURES.csv")

        # Compute correlation between each features and the speed_mean
        corr_p = np.zeros((len(features.keys())-1, 2))
        for i, key in enumerate(features.keys()[:-1]):
            corr, p = spearmanr(features["SPEED_MEAN"], features[key])
            corr_p[i, 0] = corr
            corr_p[i, 1] = p

        # Analyze results
        sig = corr_p[:, 1] > 0.05
        corr_sig = corr_p[:, 0]
        corr_sig[sig] = None
        
        # Average correlations for ecog L, lfp r, lfp l, eeg
        feature_names = features.keys()
        # Get unique features
        features_LFP = [feature for feature in feature_names if "LFP" in feature]
        unique_features = np.unique([feature[16:] for feature in features_LFP])
        types = ["LFP_R", "LFP_L", "EEG", "ECOG_R"]
        average_corr = np.zeros((len(types), len(unique_features)))
        for j, type in enumerate(types):
            for k, unique_feature in enumerate(unique_features):
                idx = [i for i, feature in enumerate(feature_names) if (type in feature) and (unique_feature in feature)]
                # Compute mean (without non significant entries)
                average_corr[j, k] = np.nanmean(corr_sig[idx])

        # Inspect result
        plt.figure(figsize=(12, 8))
        plt.imshow(average_corr, cmap="jet")
        plt.xticks(ticks=np.arange(len(unique_features)), labels=unique_features)
        plt.yticks(ticks=np.arange(len(types)), labels=types)
        plt.colorbar()
        plt.show()
        #print("end")
        """

        feature_reader = nm_analysis.Feature_Reader(
            feature_dir=output_root,
            feature_file=sub
        )
        # Decode
        model = linear_model.LinearRegression()

        feature_reader.decoder = nm_decode.Decoder(
            features=feature_reader.feature_arr,
            label=np.array(feature_reader.feature_arr["SPEED_MEAN"]),
            label_name=feature_reader.label_name,
            used_chs=feature_reader.used_chs,
            model=model,
            eval_method=metrics.r2_score,
            cv_method=model_selection.KFold(n_splits=3, shuffle=True),
        )
        performances = feature_reader.run_ML_model(
            estimate_channels=True,
            estimate_gridpoints=False,
            estimate_all_channels_combined=True,
            save_results=True,
        )
        df_per = feature_reader.get_dataframe_performances(performances)

        print(df_per)

        # Save the performance of the best channel and all channels combined

        types = ["LFP_R", "LFP_L", "EEG", "ECOG_R"]



        print("END")
