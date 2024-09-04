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
from sklearn import metrics, model_selection, linear_model, ensemble
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

# Loop through the subjects
subjects = [sub for sub in df["ID Berlin_Neurophys"][1:21] if not (sub == "L003" or sub == "EL016")]

# Get the decoding performance for each channel type
performance_all_sub = np.zeros((len(subjects), 6))
for i_sub, sub in enumerate(subjects):
    print(sub)

    if sub == "L003":
        pass
    else:
        feature_reader = nm_analysis.Feature_Reader(
            feature_dir=output_root,
            feature_file=sub
        )
        # Decode
        model = linear_model.LinearRegression()
        #model = linear_model.Lasso()

        feature_reader.decoder = nm_decode.Decoder(
            STACK_FEATURES_N_SAMPLES=True,
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

        # Plot the predicted peak speed
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 5))
        for i in range(3):
            x = feature_reader.decoder.all_ch_results["y_test_pr"][i]
            y = feature_reader.decoder.all_ch_results["y_test"][i]
            axes[i][0].plot(x)
            axes[i][0].plot(y)
            axes[i][0].plot(y)
            corr, p = spearmanr(x, y, nan_policy='omit')
            sb.regplot(x=x, y=y, ax=axes[i][1])
            axes[i][1].set_title(f"corr = {np.round(corr, 3)}, p = {np.round(p, 3)}")
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        #plt.show()
        plt.close()

        # Save the performance of the best channel and all channels combined
        if sub == "EL012" or sub == "L013":
            types = ["LFP_R", "LFP_L", "ECOG_R", "ECOG_L", "EEG"]

        else:
            types = ["LFP_L", "LFP_R", "ECOG_L", "ECOG_R", "EEG"]
        for i_type, type in enumerate(types):
            # Save the decoding performance of the best channel
            idx = [j for j, ch in enumerate(df_per["ch"]) if type in ch]
            performance_all_sub[i_sub, i_type] = np.max(df_per["performance_test"][idx])
            # Save the decoding for all channels combined
        performance_all_sub[i_sub, -1] = np.array(df_per["performance_test"])[-1]

# Analyze
# Delete nan values
mask = ~np.isnan(performance_all_sub)
performance_no_nan = [d[m] for d, m in zip(performance_all_sub.T, mask.T)]
plt.boxplot(performance_no_nan)
names = ["STN_LFP_Contralateral", "STN_LFP_Ipsilateral", "ECOG_Contralateral", "ECOG_Ipsilateral", "EEG", "Combined"]
plt.xticks(ticks=np.arange(1, 7), labels=names, fontsize=16, rotation=45)
plt.subplots_adjust(bottom=0.2)
plt.show()
print("END")
