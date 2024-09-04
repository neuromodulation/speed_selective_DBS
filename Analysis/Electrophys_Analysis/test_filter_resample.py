import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat, savemat
from scipy.stats import zscore
import numpy as np
import pandas as pd
import os
import json
import mne_bids
import matplotlib
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths, write_raw_bids
from tkinter import filedialog as fd
import shutil
matplotlib.use('Qt5Agg')
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u

# import brainvision file
path1 = "C:\\Users\\ICN\Documents\\VigorStim\\Neurophys\\Backup\\sub-EL012\\ses-EcogLfpMedOff02\\ieeg\\sub-EL012_ses-EcogLfpMedOff02_task-VigorStimL_acq-StimOnB_run-1_desc-neurophys_ieeg.vhdr"
raw1 = mne.io.read_raw_brainvision(path1, preload=True)

plt.figure()
plt.plot(raw1.times[(raw1.times > 100) & (raw1.times < 200)], raw1.get_data()[0].T[(raw1.times > 100) & (raw1.times < 200)], label="raw")

# Filter
raw1_filt = raw1.copy().filter(l_freq=2, h_freq=200)
plt.plot(raw1.times[(raw1.times > 100) & (raw1.times < 200)], raw1_filt.get_data()[0].T[(raw1.times > 100) & (raw1.times < 200)], label="filtered")

# Resample
raw1_resample = raw1.copy().resample(500)
plt.plot(raw1_resample.times[(raw1_resample.times > 100) & (raw1_resample.times < 200)], raw1_resample.get_data()[0].T[(raw1_resample.times > 100) & (raw1_resample.times < 200)], label="resampled")


path2 = "C:\\Users\\ICN\Documents\\VigorStim\\Neurophys\\Backup\\sub-EL012\\ses-EcogLfpMedOff02\\ieeg\\sub-EL012_ses-EcogLfpMedOff02_task-VigorStimL_acq-StimOnB_run-1_ieeg.vhdr"
raw2 = mne.io.read_raw_brainvision(path2, preload=True)
plt.plot(raw2.times[(raw2.times > 100) & (raw2.times < 200)], raw2.get_data()[0].T[(raw2.times > 100) & (raw2.times < 200)], label="raw2")

plt.legend()
plt.show()

print("test")