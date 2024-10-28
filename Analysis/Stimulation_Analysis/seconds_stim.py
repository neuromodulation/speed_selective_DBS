# Get teh total stimulation time in seconds

# Import useful libraries
import os
import sys
sys.path.append('../Code')
import utils as u
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

med = "Off"

# Load matrix containing 0/1 indicating which trial was stimulated
stim = np.load(f"../../../Data/{med}/processed_data/stim.npy")

# Select only stimulation blocks
stim = stim[:, :, 0, :]

# Get total sec of stimulated movements
stim_perc = ((np.sum(stim, axis=2)) * 300) / 1000

# Print mean and std values
print(np.round(np.mean(stim_perc, axis=0), 3))
print(np.round(np.std(stim_perc, axis=0), 3))

# Load matrix containing the first time point of each trial (to get an estimate of the length of one block)
time_sample = np.load(f"../../../Data/{med}/processed_data/first_time_sample.npy")
# Select only stimulation blocks
time_sample = time_sample[:, :, 0, :]
av_stim_sec = [9.44, 7.56]
for cond in range(2):
    av_dur = np.mean(np.max(time_sample[:, cond, :], axis=1) - np.min(time_sample[:, cond, :], axis=1)) - (0.5*95)
    av_perc = (av_stim_sec[cond] / av_dur) * 100
    print(f"Whole block {av_stim_sec[cond]} sec stim in {av_dur} sec results in {av_perc} %")

# Load the matrix containing the time point of movement offset/onset (to get an estimate of the whole movement time)
move_offset_time = np.load(f"../../../Data/{med}/processed_data/move_offset_time.npy")
move_onset_time = np.load(f"../../../Data/{med}/processed_data/move_onset_time.npy")
# Select only stimulation blocks
move_offset_time = move_offset_time[:, :, 0, :]
move_onset_time = move_onset_time[:, :, 0, :]
move_dur = move_offset_time - move_onset_time
for cond in range(2):
    av_dur = np.nanmean(np.nansum(move_dur[:, cond, :], axis=1))
    av_perc = (av_stim_sec[cond] / av_dur) * 100
    print(f"Only movement {av_stim_sec[cond]} sec stim in {av_dur} sec results in {av_perc} %")
