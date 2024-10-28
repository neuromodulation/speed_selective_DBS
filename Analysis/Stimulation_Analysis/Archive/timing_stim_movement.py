# Plot the distribution of stimulation over an individual movement


# Import useful libraries
import os
import sys
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Loop over On and Off medication datasets
med = "Off"
fig = plt.figure(figsize=(8.5, 3))

# Load matrix containing time of stimulation and time of peak for each movement
stim_time = np.load(f"../../../Data/{med}/processed_data/stim_time.npy")
peak_speed_time = np.load(f"../../../Data/{med}/processed_data/peak_speed_time.npy")
move_offset_time = np.load(f"../../../Data/{med}/processed_data/move_offset_time.npy")
move_onset_time = np.load(f"../../../Data/{med}/processed_data/move_onset_time.npy")

# Load matrix containing the mean speed around and aligned to the peak
speed_around_peak = np.load(f"../../../Data/{med}/processed_data/speed_around_peak.npy")

# Select only stimulation blocks
stim_time = stim_time[:, :, 0, :]
move_offset_time = move_offset_time[:, :, 0, :]
move_onset_time = move_onset_time[:, :, 0, :]
peak_speed_time = peak_speed_time[:, :, 0, :]
speed_around_peak = speed_around_peak[:, :, 0, :, :]

# Loop over stimulation conditions
cond_names = ["Slow", "Fast"]
for i in range(2):
    # Get the stimulated trials
    stim_mask = ~np.isnan(stim_time[:, i, :])
    stim_time_cond = stim_time[:, i, :][stim_mask]
    peak_speed_time_cond = peak_speed_time[:, i, :][stim_mask]
    stim_perc = stim_time_cond - peak_speed_time_cond

    # Delete outliers outside the time window
    #n_outlier = len(stim_perc[~((stim_perc < np.max(stim_perc)) & (stim_perc > np.min(stim_perc)))])
    #stim_perc = stim_perc[(stim_perc < np.max(stim_perc)) & (stim_perc > np.min(stim_perc))]
    #print(n_outlier)

    # Stimulation lasts 300 ms, so add samples for the next 300 ms
    stim_perc_long = np.array([np.hstack((x, x + np.arange(0.0167, 0.3, 0.0167))) for x in stim_perc]).flatten()

    # Plot the averaged speed around the peak with standard deviation
    plt.subplot(1, 2, i+1)
    # Remove outliers
    np.apply_along_axis(lambda m: u.fill_outliers_nan(m), axis=0, arr=speed_around_peak)
    # Compute mean and std
    mean_speed = np.nanmean(speed_around_peak[:, i, :, :][stim_mask, :], axis=0)
    mean_std = np.nanstd(speed_around_peak[:, i, :, :][stim_mask, :], axis=0)
    # Generate time array (with peak at 0)
    times = np.arange(mean_speed.shape[0]) * 0.0167
    times = times - times[int(np.floor(mean_speed.shape[0]/2))]
    plt.plot(times, mean_speed, color="black", alpha=0.7, linewidth=3)
    plt.fill_between(times, mean_speed-mean_std, mean_speed+mean_std, color="black", alpha=0.2)

    # Plot the distribution of stimulation start over an individual movement
    ax = plt.gca()  # Get current axis
    ax2 = ax.twinx()
    bins = np.linspace(min(times), max(times), 100)
    ax2.hist(stim_perc_long, bins=bins, color="plum", alpha=0.5)

    # Adjust subplot
    ax.set_ylim([0, 6000])
    ax2.set_ylim([0, 800])
    ax.set_xlim([min(times), max(times)])
    ax.set_xlabel("Time [Seconds]", fontsize=12)
    ax.set_title(cond_names[i], fontsize=14)
    if i > 0:
        ax2.set_ylabel(f"Number of stimulations \n at this timepoint", fontsize=12)
        ax.spines[['left', 'top']].set_visible(False)
        ax2.spines[['left', 'top']].set_visible(False)
        ax.get_yaxis().set_visible(False)
    else:
        ax.set_ylabel(f"Average speed \n [Pixel/Seconds]", fontsize=12)
        ax.spines[['right', 'top']].set_visible(False)
        ax2.spines[['right', 'top']].set_visible(False)
        ax2.get_yaxis().set_visible(False)


# Adiust figure
plt.subplots_adjust(bottom=0.1, left=0.15, right=0.85, wspace=0.05, hspace=0.4)
plt.suptitle(med, fontsize=15)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.svg", format="svg", bbox_inches="tight", transparent=True)
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}.png", format="png", bbox_inches="tight", transparent=True)

# Get the average time after peak of stimulation
print(np.mean(np.nanmedian(stim_time - peak_speed_time, axis=(1, 2))))
print(np.std(np.nanmedian(stim_time - peak_speed_time, axis=(1, 2))))

print(np.mean(np.nanmedian(move_offset_time - move_onset_time, axis=(1, 2))))
print(np.std(np.nanmedian(move_offset_time - stim_time, axis=(1, 2))))

print(np.mean(np.nanmedian(stim_time - move_onset_time, axis=(1, 2))))
print(np.std(np.nanmedian(stim_time - move_onset_time, axis=(1, 2))))

# Plot as barplot
dur = np.nanmedian(move_offset_time - move_onset_time, axis=(1, 2)) * 1000
dur_mean = np.mean(dur)
stim_onset = np.nanmedian(stim_time - move_onset_time, axis=(1,2)) * 1000
stim_onset_mean = np.mean(stim_onset)
color = "lightgrey"
fontsize=7
width = 0.5
fig, ax = plt.subplots(1, 1, figsize=(1.3, 1.3))
ax.bar(0, dur_mean, color=color, width=width)
for dat in dur:
    ax.plot(0, dat, color="black", marker=".", markersize=0.5)
ax.bar(1, stim_onset_mean, color=color, width=width)
for dat in stim_onset:
    ax.plot(1, dat, color="black", marker=".", markersize=0.5)
ax.fill_between([1-(width/2), 1+(width/2)], [stim_onset_mean, stim_onset_mean], [stim_onset_mean+300, stim_onset_mean+300],
                color="red", alpha=0.3, label="Stimulation", edgecolor=(1,1,1))
ax.set_xticks(ticks=[0, 1], labels=["Movement \nduration", "Stimulation \nonset"], fontsize=fontsize, rotation=0)
ax.set_ylabel("Milliseconds", fontsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize-2)
ax.spines[['right', 'top']].set_visible(False)
ax.legend(fontsize=fontsize-2)

# Save
plot_name = os.path.basename(__file__).split(".")[0]
dir_name = os.path.dirname(os.path.realpath(__file__)).split("\\")[-1]
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_times.pdf", format="pdf", transparent=True, bbox_inches="tight")
plt.savefig(f"../../../Figures/{dir_name}/{plot_name}_times.png", format="png", transparent=True, bbox_inches="tight")
plt.show()
