import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import mne
from scipy.stats import percentileofscore, spearmanr, pearsonr
from sklearn.preprocessing import PowerTransformer


def gaussianize(array):

    # Apply Box-Cox transformation
    bc = PowerTransformer(standardize=False, method="yeo-johnson")
    array = array.reshape(-1, 1)
    array_trans = bc.fit(array).transform(array)

    return array_trans.squeeze()


def norm_0_1(array):
    """Return array normalized to values between 0 and 1"""
    return (array - np.min(array)) / (np.max(array - np.min(array)))


def fill_outliers_nan(array_all, threshold=3):
    """Fill outliers in 1D array with nan given the array containing the data from all subjects"""

    # Determine the thresholds for outlier detection
    arr_flat = array_all.flatten()
    thres_max = np.nanmean(arr_flat) + threshold * np.nanstd(arr_flat)
    thres_min = np.nanmean(arr_flat) - threshold * np.nanstd(arr_flat)

    # Get index of outliers
    idx_outlier = np.where((array_all > thres_max) | (array_all < thres_min))

    # Replace with nan
    array_all[idx_outlier] = np.NAN

    return array_all


def fill_outliers_nan_ephys(array, threshold=2.5):
    """Fill outliers in 1D array with nan"""
    # Get index of outliers
    idx_outlier = np.where((array > 4e-06))[0]
    array[idx_outlier] = np.NAN
    idx_outlier = np.where(np.abs(zscore(array, nan_policy='omit')) > threshold)[0]
    array[idx_outlier] = np.NAN
    return array


def norm_perc(array, n_norm=5):
    """Normalize feature to stimulation block start and return as percentage"""
    mean_start = np.nanmean(array[..., :n_norm], axis=-1)[..., np.newaxis]
    array_norm_perc = ((array - mean_start) / mean_start) * 100
    return array_norm_perc


def norm(array, n_norm=3):
    """Normalize feature to mean of both conditions (over time)"""
    mean_start = np.nanmean(array[..., :n_norm], axis=-1)[..., np.newaxis]
    array_norm = array - mean_start
    return array_norm


def norm_mean(array, n_norm=3):
    """Normalize feature to mean of both conditions (over time)"""
    mean_start = np.nanmean(array[..., :n_norm], axis=-1)[..., np.newaxis]
    array_norm = array - mean_start
    return array_norm


def despine(sides=['right', 'top']):
    axes = plt.gca()
    axes.spines[sides].set_visible(False)


def append_previous_n_samples(X: np.ndarray, y: np.ndarray, n: int = 5):
    """
    stack feature vector for n samples
    """
    TIME_DIM = X.shape[0] - n
    FEATURE_DIM = int(n * X.shape[1])
    time_arr = np.empty((TIME_DIM, FEATURE_DIM))
    for time_idx, time_ in enumerate(np.arange(n, X.shape[0])):
        for time_point in range(n):
            time_arr[
                time_idx,
                time_point * X.shape[1] : (time_point + 1) * X.shape[1],
            ] = X[time_ - time_point, :]
    return time_arr, y[n:]


def smooth_moving_average(array, window_size=5, axis=2):
    """Return the smoothed array where values are averaged in a moving window"""
    box = np.ones(window_size) / window_size
    array_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='same'), axis=axis, arr=array)
    return array_smooth


def average_windows(array, n_steps):
    """Average over windows and return new array"""
    steps = np.linspace(0, len(array), n_steps)
    array_new = np.zeros(n_steps)
    for i in range(n_steps - 1):
        array_new[i] = np.mean(array[int(steps[i]):int(steps[i + 1])])
    return array_new


def plot_conds(array, var=None, color_slow="#00863b", color_fast="#3b0086"):
    """array = (conds x trials)
    Plot data divided into two conditions, if given add the variance as shaded area"""
    # Plot without the first 5 movements
    plt.plot(array[0, :], label="Slow", color=color_slow, linewidth=3)
    plt.plot(array[1, :], label="Fast", color=color_fast, linewidth=3)
    # Add line at 0
    plt.axhline(0, linewidth=2, color="black", linestyle="dashed")
    x = np.arange(array.shape[1])
    # Add variance as shaded area
    if var is not None:
        plt.fill_between(x, array[0, :] - var[0, :], array[0, :] + var[0, :], color=color_slow, alpha=0.2)
        plt.fill_between(x, array[1, :] - var[1, :], array[1, :] + var[1, :], color=color_fast, alpha=0.2)


def plot_bar_points_connect(matrix, colors, labels, alpha_bar=0.5, line_width=0.7):
    """Matrix: Samples x Conditions (2)
    Plot the mean as a bar and add points for each sample connected by a line"""
    pos_1 = 1.25
    pos_2 = 1.75
    plt.bar(pos_1, np.mean(matrix, axis=0)[0], color=colors[0], label=labels[0], width=0.5, alpha=alpha_bar)
    plt.bar(pos_2, np.mean(matrix, axis=0)[1], color=colors[1], label=labels[1], width=0.5, alpha=alpha_bar)

    # Add points and connecting lines
    for dat in matrix:
        plt.plot(pos_1, dat[0], marker='o', markersize=3, color=colors[0])
        plt.plot(pos_2, dat[1], marker='o', markersize=3, color=colors[1])
        # Add line connecting the points
        plt.plot([pos_1, pos_2], dat, color="black", linewidth=line_width, alpha=0.5)


def get_peak_idx(raw):
    """Return the index at which trials ends (speed crosses threshold) from raw mne object"""

    speed = raw.get_data(["SPEED"])
    blocks = raw.get_data(["BLOCK"])
    trials = raw.get_data(["TRIAL"])
    n_trials = int(np.max(trials))
    n_blocks = int(np.max(blocks))
    peak_idx = []
    for i_block in range(1, n_blocks+1):
        for i_trial in range(1, n_trials + 1):
            mask = np.where(np.logical_and(blocks == i_block, trials == i_trial))[1]
            peak_idx.append(mask[np.argmax(speed[:, mask])])

    return peak_idx


def add_new_channel(raw, new_chan, new_chan_name, type):

    info = mne.create_info([new_chan_name], raw.info['sfreq'], type)
    # Add channel to raw object
    new_chan_raw = mne.io.RawArray(new_chan, info)
    raw.add_channels([new_chan_raw], force_update_info=True)


def scale_min_max(X, min, max, X_min, X_max):
    X_std = (X - X_min) / (X_max - X_min)
    X_scaled = X_std * (max - min) + min
    return X_scaled

def permutation_correlation(x, y, n_perm=10000, method='spearman'):
    """Permutation correlation"""

    # Calculate observed Spearman correlation
    if method == 'spearman':
        observed_corr, _ = spearmanr(x, y)
    else:
        observed_corr, _ = pearsonr(x, y)

    permuted_correlations = []
    # Perform permutation test
    for _ in range(n_perm):
        np.random.shuffle(y)  # Permute one variable while keeping the other fixed
        if method == 'spearman':
            permuted_corr, _ = spearmanr(x, y)
        else:
            permuted_corr, _ = pearsonr(x, y)
        permuted_correlations.append(permuted_corr)

    # Calculate p-value
    p_value = np.mean(np.abs(permuted_correlations) >= np.abs(observed_corr))

    return observed_corr, p_value


def diff_mean_statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


# Generate permuted samples by flipping signs
def generate_permutations(data, num_permutations=10000):
    permuted_means = []
    for _ in range(num_permutations):
        flipped_data = data * np.random.choice([-1, 1], size=len(data), replace=True)
        permuted_means.append(np.mean(flipped_data))
    return np.array(permuted_means)


def create_sequences(data, label, n_steps):
    X = []
    y = []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(label[i+n_steps])
    return np.array(X), np.array(y)


def get_sig_text(p):
    if p < 0.001:
        text = "***"
    elif p < 0.01:
        text = "**"
    elif p < 0.05:
        text = "*"
    else:
        text = "n.s."
    return text