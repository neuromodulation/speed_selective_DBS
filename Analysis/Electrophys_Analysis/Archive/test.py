import numpy as np

def interpolate_nan(array):
    # Copy the original array to avoid modifying it
    array = array.copy()
    # Identify the indices of NaN values
    nans = np.isnan(array)
    # Identify the indices of non-NaN values
    non_nans = ~nans
    # Perform linear interpolation
    array[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), array[non_nans])
    return array

# Example usage
data = np.array([1, 1, np.nan, 2, 2, np.nan, 3, 3, np.nan])
interpolated_data = interpolate_nan(data)
print(interpolated_data)