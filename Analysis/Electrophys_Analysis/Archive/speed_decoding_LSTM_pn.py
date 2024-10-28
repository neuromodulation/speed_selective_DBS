# Decode speed using py_neuromodulation

import os

import mne_bids
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import sys
import math
from sklearn import metrics, model_selection, linear_model
import py_neuromodulation as nm
from xgboost import XGBClassifier
from scipy.stats import zscore
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
    nm_settings,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed
sys.path.insert(1, "C:/CODE/ac_toolbox/")
import utils as u
import matplotlib
matplotlib.use('Qt5Agg')

# Set analysis parameters
samp_freq = 30
seg_ms = 300

# Loop over blocks (break in between during which behavior is not recorded but ephys recording continues)
feature_reader = nm_analysis.FeatureReader(
    feature_dir="..\\..\\..\\Data\\Off\\processed_data\\",
    feature_file=f"{samp_freq}_{seg_ms}",
)

feature_reader.plot_all_features(ch_used="ECOG_bipolar", clim_high=4, clim_low=-4)

# Set the label
feature_reader.label_name = "SPEED_MEAN"
feature_reader.label = feature_reader.feature_arr[feature_reader.label_name]

# Plot the label
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(feature_reader.label, bins=100)
ax2.plot(feature_reader.label)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(feature_reader.label, bins=100)
ax2.plot(feature_reader.label)

# Normalize between 0 and 1
#feature_reader.label = (feature_reader.label - np.min(feature_reader.label)) / (np.max(feature_reader.label) - np.min(feature_reader.label))
#feature_reader.label = zscore(feature_reader.label)


# Split into train and test sets
data = feature_reader.feature_arr
label = feature_reader.label
train_size = int(len(data) * 0.8)
train_data, test_data = np.array(data[:train_size]), np.array(data[train_size:])
train_label, test_label = np.array(label[:train_size]), np.array(label[train_size:])
n_steps = 10
features = data.shape[-1]

# Create sequences for training set
X_train, y_train = u.create_sequences(train_data, train_label, n_steps)

# Create sequences for testing set
X_test, y_test = u.create_sequences(test_data, test_label, n_steps)

# Build the LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
model_lstm.add(Dense(units=1))
# Compiling the model
model_lstm.compile(optimizer="RMSprop", loss="mse")
# Train the model
model_lstm.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)

# Plot the training
plt.plot(model_lstm.history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
trainScore = model_lstm.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model_lstm.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# Plot the predicted vs empirical values
# make predictions
trainPredict = model_lstm.predict(X_train)
testPredict = model_lstm.predict(X_test)
