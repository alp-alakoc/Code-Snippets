import quandl
quandl.ApiConfig.api_key = "bxzypBsr4z1PxD-hw1eh"

#We pull U.S. interest rate data from Quandl in order to build an LSTM regressor (predicting U.S. rates)
data = quandl.get("FMAC/FIX15YR")

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr
import sklearn
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#We perform the necessary pre-screening and adjustments to the data prior to building the LSTM model
dat_cols = data.columns
print(dat_cols)

print(data.head(4))

print(data.describe())
print(data.isna().sum())
print(data.index)

train_data = data[data.index <= pd.to_datetime('2010-12-31')]
test_data = data[data.index > pd.to_datetime('2010-12-31')]

#We find where the Nas in the data begin after observing the isna() command outputs

print(test_data[-206:])

#As expected the nans begin from 206th observation from the end.
# We will have to effectively shorten our testing period up to 2016-01-07...

test_data = test_data[:-206]
#We will scale the two separate data-sets. In order to do this we fit a min-max scaler on the train data
# and use the resulting scaler to normalize both train and test sets. This prevents forward looking bias.

scaler = MinMaxScaler()
scaler.fit(train_data)
train_scaled = pd.DataFrame(scaler.transform(train_data))
test_scaled = pd.DataFrame(scaler.transform(test_data))

print(dat_cols) #to locate the us interest rate column

#We would like to regress the US Interest Rate data upon our other features.

# To do this we must modify and shape our data accordingly...
# We will be using moving 12 observations of feature data to help predict the US interest rate at every point in time..

train_X, train_Y = [], []
for i in range(len(train_scaled)):
    end_index = i + 12
    if (end_index >= len(train_scaled)):
        break;
    train_X.append(np.array(train_scaled.iloc[i:end_index, 1:]))  # since the us interest rate is column 0
    train_Y.append(np.array(train_scaled.iloc[end_index, 0]))
train_X = np.array(train_X)
train_Y = np.array(train_Y)

test_X, test_Y = [], []
for i in range(len(test_scaled)):
    end_index = i + 12
    if (end_index >= len(test_scaled)):
        break;
    test_X.append(np.array(test_scaled.iloc[i:end_index, 1:]))
    test_Y.append(np.array(test_scaled.iloc[end_index, 0]))
test_X = np.array(test_X)
test_Y = np.array(test_Y)

print("train_X Shape: {}".format(train_X.shape))
print("train_Y Shape: {}".format(train_Y.shape))
print("test_X Shape: {}".format(test_X.shape))
print("test_Y Shape: {}".format(test_Y.shape))

#We build and compile an LSTM model
#Afterall, the data we are working with is clearly  a time series;
# it'd be best to make use of an LSTM to capture the temporal relationships between the input features and the US IR.
# This was essentially the reasoning for the modifications made to the training and testing sets in the previous cells.

# build model
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True)) #12 previous observations with 11 model features
model.add(LSTM(64, activation='tanh'))
model.add(Dense(1))

#compile model
model.compile(loss='mse', metrics=['mse','mape','mae'], optimizer='adam')

#We train the regressor
history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=50, verbose=0)

#We plot the behaviour of the training versus validation loss from epoch to epoch
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#model.save(r"LSTM_US_IR.h5")

#we predict the values given our test input
y_pred = model.predict(test_X)

#We must revert these prediction values to their form PRIOR to scaling

print(y_pred.shape, test_X.shape[0])
#we must create an array with shape 249 by 11
rows, cols = 249, 11
arr = [[0 for i in range(cols)] for j in range(rows)]
arr = np.array(arr)
print(arr.shape)

pred_values = np.concatenate((y_pred,arr), axis=1)

prediction = pd.DataFrame(scaler.inverse_transform(pred_values)).iloc[:,0] # we only want the first column as it corresponds the US. IR
prediction = np.array(prediction)
print(prediction.shape)

#We also need to revert our actual test values to their pre-scaling figures
test_Y = test_Y.reshape([len(test_Y),1])
actual_values = np.concatenate((test_Y,arr), axis=1)

actual = pd.DataFrame(scaler.inverse_transform(actual_values)).iloc[:,0]
actual = np.array(actual)
print(actual.shape)

#we compute the pearson correlation between the unscaled values (US IR predictions versus US IR actual)
pearsonr(prediction, actual)

fig , ax = plt.subplots()
ax.plot(prediction,label='Predicted')
ax.plot(actual,label='Actual')
fig.legend()
plt.grid()
fig.suptitle('Actual Verus Predicted Interested Rate')