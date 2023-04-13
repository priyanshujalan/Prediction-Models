import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

def predict(filepath, pred_col):
  
  #dataFrame = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2021-04-30')
  dataFrame = pd.read_excel(filepath)

  # New dataFrame with only 'Close Column'
  data = dataFrame.filter([pred_col])

  # Coverting the 'data' into numpy array
  dataSet = data.values

  #Get the number of rows to train the model on
  trainingDataLength = math.ceil(len(dataSet) * 0.8)

  #Scaling the Data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaledData = scaler.fit_transform(dataSet)

  # Create training data set
  # Create Scaled training Data set 
  trainData = scaledData[0:trainingDataLength, :]

  #Split the Data into x-train (Independent Variables) and y-train(Dependent Variables) data sets
  xTrain = []
  yTrain = []

  for i in range(60, len(trainData)):
    xTrain.append(trainData[i-60:i, 0])
    yTrain.append(trainData[i, 0])

  # Coverting x-train and y-train data sets to numpy array
  xTrain = np.array(xTrain)
  yTrain = np.array(yTrain) 

  #Reshaping the Data for LSTM Model, since the model requires 3d data set and we have 2d
  (m,n) = xTrain.shape
  xTrain = np.reshape(xTrain, (m,n,1))

  # Building the LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape = (n,1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  #Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')

  # We Train the Model
  model.fit(xTrain, yTrain, batch_size=1, epochs=1)

  #Create Testing DataSet
  #Creating Scaled values of the remaining 20% of the data set left

  testData = scaledData[trainingDataLength -60:, :]

  # Create x-test and y-test data sets
  xTest = []
  yTest = dataSet[trainingDataLength:, :]

  for i in range(60, len(testData)):
    xTest.append(testData[i-60:i, 0])

  # Convert the xTest into numpy array
  xTest = np.array(xTest)

  #Reshape the Data
  xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

  #Predicted Price values
  predictions = model.predict(xTest)
  predictions = scaler.inverse_transform(predictions)

  # Get the root mean squared error (RMSE)
  rmse=np.sqrt(np.mean(((predictions- yTest)**2)))
  print("RMSE :", rmse)

  # Plot the Data
  train = data[:trainingDataLength]
  valid = data[trainingDataLength:]
  valid['Predictions'] = predictions

  # Predicting One day ahead
  last60daysData = data[-60:].values
  last60daysDataScaled = scaler.transform(last60daysData)
  xTest = []
  xTest.append(last60daysDataScaled)
  xTest = np.array(xTest)
  xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
  tomorrowPrice = model.predict(xTest)
  tomorrowPrice = scaler.inverse_transform(tomorrowPrice)
  return tomorrowPrice[0], rmse


print(predict("C:\Learning\Python\StockMarket\GBBLData.xlsx", "Max"))
