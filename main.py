import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datatime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# Load Data
comapny = 'FB'

#Data Sets
start = dt.datatime(2012,1,1)
end = dt.datatime(2024,4,1)

data = web.DataReader(company, 'yahoo', start, end)  #Get Data from Yaho Finance

#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60 #Number of Days To predict

x_train = []
y_train = []


for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print('Training data length : ',len(x_train))

#build the modle
model = Sequential()

model.add(LTSM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LTSM(units=50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LTSM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1)) #Prediction of the next closing value

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size =32)

print('Model Built Successfully!')

''' Test the Model Accuracy on Existing Data'''

test_start = dt.datetime(2024,4,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis = 0)


model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].value
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transfor,(model_inputs)

#make Predictions on Test Data

x_test = []

for x in range (prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediciton_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


#plot the test predictions
plt.plot(actual_prices, colour="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, colour 'green', label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()