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