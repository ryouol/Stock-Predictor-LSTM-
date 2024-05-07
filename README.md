Data Fetching and Preprocessing: I use yfinance to download the stock prices, then scale the prices using MinMaxScaler to make them suitable for training the LSTM, which is sensitive to the scale of input data.
Data Preparation: I convert the time series data into sequences that the LSTM can use to learn (using the past n_steps data points to predict the next one).
Model Building: The LSTM network is constructed with dropout layers to prevent overfitting.
Training: The model is trained on the prepared data.
Prediction and Visualization: After training, the model predicts stock prices for the test data, and the predictions are plotted against the actual prices to visualize the model’s performance.![Figure_1](https://github.com/ryouol/Stock-Predictor-LSTM-/assets/125412884/ed797a06-0024-4381-bfc5-fc70860793df)
