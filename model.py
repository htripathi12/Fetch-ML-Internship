import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
data = pd.read_csv('data/data_daily.csv')
data['# Date'] = pd.to_datetime(data['# Date'], format='%Y-%m-%d')
data.set_index('# Date', inplace=True)
monthly_data = data.resample('M').sum()  # Aggregate daily data to monthly
receipt_counts = monthly_data['Receipt_Count']

# Manually scale data to range [0, 1]
min_val = receipt_counts.min()
max_val = receipt_counts.max()
receipt_counts_scaled = (receipt_counts - min_val) / (max_val - min_val)

# Prepare sequences for LSTM
seq_length = 3  # Use past 3 months to predict the next month
X = []
y = []
for i in range(seq_length, len(receipt_counts_scaled)):
    X.append(receipt_counts_scaled.iloc[i - seq_length:i].values)
    y.append(receipt_counts_scaled.iloc[i])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], seq_length, 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

model.save('lstm_model.keras')