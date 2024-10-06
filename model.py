import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('data/data_daily.csv')
data['# Date'] = pd.to_datetime(data['# Date'], format='%Y-%m-%d')
data.set_index('# Date', inplace=True)
receipt_counts = data['Receipt_Count']

# Manually scale data to range [0, 1]
min_val = receipt_counts.min()
max_val = receipt_counts.max()
receipt_counts_scaled = (receipt_counts - min_val) / (max_val - min_val)

# Prepare sequences for LSTM
seq_length = 90  # Use past 3 months to predict the next day
X = []
y = []
for i in range(seq_length, len(receipt_counts_scaled)):
    X.append(receipt_counts_scaled.iloc[i - seq_length:i].values)
    y.append(receipt_counts_scaled.iloc[i])
X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], seq_length, 1))

# Test train split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
y_pred_scaled = model.predict(X_test)

# Undo scaling from [0, 1] to original range
y_pred = y_pred_scaled * (max_val - min_val) + min_val
y_actual = y_test * (max_val - min_val) + min_val

# Mean Absolute Percentage Error
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
print(f"Mean Absolute Percentage Error on Test Set: {mape}%")

# Retrain with full dataset for future predictions
# Above code is ONLY for evaluation of model performance
model.fit(X, y, epochs=20, batch_size=32)

predictions = []
last_sequence = receipt_counts_scaled[-seq_length:].values.tolist() # Last 3 months of data
# Each time a prediction is made, it is appended to the last_sequence 
# and added to the next prediction sequence
for _ in range(365):
    X_pred = np.array(last_sequence[-seq_length:]).reshape(1, seq_length, 1)
    next_pred = model.predict(X_pred)
    next_pred_value = next_pred[0, 0]
    predictions.append(next_pred_value)
    last_sequence.append(next_pred_value)

# Undo scaling from [0, 1] to original range
predictions = np.array(predictions)
predictions = predictions * (max_val - min_val) + min_val

# Create dates for 2022
dates_2022 = pd.date_range(start='2022-01-01', periods=365, freq='D')
predictions_df = pd.DataFrame({'Date': dates_2022, 'Predicted_Receipt_Count': predictions})

# Aggregate daily predictions to get monthly totals
predictions_df['Month'] = predictions_df['Date'].dt.to_period('M')
monthly_predictions = predictions_df.groupby('Month')['Predicted_Receipt_Count'].sum()

print("\nPredicted receipt counts for each month of 2022:")
print(monthly_predictions)