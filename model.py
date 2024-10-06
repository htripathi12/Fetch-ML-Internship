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
monthly_data = data.resample('M').sum() # Aggregate daily data to monthly
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

# Test train split (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

predictions = []
last_sequence = receipt_counts_scaled[-seq_length:].values.tolist()  # Last 3 months of data
for _ in range(12):
    X_pred = np.array(last_sequence[-seq_length:]).reshape(1, seq_length, 1)
    next_pred = model.predict(X_pred)
    next_pred_value = next_pred[0, 0]
    predictions.append(next_pred_value)
    last_sequence.append(next_pred_value)

# Undo scaling from [0, 1] to original range
predictions = np.array(predictions)
predictions = predictions * (max_val - min_val) + min_val

# Create dates for 2022
dates_2022 = pd.date_range(start='2022-01-01', periods=12, freq='M')
predictions_df = pd.DataFrame({'Date': dates_2022, 'Predicted_Receipt_Count': predictions})

print("\nPredicted receipt counts for each month of 2022:")
print(predictions_df)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(monthly_data.index, monthly_data['Receipt_Count'], label='Actual')
plt.plot(predictions_df['Date'], predictions_df['Predicted_Receipt_Count'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Receipt Count (Hundred Millions)')
plt.title('Predicted vs Actual Monthly Receipt Counts')
plt.legend()
plt.show()