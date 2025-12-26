import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load Data from Alpha Vantage
data_source = 'alphavantage'
api_key = 'AZ5UI81OKTCD9YSH'  # Replace with your own API key if needed
ticker = "AAL"
url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
file_to_save = f'stock_market_data-{ticker}.csv'

if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                        float(v['4. close']), float(v['1. open'])]
            df.loc[-1, :] = data_row
            df.index = df.index + 1
    print(f'Data saved to: {file_to_save}')
    df.to_csv(file_to_save)
else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)

# Sort DataFrame by date
df = df.sort_values('Date')

# Step 2: Data Preprocessing
# Calculate mid prices
df['Mid'] = (df['High'] + df['Low']) / 2.0
mid_prices = df['Mid'].values

# Determine train and test sizes
total_data_points = len(mid_prices)
train_size = int(0.8 * total_data_points)
test_size = total_data_points - train_size
train_data = mid_prices[:train_size]
test_data = mid_prices[train_size:]

print(f"Total data points: {total_data_points}")
print(f"Train data points: {train_size}")
print(f"Test data points: {test_size}")

# Normalize the data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

# Smooth training data with windowed normalization
smoothing_window_size = 2500
if train_size < smoothing_window_size:
    print(f"Warning: train_size ({train_size}) is less than smoothing_window_size ({smoothing_window_size}). Normalizing all data at once.")
    scaler.fit(train_data)
    train_data[:] = scaler.transform(train_data)
else:
    for di in range(0, train_size - smoothing_window_size + 1, smoothing_window_size):
        end_idx = min(di + smoothing_window_size, train_size)
        if end_idx > di:  # Ensure the slice is not empty
            scaler.fit(train_data[di:end_idx, :])
            train_data[di:end_idx, :] = scaler.transform(train_data[di:end_idx, :])

    # Normalize any remaining data
    if di + smoothing_window_size < train_size:
        scaler.fit(train_data[di + smoothing_window_size:, :])
        train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

# Normalize test data using the last scaler from training
test_data = scaler.transform(test_data)  # Mantém a forma (n,1)

# Apply Exponential Moving Average (EMA) to training data
EMA = 0.0
gamma = 0.1
for ti in range(train_size):
    EMA = gamma * train_data[ti, 0] + (1 - gamma) * EMA
    train_data[ti, 0] = EMA

# Reshape back to 1D arrays
train_data = train_data.reshape(-1)
test_data = test_data.reshape(-1)

# Concatenate for visualization
all_mid_data = np.concatenate([train_data, test_data])

# Step 3: Create Sequences for LSTM
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps])
    return np.array(X), np.array(y)

timesteps = 50
if len(train_data) <= timesteps:
    raise ValueError(f"Not enough training data ({len(train_data)}) for the given timesteps ({timesteps}).")
if len(test_data) <= timesteps:
    raise ValueError(f"Not enough test data ({len(test_data)}) for the given timesteps ({timesteps}).")

X_train, y_train = create_sequences(train_data, timesteps)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

X_test, y_test = create_sequences(test_data, timesteps)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 4: Build and Train LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(timesteps, 1), return_sequences=False),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Step 5: Make Predictions
predictions = model.predict(X_test)

# Calculate Mean Squared Error
mse = np.mean((predictions - y_test) ** 2)
print(f'Mean Squared Error on Test Set: {mse:.5f}')

# Step 6: Visualization
plt.figure(figsize=(18, 9))
plt.plot(range(len(all_mid_data)), all_mid_data, color='b', label='True Mid Price')
plt.plot(range(train_size + timesteps, train_size + timesteps + len(predictions)), predictions, color='r', label='Predicted Mid Price')
plt.xlabel('Date Index', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.title('Stock Price Prediction with LSTM', fontsize=18)
plt.legend(fontsize=18)
plt.show()

print("Prediction complete. Check the plot for results.")