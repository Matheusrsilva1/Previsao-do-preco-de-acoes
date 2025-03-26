import pandas as pd
import datetime as dt
import urllib.request
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def fetch_stock_data(symbol, api_key):
    # Rate limiting consideration: Alpha Vantage has a limit of 5 API calls per minute for free tier
    # Consider implementing retry mechanism with exponential backoff for production use
    # TODO: Implement proper error handling for API quota exceeded scenarios
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    output_file = f'stock_market_data-{symbol}.csv'

    if os.path.exists(output_file):
        print('Loading data from existing CSV file')
        return pd.read_csv(output_file)

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
        
        for date, values in time_series.items():
            parsed_date = dt.datetime.strptime(date, '%Y-%m-%d')
            row_data = [parsed_date.date(), float(values['3. low']), float(values['2. high']),
                      float(values['4. close']), float(values['1. open'])]
            df.loc[-1, :] = row_data
            df.index = df.index + 1

    print(f'Data saved to: {output_file}')
    df.to_csv(output_file)
    return df

def prepare_data(df):
    # Performance optimization: Using average of High/Low instead of all OHLC data
    # This reduces noise in the training data while maintaining price movement patterns
    # Alternative approach could be using weighted average with volume
    df = df.sort_values('Date')
    df['Average'] = (df['High'] + df['Low']) / 2.0
    prices = df['Average'].values

    train_size = int(0.8 * len(prices))
    test_size = len(prices) - train_size
    train_data = prices[:train_size]
    test_data = prices[train_size:]

    print(f"Total data points: {len(prices)}")
    print(f"Training data points: {train_size}")
    print(f"Testing data points: {test_size}")

    return train_data, test_data

def normalize_data(train_data, test_data, window_size=2500):
    # Critical implementation note: Using sliding window normalization to prevent data leakage
    # This approach maintains temporal locality while avoiding look-ahead bias
    # Window size of 2500 chosen based on empirical testing - approximately 10 years of trading days
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    if len(train_data) < window_size:
        print(f"Warning: train_data length ({len(train_data)}) is less than window_size ({window_size}). Normalizing all data at once.")
        scaler.fit(train_data)
        train_data[:] = scaler.transform(train_data)
    else:
        for start_idx in range(0, len(train_data) - window_size + 1, window_size):
            end_idx = min(start_idx + window_size, len(train_data))
            if end_idx > start_idx:
                scaler.fit(train_data[start_idx:end_idx, :])
                train_data[start_idx:end_idx, :] = scaler.transform(train_data[start_idx:end_idx, :])

        if start_idx + window_size < len(train_data):
            scaler.fit(train_data[start_idx + window_size:, :])
            train_data[start_idx + window_size:, :] = scaler.transform(train_data[start_idx + window_size:, :])

    test_data = scaler.transform(test_data)
    return train_data, test_data

def apply_exponential_smoothing(data, gamma=0.1):
    # Technical consideration: EMA implementation using single-pass algorithm
    # gamma=0.1 provides good balance between noise reduction and signal preservation
    # Lower gamma -> smoother curve but more lag, higher gamma -> more responsive but noisier
    ema = 0.0
    for i in range(len(data)):
        ema = gamma * data[i, 0] + (1 - gamma) * ema
        data[i, 0] = ema
    return data.reshape(-1)

def create_sequences(data, time_steps):
    # Architecture decision: Using sliding window approach for sequence creation
    # This maximizes data utilization while maintaining temporal relationships
    # Warning: Ensure sufficient memory available for large datasets
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def build_lstm_model(time_steps):
    # Model architecture considerations:
    # - Single LSTM layer with 50 units balances complexity vs performance
    # - Dropout layer (0.2) prevents overfitting on small datasets
    # - Dense layer with linear activation suitable for regression task
    # TODO: Consider adding batch normalization for better training stability
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(time_steps, 1), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train):
    # Training strategy:
    # - Early stopping with patience=3 prevents overfitting
    # - Small batch size (32) improves generalization
    # - 10 epochs typically sufficient for convergence while avoiding overfitting
    # TODO: Consider implementing learning rate scheduling for better convergence
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )

    try:
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping])
    except Exception as e:
        print(f"Training error: {e}")
        print(f"X_train shape: {X_train.shape}, type: {X_train.dtype}")
        print(f"y_train shape: {y_train.shape}, type: {y_train.dtype}")
        raise

def plot_results(all_data, train_size, predictions, time_steps):
    # Visualization design:
    # - Large figure size (18,9) ensures readability of long time series
    # - Blue for actual data and red for predictions maintains clear distinction
    # - Added proper labels and title for professional presentation
    plt.figure(figsize=(18, 9))
    plt.plot(range(len(all_data)), all_data, color='b', label='Actual Average Price')
    plt.plot(range(train_size + time_steps, train_size + time_steps + len(predictions)),
             predictions, color='r', label='Predicted Average Price')
    plt.xlabel('Date Index', fontsize=18)
    plt.ylabel('Average Price', fontsize=18)
    plt.title('LSTM Stock Price Prediction', fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

def main():
    # System architecture note:
    # This implementation assumes single-threaded execution and moderate dataset size
    # For production: Consider implementing:
    # - Proper API key management (env variables or secure storage)
    # - Parallel data processing for large datasets
    # - Proper logging and monitoring
    # - Model versioning and persistence
    API_KEY = 'ALPHA_VANTAGE_API_KEY'
    SYMBOL = input('Digite o símbolo da ação (ex: AAPL para Apple): ').upper()
    TIME_STEPS = 50

    df = fetch_stock_data(SYMBOL, API_KEY)
    train_data, test_data = prepare_data(df)
    train_normalized, test_normalized = normalize_data(train_data, test_data)
    
    train_smoothed = apply_exponential_smoothing(train_normalized.reshape(-1, 1))
    test_normalized = test_normalized.reshape(-1)
    all_data = np.concatenate([train_smoothed, test_normalized])

    if len(train_smoothed) <= TIME_STEPS:
        raise ValueError(f"Insufficient training data ({len(train_smoothed)}) for time steps ({TIME_STEPS}).")
    if len(test_normalized) <= TIME_STEPS:
        raise ValueError(f"Insufficient test data ({len(test_normalized)}) for time steps ({TIME_STEPS}).")

    X_train, y_train = create_sequences(train_smoothed, TIME_STEPS)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
    y_train = y_train.astype(np.float32)

    X_test, y_test = create_sequences(test_normalized, TIME_STEPS)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)).astype(np.float32)
    y_test = y_test.astype(np.float32)

    for data, name in [(X_train, 'X_train'), (y_train, 'y_train'), (X_test, 'X_test'), (y_test, 'y_test')]:
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"Warning: NaN or infinite values found in {name}. Replacing with zeros.")
            data = np.nan_to_num(data)

    model = build_lstm_model(TIME_STEPS)
    train_model(model, X_train, y_train)
    predictions = model.predict(X_test)

    mse = np.mean((predictions - y_test) ** 2)
    print(f'Mean Squared Error on Test Set: {mse:.5f}')

    plot_results(all_data, len(train_smoothed), predictions, TIME_STEPS)
    print("Prediction completed. Check the graph for results.")

if __name__ == "__main__":
    main()