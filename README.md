# Stock Price Prediction with LSTM

A professional-grade LSTM neural network implementation for stock price prediction using historical market data from Alpha Vantage API.

## Features

- Real-time stock data fetching via Alpha Vantage API
- Advanced data preprocessing with sliding window normalization
- LSTM-based deep learning model for time series prediction
- Exponential smoothing for noise reduction
- Interactive visualization of predictions

## Prerequisites

- Python
- TensorFlow
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy tensorflow matplotlib scikit-learn
   ```
3. Get your Alpha Vantage API key from [Alpha Vantage](https://www.alphavantage.co/)
4. Replace `ALPHA_VANTAGE_API_KEY` in `main.py` with your API key

## Usage

Run the main script:
```bash
python main.py
```

When prompted, enter the stock symbol (e.g., 'AAPL' for Apple).

## Technical Notes

- Uses sliding window normalization to prevent data leakage
- Implements EMA smoothing for noise reduction
- LSTM architecture optimized for stock prediction
- Early stopping implemented to prevent overfitting
- Handles API rate limiting considerations

## Performance Considerations

- API has a limit of 5 calls per minute (free tier)
- Caches downloaded data to CSV for reuse
- Optimized for moderate dataset sizes
- Single-threaded execution by default