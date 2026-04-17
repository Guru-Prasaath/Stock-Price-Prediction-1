"""
utils.py — Data Fetching & Preprocessing Utilities
====================================================
This module handles all data-related operations:
1. Fetching stock data from Yahoo Finance (yfinance)
2. Preprocessing (cleaning, normalization)
3. Creating time-series sequences for LSTM input
4. Inverse transforming predictions back to original scale

Written for beginners with detailed comments explaining each step.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# 1. FETCH STOCK DATA
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.
    
    HOW IT WORKS:
    - yfinance connects to Yahoo Finance API
    - Downloads OHLCV data (Open, High, Low, Close, Volume)
    - @st.cache_data caches the result for 1 hour (3600 seconds)
      so we don't re-download data on every page refresh
    
    Parameters:
        ticker (str): Stock symbol, e.g., 'AAPL' for Apple
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Stock data with columns [Open, High, Low, Close, Volume]
        None: If ticker is invalid or no data found
    """
    try:
        # Download data using yfinance
        # auto_adjust=True adjusts prices for stock splits and dividends
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Check if we got any data back
        if data.empty:
            return None
        
        # If columns are MultiIndex (happens with some yfinance versions), flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data
    
    except Exception as e:
        # If anything goes wrong (invalid ticker, network error, etc.)
        st.error(f"Error fetching data: {str(e)}")
        return None


def validate_ticker(ticker: str) -> bool:
    """
    Check if a stock ticker symbol is valid.
    
    HOW IT WORKS:
    - Tries to get info for the ticker from Yahoo Finance
    - If the ticker has a valid market price, it's real
    - Otherwise, it's invalid
    
    Parameters:
        ticker (str): Stock symbol to validate
    
    Returns:
        bool: True if valid, False if invalid
    """
    try:
        stock = yf.Ticker(ticker)
        # Try to get the stock info - if ticker is invalid, this will fail
        info = stock.info
        # Check if we got meaningful data back
        return info.get('regularMarketPrice') is not None or info.get('currentPrice') is not None
    except Exception:
        return False


def get_current_price(ticker: str) -> dict:
    """
    Get the current/latest price and basic info for a stock.
    
    Returns a dictionary with:
    - current_price: Latest trading price
    - previous_close: Yesterday's closing price  
    - company_name: Full company name
    - market_cap: Market capitalization
    
    Parameters:
        ticker (str): Stock symbol
    
    Returns:
        dict: Stock information dictionary
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'company_name': info.get('longName', ticker),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
        }
    except Exception:
        return {
            'current_price': 0,
            'previous_close': 0,
            'company_name': ticker,
            'market_cap': 0,
            'currency': 'USD',
            'exchange': 'N/A',
            'sector': 'N/A',
            'fifty_two_week_high': 0,
            'fifty_two_week_low': 0,
        }


# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Clean and normalize the stock data for LSTM input.
    
    STEP-BY-STEP EXPLANATION:
    
    Step 1: Extract 'Close' price
        - We only need the closing price for prediction
        - Close price = the last price the stock traded at each day
    
    Step 2: Handle missing values
        - Some days might have missing data (holidays, errors)
        - forward fill (ffill): fills missing value with the previous day's value
        - backward fill (bfill): fills any remaining gaps at the start
        - dropna: removes any still-remaining NaN values
    
    Step 3: Normalize data using MinMaxScaler
        - LSTM networks work better with data between 0 and 1
        - MinMaxScaler transforms: scaled_value = (value - min) / (max - min)
        - Example: If prices range from $100 to $200:
          $100 → 0.0, $150 → 0.5, $200 → 1.0
    
    Parameters:
        df (pd.DataFrame): Raw stock data from yfinance
    
    Returns:
        tuple: (scaled_data, scaler, close_prices)
            - scaled_data: numpy array of normalized prices (between 0 and 1)
            - scaler: the MinMaxScaler object (needed to convert predictions back)
            - close_prices: original close prices as a numpy array
    """
    # Step 1: Extract the 'Close' price column
    close_prices = df['Close'].values.reshape(-1, 1)
    
    # Step 2: Handle missing values
    # Convert to DataFrame temporarily for easy filling
    close_df = pd.DataFrame(close_prices, columns=['Close'])
    close_df = close_df.ffill()   # Forward fill: use previous value
    close_df = close_df.bfill()   # Backward fill: fill any gaps at the start
    close_df = close_df.dropna()  # Drop any remaining NaN (safety check)
    close_prices = close_df.values
    
    # Step 3: Normalize using MinMaxScaler (scale to range 0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    return scaled_data, scaler, close_prices


# ─────────────────────────────────────────────
# 3. SEQUENCE CREATION
# ─────────────────────────────────────────────

def create_sequences(data: np.ndarray, sequence_length: int = 60) -> tuple:
    """
    Create input sequences for LSTM model.
    
    HOW TIME-SERIES SEQUENCES WORK:
    ================================
    
    LSTM models learn patterns from sequences of past data points.
    We use a "sliding window" approach:
    
    Imagine stock prices for 10 days: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    With sequence_length=5, we create:
    
    Sequence 1: [10, 11, 12, 13, 14] → Predict: 15
    Sequence 2: [11, 12, 13, 14, 15] → Predict: 16
    Sequence 3: [12, 13, 14, 15, 16] → Predict: 17
    ...and so on
    
    The model learns: "Given the last 60 days of prices, what's the next day's price?"
    
    We use 60 timesteps because:
    - It captures ~3 months of trading data (approx. 60 trading days)
    - This gives the model enough historical context
    - It's a commonly used value in financial time-series prediction
    
    Parameters:
        data (np.ndarray): Normalized price data (values between 0 and 1)
        sequence_length (int): Number of past days to use (default: 60)
    
    Returns:
        tuple: (X, y)
            - X: Input sequences, shape (num_samples, sequence_length, 1)
            - y: Target values (the next day's price for each sequence)
    """
    X, y = [], []
    
    # Slide the window across the data
    for i in range(sequence_length, len(data)):
        # X: take 'sequence_length' values ending at position i
        X.append(data[i - sequence_length:i, 0])
        # y: the value at position i (what we want to predict)
        y.append(data[i, 0])
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to 3D for LSTM: (samples, timesteps, features)
    # LSTM expects input shape: (batch_size, sequence_length, num_features)
    # We have 1 feature (Close price), so the last dimension is 1
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> tuple:
    """
    Split data into training and testing sets.
    
    WHY WE SPLIT:
    - Training set: The model learns patterns from this data
    - Testing set: We check if the model can predict unseen data correctly
    - We use 80% for training and 20% for testing (common split)
    
    IMPORTANT: We do NOT shuffle time-series data!
    Unlike regular ML, the order of data points matters here.
    Future data should never leak into training data.
    
    Parameters:
        X (np.ndarray): Input sequences
        y (np.ndarray): Target values
        train_ratio (float): Fraction of data for training (default: 0.8 = 80%)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Calculate the split point
    split_index = int(len(X) * train_ratio)
    
    # Split without shuffling (order matters in time-series!)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test


def inverse_transform(scaler: MinMaxScaler, data: np.ndarray) -> np.ndarray:
    """
    Convert normalized predictions back to original price scale.
    
    WHY THIS IS NEEDED:
    - Our model outputs values between 0 and 1 (because we normalized)
    - To show actual dollar amounts, we need to "un-normalize" them
    - The scaler remembers the original min/max values and reverses the formula:
      original_value = scaled_value * (max - min) + min
    
    Parameters:
        scaler (MinMaxScaler): The same scaler used for normalization
        data (np.ndarray): Normalized predictions from the model
    
    Returns:
        np.ndarray: Predictions in original price scale (actual dollar amounts)
    """
    # Reshape to 2D if needed (scaler expects 2D input)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    return scaler.inverse_transform(data)


def format_large_number(num: float) -> str:
    """
    Format large numbers for display (e.g., 2500000000 → '$2.50B').
    
    Parameters:
        num (float): Number to format
    
    Returns:
        str: Formatted string
    """
    if num >= 1e12:
        return f"${num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"${num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"${num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"${num / 1e3:.2f}K"
    else:
        return f"${num:.2f}"
