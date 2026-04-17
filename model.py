"""
model.py — LSTM Model Architecture & Training
===============================================
This module handles the deep learning side of the project:
1. Building the LSTM neural network
2. Training the model on stock data
3. Making predictions

Written for beginners with detailed architecture explanations.

WHAT IS LSTM?
=============
LSTM = Long Short-Term Memory

It's a special type of neural network designed for SEQUENTIAL data
(data where order matters, like stock prices over time).

Regular neural networks can't "remember" past information.
LSTMs have a special "memory cell" that can:
- Remember important information from the past
- Forget irrelevant information
- Decide what new information to store

This makes them perfect for predicting stock prices, because
today's price depends on patterns from previous days.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ─────────────────────────────────────────────
# 1. BUILD LSTM MODEL
# ─────────────────────────────────────────────

def build_model(sequence_length: int = 60) -> Sequential:
    """
    Build and compile the LSTM neural network.
    
    ARCHITECTURE EXPLANATION:
    ========================
    
    Our model has this structure (like stacking building blocks):
    
    ┌─────────────────────────────────┐
    │   Input Layer                   │  ← Takes 60 days of price data
    │   Shape: (60 timesteps, 1)      │
    ├─────────────────────────────────┤
    │   LSTM Layer 1 (128 units)      │  ← Learns basic patterns
    │   return_sequences=True         │     (trends, simple movements)
    ├─────────────────────────────────┤
    │   Dropout (20%)                 │  ← Prevents overfitting
    ├─────────────────────────────────┤
    │   LSTM Layer 2 (64 units)       │  ← Learns complex patterns
    │   return_sequences=True         │     (combined trends)
    ├─────────────────────────────────┤
    │   Dropout (20%)                 │  ← Prevents overfitting
    ├─────────────────────────────────┤
    │   LSTM Layer 3 (32 units)       │  ← Learns high-level patterns
    │   return_sequences=False        │     (final representation)
    ├─────────────────────────────────┤
    │   Dropout (20%)                 │  ← Prevents overfitting
    ├─────────────────────────────────┤
    │   Dense Layer (16 neurons)      │  ← Processes learned patterns
    │   activation='relu'             │
    ├─────────────────────────────────┤
    │   Dense Layer (1 neuron)        │  ← Output: predicted price
    └─────────────────────────────────┘
    
    KEY CONCEPTS:
    
    - LSTM units: Think of these as "memory slots". More units = more patterns 
      the layer can remember. We decrease units (128→64→32) because each layer 
      creates a more compressed, abstract representation.
    
    - return_sequences=True: Passes the full sequence to the next LSTM layer.
      The last LSTM layer uses False because it only outputs one final value.
    
    - Dropout(0.2): Randomly "turns off" 20% of neurons during training.
      This prevents the model from memorizing training data (overfitting)
      and forces it to learn general patterns instead.
    
    - Dense(16) + Dense(1): Regular neural network layers that process 
      the LSTM output and produce the final predicted price.
    
    - Adam optimizer: An efficient algorithm that adjusts model weights 
      during training. It's like a smart "learner" that adapts its learning speed.
    
    - MSE loss: Mean Squared Error — measures how far predictions are from 
      actual values. The model tries to minimize this value during training.
    
    Parameters:
        sequence_length (int): Number of past days used (default: 60)
    
    Returns:
        Sequential: Compiled Keras model ready for training
    """
    model = Sequential([
        # Input specification — tells the model what shape of data to expect
        Input(shape=(sequence_length, 1)),
        
        # LSTM Layer 1: 128 memory units
        # return_sequences=True → output goes to next LSTM layer
        LSTM(units=128, return_sequences=True),
        
        # Dropout: randomly disable 20% of neurons to prevent overfitting
        Dropout(0.2),
        
        # LSTM Layer 2: 64 memory units (fewer = more compressed representation)
        LSTM(units=64, return_sequences=True),
        
        # Another dropout layer
        Dropout(0.2),
        
        # LSTM Layer 3: 32 memory units (final LSTM layer)
        # return_sequences=False → only output the last timestep's value
        LSTM(units=32, return_sequences=False),
        
        # Final dropout
        Dropout(0.2),
        
        # Dense layer: regular neural network layer with 16 neurons
        # ReLU activation: outputs the value if positive, 0 if negative
        Dense(units=16, activation='relu'),
        
        # Output layer: single neuron that outputs the predicted price
        Dense(units=1)
    ])
    
    # Compile the model — configure how it learns
    # - optimizer='adam': Smart weight-adjustment algorithm
    # - loss='mse': Mean Squared Error — what the model tries to minimize
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    return model


# ─────────────────────────────────────────────
# 2. TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(model, X_train: np.ndarray, y_train: np.ndarray,
                epochs: int = 50, batch_size: int = 32) -> dict:
    """
    Train the LSTM model on the stock data.
    
    HOW TRAINING WORKS:
    ===================
    
    Training is like studying for an exam:
    
    1. The model sees the training data (X_train) and makes predictions
    2. It calculates how wrong its predictions were (loss)
    3. It adjusts its internal weights to reduce the error
    4. Repeat for many "epochs" (complete passes through all data)
    
    KEY TERMS:
    
    - Epoch: One complete pass through ALL training data
      Like reading a textbook cover-to-cover once
    
    - Batch size: Number of samples processed at once
      Instead of learning from all data at once (too much),
      the model learns from small batches (like studying one chapter at a time)
    
    - Validation split: 10% of training data is set aside
      to check if the model is actually learning (not just memorizing)
    
    - EarlyStopping: Automatically stops training if the model
      stops improving — saves time and prevents overfitting
    
    Parameters:
        model: The compiled LSTM model
        X_train (np.ndarray): Training input sequences
        y_train (np.ndarray): Training target values
        epochs (int): Number of training passes (default: 50)
        batch_size (int): Samples per training step (default: 32)
    
    Returns:
        dict: Training history containing loss values per epoch
    """
    # Early stopping: stop training if no improvement for 5 consecutive epochs
    # 'patience=5' means: wait 5 epochs before giving up
    # 'restore_best_weights=True' means: use the model from the best epoch
    early_stopping = EarlyStopping(
        monitor='val_loss',       # Watch the validation loss
        patience=5,               # Wait 5 epochs for improvement
        restore_best_weights=True # Keep the best model weights
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,      # Use 10% of training data for validation
        callbacks=[early_stopping], # Stop early if not improving
        verbose=0,                  # Suppress output (we use Streamlit progress)
        shuffle=False               # Don't shuffle time-series data!
    )
    
    return history.history


# ─────────────────────────────────────────────
# 3. MAKE PREDICTIONS
# ─────────────────────────────────────────────

def predict_prices(model, X_test: np.ndarray) -> np.ndarray:
    """
    Use the trained model to predict stock prices.
    
    HOW PREDICTION WORKS:
    =====================
    
    After training, the model has learned patterns in stock price movements.
    
    We give it a sequence of 60 days of prices (X_test), and it outputs
    what it thinks the NEXT day's price will be.
    
    Note: These predictions are still in normalized form (0 to 1).
    We'll need to use inverse_transform() from utils.py to convert
    them back to actual dollar amounts.
    
    Parameters:
        model: The trained LSTM model
        X_test (np.ndarray): Test input sequences
    
    Returns:
        np.ndarray: Predicted prices (still normalized between 0 and 1)
    """
    predictions = model.predict(X_test, verbose=0)
    return predictions


# ─────────────────────────────────────────────
# 4. EVALUATION METRICS
# ─────────────────────────────────────────────

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate evaluation metrics to measure prediction accuracy.
    
    METRICS EXPLAINED:
    ==================
    
    1. RMSE (Root Mean Squared Error):
       - Measures average prediction error in the SAME UNITS as stock price
       - Formula: sqrt(mean((actual - predicted)²))
       - Example: RMSE of $5.23 means predictions are off by ~$5.23 on average
       - LOWER is BETTER
       - Penalizes large errors more than small ones
    
    2. MAE (Mean Absolute Error):
       - Average absolute difference between actual and predicted
       - Formula: mean(|actual - predicted|)
       - Example: MAE of $3.50 means predictions are off by ~$3.50 on average
       - LOWER is BETTER
       - Treats all errors equally
    
    3. MAPE (Mean Absolute Percentage Error):
       - Error as a PERCENTAGE of actual values
       - Formula: mean(|actual - predicted| / actual) × 100
       - Example: MAPE of 2.5% means predictions are ~2.5% off on average
       - LOWER is BETTER
       - Easy to interpret regardless of stock price
    
    Parameters:
        actual (np.ndarray): Real stock prices
        predicted (np.ndarray): Model's predicted prices
    
    Returns:
        dict: Dictionary with RMSE, MAE, and MAPE values
    """
    # Flatten arrays to 1D for calculation
    actual = actual.flatten()
    predicted = predicted.flatten()
    
    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # MAPE: Mean Absolute Percentage Error
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE': round(mape, 2)
    }
