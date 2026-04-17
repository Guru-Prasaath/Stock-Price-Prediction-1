# 📈 StockSense AI — LSTM-Based Stock Price Prediction

> A modern, end-to-end deep learning project that predicts stock prices using LSTM neural networks with a premium Streamlit dashboard inspired by Apple Stocks and TradingView.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

1. [Introduction](#-introduction)
2. [Objective](#-objective)
3. [Project Structure](#-project-structure)
4. [Setup Instructions](#-setup-instructions)
5. [How to Run](#-how-to-run)
6. [Methodology](#-methodology)
7. [Block Diagram](#-block-diagram)
8. [Model Architecture](#-model-architecture)
9. [Results & Evaluation](#-results--evaluation)
10. [Features](#-features)
11. [Expected Output](#-expected-output)
12. [Viva Questions & Answers](#-viva-questions--answers)
13. [Conclusion](#-conclusion)

---

## 📘 Introduction

Stock price prediction is one of the most challenging and fascinating problems in the intersection of finance and machine learning. Traditional approaches rely on fundamental analysis (studying company financials) and technical analysis (studying chart patterns). However, modern deep learning methods can automatically discover complex patterns in historical price data.

**Long Short-Term Memory (LSTM)** networks are a special kind of recurrent neural network (RNN) that are excellent at learning from sequences of data — making them ideal for time-series prediction like stock prices.

This project demonstrates how to:
- Collect real-time stock data using Yahoo Finance
- Preprocess and normalize data for neural network training
- Build and train a multi-layer LSTM model
- Generate price predictions and evaluate accuracy
- Present everything in a beautiful, interactive web dashboard

---

## 🎯 Objective

Build a complete end-to-end system that:

1. **Fetches** real-time stock data dynamically (no manual CSV downloads)
2. **Preprocesses** the data by handling missing values and normalizing prices
3. **Trains** a deep LSTM neural network to learn price patterns
4. **Predicts** future stock prices and evaluates accuracy
5. **Visualizes** results in a modern, professional dashboard

---

## 📁 Project Structure

```
Stock Price Prediction/
│
├── .streamlit/
│   └── config.toml          # Streamlit theme (dark mode colors)
│
├── assets/
│   └── style.css             # Custom CSS (glassmorphism, gradients, animations)
│
├── app.py                    # 🎨 Main Streamlit web application
│                              #    - Sidebar with stock configuration
│                              #    - Dashboard with metric cards & charts
│                              #    - Raw Data viewer with download
│                              #    - Model Info page with architecture
│                              #    - About page with documentation
│
├── model.py                  # 🧠 LSTM model architecture & training
│                              #    - build_model(): 3-layer LSTM + Dropout
│                              #    - train_model(): Training with early stopping
│                              #    - predict_prices(): Generate predictions
│                              #    - calculate_metrics(): RMSE, MAE, MAPE
│
├── utils.py                  # 🔧 Data fetching & preprocessing utilities
│                              #    - fetch_stock_data(): yfinance download
│                              #    - preprocess_data(): MinMaxScaler normalization
│                              #    - create_sequences(): Sliding window (60 days)
│                              #    - split_data(): 80/20 train/test split
│
├── requirements.txt          # 📦 Python dependencies
│
└── README.md                 # 📖 This documentation file
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** (Python package manager)
- Internet connection (for downloading stock data)

### Step-by-Step Setup

**Step 1: Open Terminal**

On Mac, press `Cmd + Space`, type "Terminal", and hit Enter.

**Step 2: Navigate to the project folder**

```bash
cd "/Users/user5/Documents/Stock Price Prediction"
```

**Step 3: Create a virtual environment (recommended)**

A virtual environment keeps project dependencies isolated from your system Python.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `yfinance` | Yahoo Finance stock data API |
| `tensorflow` | Deep learning framework (LSTM) |
| `scikit-learn` | Data preprocessing (MinMaxScaler) |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |

> **Note:** TensorFlow installation may take a few minutes. On Apple Silicon Macs, it will automatically use Metal GPU acceleration.

---

## ▶️ How to Run

**Step 1: Make sure your virtual environment is activated**

```bash
source venv/bin/activate
```

**Step 2: Run the Streamlit app**

```bash
streamlit run app.py
```

**Step 3: Open in browser**

Streamlit will automatically open your browser to `http://localhost:8501`

If it doesn't, manually open that URL in your browser.

**Step 4: Use the dashboard**

1. Enter a stock ticker (e.g., `AAPL`, `TSLA`, `GOOGL`, `MSFT`)
2. Select a date range
3. Adjust model settings (epochs, batch size, sequence length)
4. Click **"🚀 Train Model & Predict"**
5. Wait for training to complete (1-3 minutes)
6. Explore the dashboard, charts, and data!

---

## 🔬 Methodology

### 1. Data Collection

Data is fetched from **Yahoo Finance** using the `yfinance` library. This includes:
- **Open**: Price at market open
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Price at market close (this is what we predict)
- **Volume**: Number of shares traded

### 2. Data Preprocessing

- **Missing Values**: Handled with forward fill (use previous day's value) and backward fill
- **Feature Selection**: Only the 'Close' price is used as our prediction target
- **Normalization**: MinMaxScaler transforms prices to [0, 1] range
  - Formula: `scaled = (value - min) / (max - min)`
  - Neural networks learn faster and more accurately with normalized data

### 3. Sequence Creation (Sliding Window)

We create sequences of 60 consecutive days as input, with the 61st day as the target:

```
Day 1-60   → Predict Day 61
Day 2-61   → Predict Day 62
Day 3-62   → Predict Day 63
...
```

This teaches the model: *"Given the last 60 days of prices, what will tomorrow's price be?"*

### 4. Train/Test Split

- **80% Training**: Model learns patterns from this data
- **20% Testing**: Model is evaluated on data it has never seen
- **No shuffling**: Order is preserved (critical for time-series data!)

### 5. LSTM Model Training

The model trains for multiple epochs (passes through the data), adjusting its weights to minimize the Mean Squared Error (MSE) loss function. Early stopping prevents overfitting.

### 6. Prediction & Evaluation

Predictions are inverse-transformed back to dollar values and compared against actual prices using RMSE, MAE, and MAPE metrics.

---

## 🔄 Block Diagram

```
┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│  📥 DATA     │     │  🔧 PREPROCESS  │     │  📊 SEQUENCES  │
│  COLLECTION  │ ──→ │                 │ ──→ │                │
│              │     │ • Handle NaN    │     │ • 60-day       │
│ • yfinance   │     │ • Extract Close │     │   sliding      │
│ • OHLCV data │     │ • MinMaxScaler  │     │   window       │
│ • Any ticker │     │   (0 to 1)      │     │ • X → y pairs  │
└──────────────┘     └─────────────────┘     └────────┬───────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│ 📈 VISUALIZE │     │  🎯 PREDICT     │     │  🧠 LSTM       │
│              │ ←── │                 │ ←── │  MODEL         │
│ • Plotly     │     │ • Next-day price│     │                │
│ • Actual vs  │     │ • Inverse scale │     │ • 3 LSTM layers│
│   Predicted  │     │ • Compare with  │     │ • Dropout      │
│ • Metrics    │     │   actual prices │     │ • Dense output │
│ • Dashboard  │     │                 │     │ • Adam + MSE   │
└──────────────┘     └─────────────────┘     └────────────────┘
```

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────┐
│  📥 Input Layer                     │  Shape: (60, 1)
│  60 timesteps × 1 feature (Close)   │
├─────────────────────────────────────┤
│  🧠 LSTM Layer 1 — 128 units       │  Learns basic price patterns
│  return_sequences = True            │  (trends, simple movements)
├─────────────────────────────────────┤
│  ❌ Dropout — 20%                   │  Prevents overfitting
├─────────────────────────────────────┤
│  🧠 LSTM Layer 2 — 64 units        │  Learns complex patterns
│  return_sequences = True            │  (combined trends, volatility)
├─────────────────────────────────────┤
│  ❌ Dropout — 20%                   │  Prevents overfitting
├─────────────────────────────────────┤
│  🧠 LSTM Layer 3 — 32 units        │  Final abstraction
│  return_sequences = False           │  (high-level patterns)
├─────────────────────────────────────┤
│  ❌ Dropout — 20%                   │  Prevents overfitting
├─────────────────────────────────────┤
│  🔗 Dense Layer — 16 neurons       │  Processes patterns (ReLU)
├─────────────────────────────────────┤
│  📤 Output Layer — 1 neuron        │  Predicted next-day price
└─────────────────────────────────────┘
```

### Why This Architecture?

- **3 LSTM layers** with decreasing units (128→64→32) create a hierarchical feature extraction — each layer builds upon the previous one's learned patterns
- **Dropout (20%)** between layers prevents the model from memorizing training data
- **Dense layers** at the end process the temporal features into a single price prediction
- **Adam optimizer** adapts the learning rate automatically for efficient training
- **MSE loss** is the standard loss function for regression problems

---

## 📊 Results & Evaluation

### Metrics Explained

| Metric | Formula | What It Means | Good Value |
|---|---|---|---|
| **RMSE** | √(mean((actual - predicted)²)) | Average error in dollars; penalizes large errors more | Lower is better; < $10 for major stocks |
| **MAE** | mean(\|actual - predicted\|) | Average absolute error in dollars; treats all errors equally | Lower is better; < $5 for major stocks |
| **MAPE** | mean(\|actual - predicted\| / actual) × 100% | Error as percentage; easy to compare across stocks | < 5% is good; < 2% is excellent |

### Typical Results

For major stocks (AAPL, TSLA, GOOGL) with 5 years of data and 50 training epochs:
- MAPE typically ranges from **1-5%**
- The model captures the **general trend direction** effectively
- Predictions are more accurate during **stable periods** and less accurate during **high volatility**

---

## ✨ Features

### Dashboard UI
- 📊 **Interactive Plotly charts** with hover, zoom, and range selection
- 💳 **Glassmorphism metric cards** with gradient accent bars
- 🎨 **Dark mode** inspired by Apple Stocks
- 📱 **Responsive layout** that works on different screen sizes
- ⏳ **Animated loading indicators** during model training
- 🟢🔴 **Color-coded trends** (green = up, red = down)

### Technical Features  
- ⚡ **Data caching** with `@st.cache_data` for instant reloads
- 🛡️ **Error handling** for invalid tickers and network issues
- 🔧 **Configurable hyperparameters** (epochs, batch size, sequence length)
- 📥 **CSV download** option for raw data
- 🛑 **Early stopping** to prevent overfitting
- 📈 **Multiple chart types**: area, line, bar, overlay

---

## 🖥️ Expected Output

When you run the app, you'll see:

1. **Sidebar**: Stock ticker input, date range selector, model settings, train button
2. **Dashboard Page**:
   - Company name with current price and daily change
   - 4 metric cards (Price, Prediction, RMSE, Market Cap)
   - Interactive price history chart with time range buttons (1M, 3M, 6M, 1Y, ALL)
   - Actual vs Predicted overlay chart (after training)
   - Evaluation metrics cards (RMSE, MAE, MAPE)
   - Volume chart with green/red color coding
3. **Raw Data Page**: Sortable data table, statistics, missing values check, CSV download
4. **Model Info Page**: Architecture diagram, hyperparameters, training loss curve
5. **About Page**: Block diagram, methodology, results, FAQ

---

## ❓ Viva Questions & Answers

### 1. What is LSTM and why is it used for stock prediction?

**Answer:** LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) designed to handle sequential data and learn long-term dependencies. Unlike simple RNNs that suffer from the "vanishing gradient" problem (they forget information over long sequences), LSTMs have a special cell state that can **selectively remember or forget** information over many timesteps. This makes them ideal for stock prediction because stock prices depend on patterns from days, weeks, or even months ago.

### 2. What is the vanishing gradient problem? How does LSTM solve it?

**Answer:** In regular RNNs, during backpropagation, gradients (the signals used to update weights) become extremely small as they propagate backward through many timesteps. This means the network can't learn from distant past data. LSTMs solve this with three **gates** (forget, input, output) and a **cell state** — a direct path for information to flow across timesteps without being multiplied by weights at every step, preserving the gradient signal.

### 3. Why do we normalize data using MinMaxScaler?

**Answer:** Neural networks work best when input data is in a small, consistent range (0 to 1). Without normalization, stock prices (which can range from $1 to $1000+) would cause the network to learn very slowly or not at all. MinMaxScaler transforms data using: `scaled = (value - min) / (max - min)`. This preserves relative differences while constraining values to [0, 1].

### 4. What is the purpose of the Dropout layer?

**Answer:** Dropout is a **regularization technique** that randomly "drops" (deactivates) a percentage of neurons during each training step. With 20% dropout, 1 in 5 neurons is randomly turned off each iteration. This prevents **overfitting** — when the model memorizes training data instead of learning generalizable patterns. It forces the network to develop redundant representations, making it more robust.

### 5. Why don't we shuffle time-series data when splitting?

**Answer:** In time-series data, the **order of data points matters**. Shuffling would mix future data into the training set, causing "data leakage" — the model would learn from future information that wouldn't be available in real-world predictions. We must train on older data and test on newer data to simulate realistic conditions.

### 6. What is the difference between RMSE and MAE?

**Answer:** 
- **MAE (Mean Absolute Error)**: Average of |actual - predicted|. Treats all errors equally. If predictions are off by $3, $5, and $7, MAE = ($3+$5+$7)/3 = $5.
- **RMSE (Root Mean Squared Error)**: Square root of the average of (actual - predicted)². **Penalizes large errors more** because of the squaring. Using the same example: RMSE = √((9+25+49)/3) = √27.67 = $5.26. RMSE ≥ MAE always; the bigger the gap between them, the more variable the errors are.

### 7. What does return_sequences mean in LSTM?

**Answer:** `return_sequences=True` makes the LSTM layer output the hidden state at **every timestep** (a full sequence), not just the final one. This is necessary when stacking LSTM layers — the next LSTM layer needs the complete sequence as input. The **last** LSTM layer typically uses `return_sequences=False` because we only need one output value (the predicted price).

### 8. Why do we use 60 timesteps as the sequence length?

**Answer:** 60 trading days ≈ approximately 3 months of market data. This provides enough historical context for the model to learn meaningful patterns (short-term trends, weekly patterns, monthly cycles) without being too computationally expensive. The choice is empirical — researchers have found 30-90 days works well for daily stock data. Too few days = not enough context; too many = noise and slow training.

### 9. What is Early Stopping and why is it important?

**Answer:** Early stopping is a training strategy that **monitors validation loss** and stops training when it stops improving for a specified number of epochs (called "patience"). This is important because:
1. **Saves time**: No need to train for unnecessary epochs
2. **Prevents overfitting**: The model stops before it starts memorizing training data
3. **Optimal model**: `restore_best_weights=True` keeps the weights from the epoch with the lowest validation loss

### 10. Can LSTM accurately predict stock prices? What are the limitations?

**Answer:** LSTM can learn **trends and patterns** in historical prices, but it **cannot accurately predict exact prices** because:
1. Stock prices are influenced by **external events** (news, earnings, geopolitics) that aren't in price data
2. Markets are partially **random** (Efficient Market Hypothesis)
3. Past patterns may not repeat in the future
4. The model only uses closing prices — professional systems use hundreds of features
5. **No model can consistently beat the market** — this is why it's an educational project, not a trading system

### 11. What is the Adam optimizer?

**Answer:** Adam (Adaptive Moment Estimation) is an optimization algorithm that combines benefits of two other algorithms: AdaGrad (adapts learning rates per parameter) and RMSProp (uses moving average of squared gradients). It maintains **two running averages** for each weight: the mean and variance of gradients. This allows it to automatically adjust the learning rate for each parameter, making training faster and more stable than standard SGD.

### 12. How does MinMaxScaler inverse_transform work?

**Answer:** After prediction, outputs are in the scaled [0, 1] range. `inverse_transform` reverses the scaling using the original min/max values stored by the scaler: `original = scaled × (max - min) + min`. This converts predictions back to actual dollar amounts for meaningful interpretation.

### 13. What is batch size and how does it affect training?

**Answer:** Batch size is the number of training samples processed before updating model weights:
- **Small batch (16)**: More frequent weight updates, noisier but can escape local minima, slower training
- **Large batch (128)**: Less frequent updates, smoother but may miss subtle patterns, faster training
- **32** is a commonly used compromise that balances speed and accuracy

### 14. What would happen if we used only one LSTM layer?

**Answer:** A single LSTM layer can learn basic patterns but may miss complex, hierarchical relationships in the data. Multiple layers create a **feature hierarchy**: the first layer learns simple patterns (daily fluctuations), the second learns intermediate patterns (weekly trends), and the third learns high-level patterns (monthly movements). However, more layers also mean more training time and risk of overfitting — hence the use of Dropout between layers.

### 15. Why is Streamlit used instead of Flask or Django?

**Answer:** Streamlit is chosen because:
1. **Simplicity**: Build interactive web apps with pure Python — no HTML/JS required
2. **Speed**: Apps are created in minutes, not hours
3. **Data-centric**: Built specifically for data science and ML applications
4. **Widgets**: Built-in interactive components (sliders, inputs, charts)
5. **Auto-reload**: App updates automatically when code changes
6. Flask/Django are more powerful but require much more boilerplate code

---

## 📝 Conclusion

This project demonstrates a complete machine learning pipeline from data collection to deployment:

1. **Data Collection**: Successfully fetches real-time stock data from Yahoo Finance
2. **Preprocessing**: Normalizes data effectively using MinMaxScaler
3. **Model**: A 3-layer LSTM architecture that learns temporal patterns
4. **Prediction**: Generates reasonable price predictions with low error rates
5. **Visualization**: Professional, interactive dashboard for data exploration

The project serves as an excellent educational tool for understanding deep learning applied to financial data, while also showcasing modern web application development with Streamlit.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is NOT financial advice and should NOT be used for actual stock trading decisions. Stock markets are influenced by countless factors that no model can fully capture.

---

## 📄 License

MIT License — Free to use and modify for educational purposes.
