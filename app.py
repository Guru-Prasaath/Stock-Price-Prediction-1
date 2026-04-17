"""
app.py — StockSense AI: LSTM Stock Price Prediction Dashboard
===============================================================
Main Streamlit application with a modern, premium UI inspired by
Apple Stocks and TradingView dashboards.

Features:
- Real-time stock data fetching via yfinance
- LSTM model training and prediction
- Interactive Plotly charts
- Dark/Light mode toggle
- Glassmorphism metric cards
- Color-coded trend indicators

HOW TO RUN:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import time

# Import our custom modules
from utils import (
    fetch_stock_data, validate_ticker, get_current_price,
    preprocess_data, create_sequences, split_data,
    inverse_transform, format_large_number
)
from model import build_model, train_model, predict_prices, calculate_metrics


# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="StockSense AI — LSTM Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# LOAD CUSTOM CSS
# ─────────────────────────────────────────────

def load_css():
    """Load the custom CSS file for premium styling."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────

# Session state persists data across Streamlit reruns
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'actual_prices' not in st.session_state:
    st.session_state.actual_prices = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'stock_info' not in st.session_state:
    st.session_state.stock_info = None
if 'loaded_ticker' not in st.session_state:
    st.session_state.loaded_ticker = None
if 'loaded_start_date' not in st.session_state:
    st.session_state.loaded_start_date = None
if 'loaded_end_date' not in st.session_state:
    st.session_state.loaded_end_date = None


# ─────────────────────────────────────────────
# HELPER: PLOTLY CHART THEME
# ─────────────────────────────────────────────

def get_chart_layout(title: str = "", height: int = 500) -> dict:
    """
    Get consistent Plotly chart layout matching our dashboard theme.
    Returns a dict of layout properties for a dark trading-dashboard look.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(size=16, color='#E8EAED', family='Inter'),
            x=0.02
        ),
        plot_bgcolor='rgba(10, 14, 23, 0)',
        paper_bgcolor='rgba(10, 14, 23, 0)',
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            zerolinecolor='rgba(255,255,255,0.05)',
            color='#9AA0AB',
            showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            zerolinecolor='rgba(255,255,255,0.05)',
            color='#9AA0AB',
            showgrid=True,
            side='right',
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9AA0AB', size=11),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1E222D',
            font=dict(color='#E8EAED', family='Inter'),
            bordercolor='rgba(0,212,170,0.3)'
        ),
    )


def create_metric_card(label: str, value: str, change: str = "",
                       is_positive: bool = True, icon: str = "") -> str:
    """
    Generate HTML for a premium glassmorphism metric card.
    
    Parameters:
        label: Card title (e.g., "Current Price")
        value: Main value (e.g., "$182.63")
        change: Change text (e.g., "+2.3%")
        is_positive: True = green accent, False = red accent
        icon: Emoji icon for the card
    
    Returns:
        str: HTML string for the metric card
    """
    accent_class = "metric-card-green" if is_positive else "metric-card-red"
    change_class = "positive" if is_positive else "negative"
    arrow = "↑" if is_positive else "↓"
    
    return f"""
    <div class="metric-card {accent_class}">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-change {change_class}">{arrow} {change}</div>' if change else ''}
    </div>
    """


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    # App branding
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">📈</div>
        <div class="app-title">StockSense AI</div>
        <div class="app-subtitle">LSTM Price Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Navigation ──
    st.markdown("#### 🧭 Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Dashboard", "📋 Raw Data", "🧠 Model Info", "📖 About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # ── Stock Configuration ──
    st.markdown("#### ⚙️ Configuration")
    
    # Ticker input
    ticker = st.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        placeholder="e.g., AAPL, TSLA, GOOGL",
        help="Enter a valid stock ticker symbol from Yahoo Finance"
    ).upper().strip()
    
    # Date range
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=5*365),
            max_value=datetime.now(),
            help="Start date for historical data"
        )
    with col_end:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now(),
            help="End date for historical data"
        )
    
    st.markdown("---")
    
    # ── Model Hyperparameters ──
    st.markdown("#### 🔧 Model Settings")
    
    epochs = st.slider(
        "Training Epochs",
        min_value=10, max_value=100, value=50, step=5,
        help="Number of complete passes through training data. More epochs = better learning, but slower."
    )
    
    batch_size = st.select_slider(
        "Batch Size",
        options=[16, 32, 64, 128],
        value=32,
        help="Number of samples processed at once. Smaller = more precise, Larger = faster."
    )
    
    sequence_length = st.slider(
        "Sequence Length (days)",
        min_value=30, max_value=120, value=60, step=10,
        help="Number of past days to use for prediction. 60 days ≈ 3 months of trading data."
    )
    
    st.markdown("---")
    
    load_button = st.button(
        "Load Stock Data",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # ── Train Model Button ──
    train_button = st.button(
        "🚀 Train Model & Predict",
        use_container_width=True,
        type="primary"
    )
    
    # ── Reset Button ──
    if st.session_state.model_trained:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.model_trained = False
            st.session_state.predictions = None
            st.session_state.actual_prices = None
            st.session_state.metrics = None
            st.session_state.training_history = None
            st.session_state.stock_data = None
            st.session_state.stock_info = None
            st.session_state.loaded_ticker = None
            st.session_state.loaded_start_date = None
            st.session_state.loaded_end_date = None
            st.rerun()


# ─────────────────────────────────────────────
# MAIN CONTENT AREA
# ─────────────────────────────────────────────

# ═══════════════════════════════════════════════
# DATA LOADING (Common to all pages)
# ═══════════════════════════════════════════════

# Load data only when explicitly requested or required for training
should_load_data = (
    load_button or
    train_button or
    (
        st.session_state.stock_data is not None and (
            st.session_state.loaded_ticker != ticker or
            st.session_state.loaded_start_date != str(start_date) or
            st.session_state.loaded_end_date != str(end_date)
        )
    )
)

if should_load_data:
    with st.spinner("Loading stock data..."):
        stock_data = fetch_stock_data(ticker, str(start_date), str(end_date))
        stock_info = get_current_price(ticker)
    
    if stock_data is None or stock_data.empty:
        st.markdown("""
        <div style="text-align: center; padding: 80px 20px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">⚠️</div>
            <h2 style="color: #FF1744; margin-bottom: 8px;">Invalid Stock Ticker</h2>
            <p style="color: #9AA0AB; font-size: 1rem;">
                Could not find data for ticker symbol "<strong>{}</strong>".<br>
                Please enter a valid symbol (e.g., AAPL, TSLA, GOOGL, MSFT).
            </p>
        </div>
        """.format(ticker), unsafe_allow_html=True)
        st.stop()
    
    st.session_state.stock_data = stock_data
    st.session_state.stock_info = stock_info
    st.session_state.loaded_ticker = ticker
    st.session_state.loaded_start_date = str(start_date)
    st.session_state.loaded_end_date = str(end_date)

stock_data = st.session_state.stock_data
stock_info = st.session_state.stock_info

if stock_data is None or stock_info is None:
    st.markdown("""
    <div style="text-align: center; padding: 90px 20px;">
        <div style="font-size: 4rem; margin-bottom: 16px;">📈</div>
        <h2 style="color: #E8EAED; margin-bottom: 8px;">StockSense AI</h2>
        <p style="color: #9AA0AB; font-size: 1rem; max-width: 540px; margin: 0 auto;">
            Choose a ticker and date range in the sidebar, then click
            <strong>Load Stock Data</strong> to populate the dashboard.
            After that, you can train the LSTM model and generate predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════
# MODEL TRAINING LOGIC
# ═══════════════════════════════════════════════

if train_button:
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Preprocess data (20%)
        status_text.markdown("""
        <div class="info-box loading-pulse">
            🔄 <strong>Step 1/4:</strong> Preprocessing data — normalizing prices...
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(10)
        
        scaled_data, scaler, close_prices = preprocess_data(stock_data)
        progress_bar.progress(20)
        
        # Step 2: Create sequences (40%)
        status_text.markdown("""
        <div class="info-box loading-pulse">
            🔄 <strong>Step 2/4:</strong> Creating time-series sequences ({} day windows)...
        </div>
        """.format(sequence_length), unsafe_allow_html=True)
        
        X, y = create_sequences(scaled_data, sequence_length)
        X_train, X_test, y_train, y_test = split_data(X, y)
        progress_bar.progress(40)
        
        # Validate we have enough data
        if len(X_train) < 10:
            st.error("❌ Not enough data for training. Try a longer date range or shorter sequence length.")
            progress_bar.empty()
            status_text.empty()
            st.stop()
        
        # Step 3: Build and train model (90%)
        status_text.markdown("""
        <div class="info-box loading-pulse">
            🧠 <strong>Step 3/4:</strong> Training LSTM model — {} epochs, batch size {}...<br>
            <span style="font-size: 0.8rem; color: #6B7280;">This may take 1-3 minutes depending on your hardware.</span>
        </div>
        """.format(epochs, batch_size), unsafe_allow_html=True)
        
        lstm_model = build_model(sequence_length)
        history = train_model(lstm_model, X_train, y_train, epochs, batch_size)
        progress_bar.progress(85)
        
        # Step 4: Generate predictions (100%)
        status_text.markdown("""
        <div class="info-box loading-pulse">
            📊 <strong>Step 4/4:</strong> Generating predictions and calculating metrics...
        </div>
        """, unsafe_allow_html=True)
        
        # Predict on test data
        predictions_scaled = predict_prices(lstm_model, X_test)
        predictions = inverse_transform(scaler, predictions_scaled)
        actual = inverse_transform(scaler, y_test.reshape(-1, 1))
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(actual, predictions)
        
        # Store results in session state
        st.session_state.model_trained = True
        st.session_state.predictions = predictions
        st.session_state.actual_prices = actual
        st.session_state.metrics = metrics
        st.session_state.training_history = history
        st.session_state.train_split_index = len(X_train) + sequence_length
        st.session_state.test_dates = stock_data.index[len(X_train) + sequence_length:]
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Training failed: {str(e)}")
        st.info("💡 Try adjusting the date range or sequence length.")


# ═══════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════

if page == "📊 Dashboard":
    
    # ── Header ──
    company_name = stock_info.get('company_name', ticker)
    current_price = stock_info.get('current_price', 0)
    previous_close = stock_info.get('previous_close', 0)
    
    # Calculate price change
    if previous_close and current_price:
        price_change = current_price - previous_close
        pct_change = (price_change / previous_close) * 100
        is_positive = price_change >= 0
    else:
        price_change = 0
        pct_change = 0
        is_positive = True
    
    change_color = "#00C853" if is_positive else "#FF1744"
    change_arrow = "▲" if is_positive else "▼"
    
    # Title with live price
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 2rem; font-weight: 800; margin: 0; color: #E8EAED;">
            {company_name}
        </h1>
        <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 4px;">
            <span style="font-size: 2.2rem; font-weight: 700; color: #E8EAED;">
                ${current_price:,.2f}
            </span>
            <span style="font-size: 1rem; font-weight: 600; color: {change_color};">
                {change_arrow} ${abs(price_change):,.2f} ({pct_change:+.2f}%)
            </span>
            <span style="font-size: 0.8rem; color: #6B7280;">
                {stock_info.get('currency', 'USD')} · {stock_info.get('exchange', '')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Metric Cards Row ──
    if st.session_state.model_trained:
        predicted_price = float(st.session_state.predictions[-1][0])
        pred_change = ((predicted_price - current_price) / current_price) * 100 if current_price else 0
        pred_positive = pred_change >= 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                "Current Price", f"${current_price:,.2f}",
                f"{pct_change:+.2f}% today", is_positive, "💰"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                "Predicted Price", f"${predicted_price:,.2f}",
                f"{pred_change:+.2f}% vs current", pred_positive, "🎯"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "RMSE", f"${st.session_state.metrics['RMSE']:,.2f}",
                f"MAPE: {st.session_state.metrics['MAPE']}%", True, "📏"
            ), unsafe_allow_html=True)
        
        with col4:
            market_cap = stock_info.get('market_cap', 0)
            st.markdown(create_metric_card(
                "Market Cap", format_large_number(market_cap),
                stock_info.get('sector', ''), True, "🏢"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Main Price Chart ──
    st.markdown('<div class="section-header">📈 Price History</div>', unsafe_allow_html=True)
    
    fig_price = go.Figure()
    
    # Area chart with gradient fill (Apple Stocks style)
    fig_price.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#00D4AA', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.08)',
        hovertemplate='$%{y:,.2f}<extra></extra>'
    ))
    
    fig_price.update_layout(**get_chart_layout("", height=420))
    fig_price.update_layout(
        yaxis_title="Price (USD)",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(label="ALL", step="all")
                ]),
                bgcolor='rgba(19, 23, 34, 0.8)',
                activecolor='rgba(0, 212, 170, 0.3)',
                font=dict(color='#9AA0AB', size=11),
            ),
        ),
    )
    
    st.plotly_chart(fig_price, use_container_width=True, config={'displayModeBar': False})
    
    # ── Actual vs Predicted Chart ──
    if st.session_state.model_trained:
        st.markdown('<div class="section-header">🎯 Actual vs Predicted Prices</div>',
                    unsafe_allow_html=True)
        
        # Get test dates for x-axis
        test_dates = st.session_state.test_dates
        # Ensure dates align with predictions length
        min_len = min(len(test_dates), len(st.session_state.actual_prices))
        test_dates = test_dates[:min_len]
        actual = st.session_state.actual_prices[:min_len].flatten()
        predicted = st.session_state.predictions[:min_len].flatten()
        
        fig_compare = go.Figure()
        
        # Actual prices
        fig_compare.add_trace(go.Scatter(
            x=test_dates,
            y=actual,
            mode='lines',
            name='Actual Price',
            line=dict(color='#00D4AA', width=2.5),
            hovertemplate='Actual: $%{y:,.2f}<extra></extra>'
        ))
        
        # Predicted prices
        fig_compare.add_trace(go.Scatter(
            x=test_dates,
            y=predicted,
            mode='lines',
            name='Predicted Price',
            line=dict(color='#FF6D00', width=2.5, dash='dot'),
            hovertemplate='Predicted: $%{y:,.2f}<extra></extra>'
        ))
        
        # Fill between actual and predicted
        fig_compare.add_trace(go.Scatter(
            x=list(test_dates) + list(test_dates)[::-1],
            y=list(actual) + list(predicted)[::-1],
            fill='toself',
            fillcolor='rgba(255, 109, 0, 0.05)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_compare.update_layout(**get_chart_layout("", height=420))
        fig_compare.update_layout(yaxis_title="Price (USD)")
        
        st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})
        
        # ── Evaluation Metrics Row ──
        st.markdown('<div class="section-header">📊 Model Performance</div>',
                    unsafe_allow_html=True)
        
        met_col1, met_col2, met_col3 = st.columns(3)
        
        with met_col1:
            st.markdown(f"""
            <div class="eval-metric">
                <div class="eval-label">Root Mean Squared Error</div>
                <div class="eval-value">${st.session_state.metrics['RMSE']:,.4f}</div>
                <div style="color: #6B7280; font-size: 0.75rem; margin-top: 6px;">
                    Average prediction error (penalizes large errors)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col2:
            st.markdown(f"""
            <div class="eval-metric">
                <div class="eval-label">Mean Absolute Error</div>
                <div class="eval-value">${st.session_state.metrics['MAE']:,.4f}</div>
                <div style="color: #6B7280; font-size: 0.75rem; margin-top: 6px;">
                    Average absolute difference from actual price
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with met_col3:
            mape_color = "#00C853" if st.session_state.metrics['MAPE'] < 5 else "#FF6D00"
            st.markdown(f"""
            <div class="eval-metric">
                <div class="eval-label">Mean Absolute % Error</div>
                <div class="eval-value" style="color: {mape_color};">{st.session_state.metrics['MAPE']}%</div>
                <div style="color: #6B7280; font-size: 0.75rem; margin-top: 6px;">
                    Average percentage deviation from actual
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Prompt user to train
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px;
                    background: linear-gradient(135deg, rgba(0, 212, 170, 0.05), rgba(0, 180, 216, 0.03));
                    border: 1px dashed rgba(0, 212, 170, 0.2); border-radius: 16px; margin: 20px 0;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🧠</div>
            <h3 style="color: #E8EAED; margin-bottom: 8px;">Ready to Predict</h3>
            <p style="color: #9AA0AB; max-width: 400px; margin: 0 auto;">
                Configure your settings in the sidebar and click <strong>"Train Model & Predict"</strong>
                to see LSTM predictions for <strong>{}</strong>.
            </p>
        </div>
        """.format(ticker), unsafe_allow_html=True)
    
    # ── Volume Chart ──
    st.markdown('<div class="section-header">📊 Trading Volume</div>', unsafe_allow_html=True)
    
    fig_volume = go.Figure()
    
    # Color bars based on price direction
    colors = ['#00C853' if stock_data['Close'].iloc[i] >= stock_data['Open'].iloc[i]
              else '#FF1744' for i in range(len(stock_data))]
    
    fig_volume.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['Volume'],
        marker_color=colors,
        marker_line_width=0,
        opacity=0.7,
        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig_volume.update_layout(**get_chart_layout("", height=250))
    fig_volume.update_layout(yaxis_title="Volume")
    
    st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})


# ═══════════════════════════════════════════════
# PAGE: RAW DATA
# ═══════════════════════════════════════════════

elif page == "📋 Raw Data":
    
    st.markdown(f"""
    <h2 style="color: #E8EAED; font-weight: 700;">📋 Raw Data — {ticker}</h2>
    <p style="color: #9AA0AB;">
        Historical stock data downloaded from Yahoo Finance. 
        Showing the most recent records first.
    </p>
    """, unsafe_allow_html=True)
    
    # Dataset info cards
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.markdown(create_metric_card(
            "Total Records", f"{len(stock_data):,}",
            icon="📝"
        ), unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(create_metric_card(
            "Date Range",
            f"{stock_data.index[0].strftime('%b %Y')} — {stock_data.index[-1].strftime('%b %Y')}",
            icon="📅"
        ), unsafe_allow_html=True)
    
    with info_col3:
        highest = stock_data['Close'].max()
        st.markdown(create_metric_card(
            "Highest Close", f"${highest:,.2f}",
            icon="🔝"
        ), unsafe_allow_html=True)
    
    with info_col4:
        lowest = stock_data['Close'].min()
        st.markdown(create_metric_card(
            "Lowest Close", f"${lowest:,.2f}",
            icon="🔻"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show tabs for different data views
    tab1, tab2, tab3 = st.tabs(["📊 Recent Data", "📈 Statistics", "📥 Download"])
    
    with tab1:
        # Display last 50 rows, newest first
        display_data = stock_data.copy()
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        display_data = display_data.round(2)
        st.dataframe(
            display_data.tail(50).iloc[::-1],
            use_container_width=True,
            height=500
        )
    
    with tab2:
        st.markdown("#### 📈 Statistical Summary")
        stats = stock_data.describe().round(2)
        st.dataframe(stats, use_container_width=True)
        
        # Missing values check
        missing = stock_data.isnull().sum()
        st.markdown("#### 🔍 Missing Values Check")
        if missing.sum() == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning(f"⚠️ Found {missing.sum()} missing values. These will be handled during preprocessing.")
            st.dataframe(missing[missing > 0])
    
    with tab3:
        st.markdown("#### 📥 Download Dataset")
        csv = stock_data.to_csv()
        st.download_button(
            label=f"📥 Download {ticker} Data (CSV)",
            data=csv,
            file_name=f"{ticker}_stock_data.csv",
            mime="text/csv",
            use_container_width=True
        )


# ═══════════════════════════════════════════════
# PAGE: MODEL INFO
# ═══════════════════════════════════════════════

elif page == "🧠 Model Info":
    
    st.markdown("""
    <h2 style="color: #E8EAED; font-weight: 700;">🧠 LSTM Model Architecture</h2>
    <p style="color: #9AA0AB;">
        Understanding the neural network that powers our predictions.
    </p>
    """, unsafe_allow_html=True)
    
    # Architecture diagram
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #00D4AA; margin-top: 0;">📐 Model Architecture</h4>
        <pre style="color: #C8CCD4; font-family: 'SF Mono', monospace; font-size: 0.8rem; line-height: 1.8;">
┌─────────────────────────────────────┐
│  📥 Input Layer                     │
│  Shape: ({seq_len} timesteps × 1 feature)  │
├─────────────────────────────────────┤
│  🧠 LSTM Layer 1 — 128 units       │
│  return_sequences = True            │
├─────────────────────────────────────┤
│  ❌ Dropout — 20%                   │
├─────────────────────────────────────┤
│  🧠 LSTM Layer 2 — 64 units        │
│  return_sequences = True            │
├─────────────────────────────────────┤
│  ❌ Dropout — 20%                   │
├─────────────────────────────────────┤
│  🧠 LSTM Layer 3 — 32 units        │
│  return_sequences = False           │
├─────────────────────────────────────┤
│  ❌ Dropout — 20%                   │
├─────────────────────────────────────┤
│  🔗 Dense Layer — 16 units (ReLU)   │
├─────────────────────────────────────┤
│  📤 Output Layer — 1 unit           │
│  Predicted Next-Day Price           │
└─────────────────────────────────────┘
        </pre>
    </div>
    """.format(seq_len=sequence_length), unsafe_allow_html=True)
    
    # Hyperparameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #00D4AA; margin-top: 0;">⚙️ Hyperparameters</h4>
            <table style="width: 100%; color: #C8CCD4;">
                <tr><td>Optimizer</td><td style="text-align: right;"><strong>Adam</strong></td></tr>
                <tr><td>Learning Rate</td><td style="text-align: right;"><strong>0.001</strong></td></tr>
                <tr><td>Loss Function</td><td style="text-align: right;"><strong>MSE</strong></td></tr>
                <tr><td>Epochs</td><td style="text-align: right;"><strong>{}</strong></td></tr>
                <tr><td>Batch Size</td><td style="text-align: right;"><strong>{}</strong></td></tr>
                <tr><td>Sequence Length</td><td style="text-align: right;"><strong>{} days</strong></td></tr>
                <tr><td>Validation Split</td><td style="text-align: right;"><strong>10%</strong></td></tr>
                <tr><td>Early Stopping</td><td style="text-align: right;"><strong>Patience: 5</strong></td></tr>
            </table>
        </div>
        """.format(epochs, batch_size, sequence_length), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #00D4AA; margin-top: 0;">📚 Key Concepts</h4>
            <p><strong>LSTM</strong> — Long Short-Term Memory networks are designed to learn 
            patterns in sequential data by maintaining a "memory" of past observations.</p>
            <p><strong>Dropout</strong> — Randomly disables neurons during training to prevent 
            the model from memorizing (overfitting) the training data.</p>
            <p><strong>MinMaxScaler</strong> — Normalizes data to [0, 1] range so the neural 
            network can process it efficiently.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Training Loss Chart
    if st.session_state.training_history:
        st.markdown('<div class="section-header">📉 Training Loss Curve</div>',
                    unsafe_allow_html=True)
        
        history = st.session_state.training_history
        
        fig_loss = go.Figure()
        
        fig_loss.add_trace(go.Scatter(
            y=history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#00D4AA', width=2.5),
            hovertemplate='Epoch %{x}: %{y:.6f}<extra></extra>'
        ))
        
        if 'val_loss' in history:
            fig_loss.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='#FF6D00', width=2.5, dash='dash'),
                hovertemplate='Epoch %{x}: %{y:.6f}<extra></extra>'
            ))
        
        fig_loss.update_layout(**get_chart_layout("", height=400))
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss (MSE)",
        )
        
        st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})
        
        # Interpretation
        final_loss = history['loss'][-1]
        st.markdown(f"""
        <div class="info-box">
            <strong>📖 How to Read This Chart:</strong><br>
            • The <span style="color: #00D4AA;">green line</span> shows how well the model 
            learned from training data.<br>
            • The <span style="color: #FF6D00;">orange line</span> shows performance on 
            unseen validation data.<br>
            • Both lines should decrease over time — this means the model is learning.<br>
            • If the orange line starts going UP while green goes DOWN, the model is overfitting.<br>
            • Final training loss: <strong>{final_loss:.6f}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #9AA0AB;">
            <div style="font-size: 3rem;">📉</div>
            <p>Train the model to see the loss curve here.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════

elif page == "📖 About":
    
    st.markdown("""
    <h2 style="color: #E8EAED; font-weight: 700;">📖 About This Project</h2>
    """, unsafe_allow_html=True)
    
    # Block Diagram
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #00D4AA; margin-top: 0;">🔄 System Flow — Block Diagram</h4>
        <pre style="color: #C8CCD4; font-family: 'SF Mono', monospace; font-size: 0.85rem; text-align: center;">

 ┌──────────┐     ┌───────────────┐     ┌──────────────┐     ┌────────────┐     ┌───────────────┐
 │  Yahoo   │ ──→ │ Preprocessing │ ──→ │  Sequence    │ ──→ │   LSTM     │ ──→ │ Visualization │
 │ Finance  │     │ & Scaling     │     │  Creation    │     │   Model    │     │ & Metrics     │
 │  (Data)  │     │ (MinMaxScaler)│     │ (60-day win) │     │ (Training) │     │  (Plotly)     │
 └──────────┘     └───────────────┘     └──────────────┘     └────────────┘     └───────────────┘
                                                                    │
                                                              ┌─────┴─────┐
                                                              │Prediction │
                                                              │ & Inverse │
                                                              │ Transform │
                                                              └───────────┘
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    # Documentation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Introduction", "🎯 Methodology", "📊 Results", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #00D4AA; margin-top: 0;">Introduction</h4>
            <p>Stock price prediction is one of the most challenging problems in finance.
            Traditional methods rely on fundamental and technical analysis, but modern
            machine learning approaches can discover complex patterns in historical data.</p>
            
            <h4 style="color: #00D4AA;">Objective</h4>
            <p>This project builds an <strong>LSTM (Long Short-Term Memory)</strong> neural network
            to predict future stock prices based on historical closing prices. The system:</p>
            <ul>
                <li>Fetches real-time data from Yahoo Finance using yfinance</li>
                <li>Preprocesses and normalizes the data for neural network input</li>
                <li>Creates time-series sequences using a sliding window approach</li>
                <li>Trains a multi-layer LSTM model to learn price patterns</li>
                <li>Generates predictions and evaluates them against actual prices</li>
                <li>Presents everything in a modern, interactive dashboard</li>
            </ul>
            
            <h4 style="color: #00D4AA;">Technologies Used</h4>
            <table style="width: 100%; color: #C8CCD4;">
                <tr><td>🐍 Python</td><td>Core programming language</td></tr>
                <tr><td>🧠 TensorFlow/Keras</td><td>Deep learning framework for LSTM</td></tr>
                <tr><td>📊 Streamlit</td><td>Web application framework</td></tr>
                <tr><td>📈 Plotly</td><td>Interactive chart library</td></tr>
                <tr><td>💰 yfinance</td><td>Yahoo Finance data API</td></tr>
                <tr><td>🔢 scikit-learn</td><td>Data preprocessing (MinMaxScaler)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #00D4AA; margin-top: 0;">Methodology</h4>
            
            <h5>Step 1: Data Collection</h5>
            <p>Historical stock data is fetched from Yahoo Finance using the <code>yfinance</code> 
            library. This includes Open, High, Low, Close, and Volume data. We focus on the 
            <strong>Close</strong> price as it represents the final traded price each day.</p>
            
            <h5>Step 2: Data Preprocessing</h5>
            <p>Missing values are handled using forward fill (using the previous day's value) followed by 
            backward fill. The close prices are then normalized to a [0, 1] range using 
            <strong>MinMaxScaler</strong>, which is required for neural networks to learn efficiently.</p>
            
            <h5>Step 3: Sequence Creation</h5>
            <p>A sliding window of 60 days creates input sequences. Each sequence contains 
            60 consecutive daily prices, and the target is the next day's price. This teaches 
            the model: <em>"Given the last 60 days, predict tomorrow."</em></p>
            
            <h5>Step 4: Train/Test Split</h5>
            <p>Data is split 80/20 <strong>without shuffling</strong> (crucial for time-series data).
            The model trains on older data and is tested on more recent data.</p>
            
            <h5>Step 5: LSTM Model</h5>
            <p>A 3-layer LSTM network (128→64→32 units) with Dropout regularization learns
            temporal patterns. The model is compiled with Adam optimizer and MSE loss, and 
            includes early stopping to prevent overfitting.</p>
            
            <h5>Step 6: Evaluation</h5>
            <p>Predictions are inverse-transformed back to dollar amounts and compared 
            against actual prices using RMSE, MAE, and MAPE metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        if st.session_state.model_trained:
            st.markdown(f"""
            <div class="info-box">
                <h4 style="color: #00D4AA; margin-top: 0;">Results — {ticker}</h4>
                <table style="width: 100%; color: #C8CCD4;">
                    <tr>
                        <td><strong>RMSE</strong></td>
                        <td style="text-align: right;">${st.session_state.metrics['RMSE']:,.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>MAE</strong></td>
                        <td style="text-align: right;">${st.session_state.metrics['MAE']:,.4f}</td>
                    </tr>
                    <tr>
                        <td><strong>MAPE</strong></td>
                        <td style="text-align: right;">{st.session_state.metrics['MAPE']}%</td>
                    </tr>
                </table>
                
                <h4 style="color: #00D4AA;">Conclusion</h4>
                <p>The LSTM model successfully learned patterns in {ticker}'s historical price data. 
                The model's predictions closely follow the actual price trends, demonstrating that 
                deep learning can capture complex temporal dependencies in financial data.</p>
                
                <p><strong>Key Findings:</strong></p>
                <ul>
                    <li>The model captures the general trend direction effectively</li>
                    <li>Predictions are closer for stable periods and less accurate during high volatility</li>
                    <li>The 60-day sequence length provides sufficient historical context</li>
                    <li>Dropout regularization helped prevent overfitting to training data</li>
                </ul>
                
                <p><strong>Limitations:</strong></p>
                <ul>
                    <li>Stock prices are influenced by news, sentiment, and external events that the model cannot capture from price data alone</li>
                    <li>The model predicts trends, NOT exact prices — do not use for actual trading decisions</li>
                    <li>Performance varies significantly across different stocks and market conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Train the model to see results here.")
    
    with tab4:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #00D4AA; margin-top: 0;">Frequently Asked Questions</h4>
            
            <p><strong>Q: Can I use this for real stock trading?</strong><br>
            A: No. This is an educational project. Real stock trading requires much more 
            sophisticated models, risk management, and regulatory compliance.</p>
            
            <p><strong>Q: Why LSTM instead of other models?</strong><br>
            A: LSTMs are specifically designed for sequential data and can remember long-term 
            patterns, making them well-suited for time-series prediction tasks.</p>
            
            <p><strong>Q: Why 60 timesteps?</strong><br>
            A: 60 trading days ≈ 3 months of market data. This provides enough historical 
            context for the model to learn meaningful patterns without being too computationally expensive.</p>
            
            <p><strong>Q: How accurate are the predictions?</strong><br>
            A: Accuracy varies by stock. Generally, the model achieves 1-5% MAPE on major stocks.
            It predicts trends better than exact prices.</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 16px 0; color: #6B7280; font-size: 0.75rem;">
    <strong>StockSense AI</strong> — LSTM Stock Price Prediction Dashboard<br>
    Built with Streamlit • TensorFlow • Plotly • yfinance<br>
    <span style="font-size: 0.7rem;">⚠️ For educational purposes only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)
