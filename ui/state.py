"""Cached data access and cross-page session state."""

from __future__ import annotations

import warnings

import streamlit as st

from stocksense import pipeline
from stocksense.config import Config, load_config
from stocksense.data import DataError, fetch_company_info
from stocksense.features import feature_columns, labeled
from stocksense.models import model_factory

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

POPULAR = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta Platforms", "TSLA": "Tesla", "JPM": "JPMorgan Chase",
    "V": "Visa", "NFLX": "Netflix", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF",
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy", "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
}
PERIODS = {"1Y": 1, "2Y": 2, "5Y": 5, "10Y": 10}
MIN_YEARS_FOR_MODELS = 2


@st.cache_resource
def config() -> Config:
    return load_config()


@st.cache_data(ttl=3600, show_spinner=False)
def dataset(ticker: str, years: int) -> pipeline.Dataset:
    return pipeline.load_dataset(config(), ticker, years)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def company(ticker: str) -> dict:
    return fetch_company_info(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def forecast(ticker: str, years: int, horizon: int):
    return pipeline.forecast(dataset(ticker, years), config(), horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def xgb_importance(ticker: str, years: int):
    """Global SHAP importance of XGBoost refit on all labeled history."""
    data = labeled(dataset(ticker, years).frame)
    model = model_factory("xgboost", config())().fit(data, feature_columns(data))
    return model.feature_importance(data)


def selection() -> tuple[str, int]:
    """The ticker/period chosen in the global scope bar."""
    ticker = (st.session_state.get("ticker") or "AAPL").upper().strip()
    years = PERIODS[st.session_state.get("period") or "5Y"]
    return ticker, years


def current_dataset() -> pipeline.Dataset:
    ticker, years = selection()
    try:
        return dataset(ticker, years)
    except DataError as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()


def lab_key(ticker: str, years: int) -> str:
    return f"{ticker}|{years}"


def lab_result():
    """Walk-forward results for the current selection, if Model Lab has run."""
    ticker, years = selection()
    return st.session_state.get("lab", {}).get(lab_key(ticker, years))


def store_lab_result(result) -> None:
    ticker, years = selection()
    st.session_state.setdefault("lab", {})[lab_key(ticker, years)] = result
