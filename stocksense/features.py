"""Leakage-safe feature engineering.

Every feature on row ``t`` is computed only from information available at the
close of day ``t``. The modelling target on row ``t`` is the *next* day's log
return, ``log(C[t+1] / C[t])``. ``tests/test_features.py`` verifies this by
checking that truncating the future never changes past feature values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns in the model frame that are *not* model inputs.
NON_FEATURES = ("close", "ret", "target")

FEATURE_DESCRIPTIONS = {
    "ret_lag_0": "Today's log return",
    "ret_lag_1": "Log return 1 day ago",
    "ret_lag_2": "Log return 2 days ago",
    "ret_lag_3": "Log return 3 days ago",
    "ret_lag_4": "Log return 4 days ago",
    "mom_5": "5-day momentum (log)",
    "mom_21": "21-day momentum (log)",
    "vol_10": "10-day realised volatility",
    "vol_21": "21-day realised volatility",
    "rsi_14": "14-day RSI (0–1)",
    "macd": "MACD line / price",
    "macd_hist": "MACD histogram / price",
    "bb_pctb": "Bollinger %B (20d, 2σ)",
    "sma_gap_10": "Price vs 10-day SMA",
    "sma_gap_50": "Price vs 50-day SMA",
    "hl_range": "Intraday high–low range",
    "oc_ret": "Open-to-close log return",
    "volume_z": "Volume z-score (20d)",
    "mkt_ret_0": "Market (SPY) log return",
    "mkt_mom_5": "Market 5-day momentum",
    "mkt_vol_21": "Market 21-day volatility",
    "vix_level": "VIX level / 100",
    "vix_chg": "VIX daily log change",
}


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index scaled to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (1 - 1 / (1 + rs)).fillna(1.0).where(gain.notna())


def build_features(prices: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn OHLCV bars (plus optional market context) into a model frame.

    Returns a frame with ``close``, ``ret`` (today's log return), engineered
    features and ``target`` (tomorrow's log return; NaN on the final row).
    """
    close, high, low, open_ = prices["Close"], prices["High"], prices["Low"], prices["Open"]
    ret = np.log(close).diff()

    f = pd.DataFrame(index=prices.index)
    f["close"] = close
    f["ret"] = ret

    for lag in range(5):
        f[f"ret_lag_{lag}"] = ret.shift(lag)
    f["mom_5"] = np.log(close / close.shift(5))
    f["mom_21"] = np.log(close / close.shift(21))
    f["vol_10"] = ret.rolling(10).std()
    f["vol_21"] = ret.rolling(21).std()
    f["rsi_14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd / close
    f["macd_hist"] = (macd - signal) / close

    sma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_pctb"] = (close - (sma20 - 2 * sd20)) / (4 * sd20)
    f["sma_gap_10"] = close / close.rolling(10).mean() - 1
    f["sma_gap_50"] = close / close.rolling(50).mean() - 1
    f["hl_range"] = (high - low) / close
    f["oc_ret"] = np.log(close / open_)

    if "Volume" in prices and prices["Volume"].gt(0).any():
        log_vol = np.log(prices["Volume"].replace(0, np.nan)).ffill()
        f["volume_z"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()

    if market is not None and not market.empty:
        # Forward-fill only (never back-fill) so context is as-of each date.
        m = market.reindex(market.index.union(f.index)).ffill().reindex(f.index)
        if "mkt_close" in m:
            mret = np.log(m["mkt_close"]).diff()
            f["mkt_ret_0"] = mret
            f["mkt_mom_5"] = np.log(m["mkt_close"] / m["mkt_close"].shift(5))
            f["mkt_vol_21"] = mret.rolling(21).std()
        if "vix_close" in m:
            f["vix_level"] = m["vix_close"] / 100
            f["vix_chg"] = np.log(m["vix_close"]).diff()

    f["target"] = ret.shift(-1)
    f = f.replace([np.inf, -np.inf], np.nan)
    # Warm-up rows lack complete features; the last row keeps a NaN target
    # because it is the row we forecast from.
    return f.dropna(subset=feature_columns(f))


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in NON_FEATURES]


def labeled(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose next-day target is known (i.e. usable for training/testing)."""
    return frame[frame["target"].notna()]
