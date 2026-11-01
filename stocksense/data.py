"""Market data acquisition with an on-disk Parquet cache."""

from __future__ import annotations

import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(RuntimeError):
    """Raised when price data cannot be obtained for a ticker."""


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[[c for c in OHLCV if c in df.columns]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    return df.astype(float)


def _cache_path(cache_dir: str | Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_").replace("=", "_")
    return Path(ROOT, cache_dir) / f"{safe}_{start}_{end}.parquet"


def fetch_ohlcv(
    ticker: str,
    start: str | date,
    end: str | date,
    cache_dir: str | Path | None = "data/cache",
    ttl_hours: float = 12,
) -> pd.DataFrame:
    """Download daily, split/dividend-adjusted OHLCV bars for ``ticker``.

    Results are cached as Parquet so repeated runs (and tests of the app)
    don't hammer Yahoo Finance.
    """
    import yfinance as yf

    start, end = str(start), str(end)
    path = _cache_path(cache_dir, ticker, start, end) if cache_dir else None
    if path and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return pd.read_parquet(path)

    try:
        with warnings.catch_warnings():  # yfinance emits pandas deprecation noise
            warnings.simplefilter("ignore")
            raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                              progress=False, threads=False)
    except Exception as exc:  # network errors, rate limits, ...
        raise DataError(f"Could not download data for {ticker}: {exc}") from exc
    if raw is None or raw.empty:
        raise DataError(f"No price data found for '{ticker}'. Check the symbol.")

    df = _normalize(raw)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    return df


def date_range_for(years: int, end: date | None = None) -> tuple[str, str]:
    end = end or date.today() + timedelta(days=1)  # yfinance `end` is exclusive
    start = end - timedelta(days=int(365.25 * years))
    return str(start), str(end)


def load_market_context(
    start: str, end: str, market_ticker: str | None, vix_ticker: str | None,
    cache_dir: str | Path | None = "data/cache", ttl_hours: float = 12,
) -> pd.DataFrame | None:
    """Closing levels of a market index and the VIX, for context features.

    Missing context is not fatal — models simply train without it.
    """
    cols = {}
    for name, sym in (("mkt_close", market_ticker), ("vix_close", vix_ticker)):
        if not sym:
            continue
        try:
            cols[name] = fetch_ohlcv(sym, start, end, cache_dir, ttl_hours)["Close"]
        except DataError:
            continue
    return pd.DataFrame(cols) if cols else None


def fetch_company_info(ticker: str) -> dict:
    """Best-effort company metadata; never raises."""
    info = {"name": ticker, "currency": "USD", "exchange": None, "sector": None,
            "industry": None, "market_cap": None, "summary": None}
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).info or {}
        info.update(
            name=raw.get("longName") or raw.get("shortName") or ticker,
            currency=raw.get("currency") or "USD",
            exchange=raw.get("fullExchangeName") or raw.get("exchange"),
            sector=raw.get("sector"),
            industry=raw.get("industry"),
            market_cap=raw.get("marketCap"),
            summary=raw.get("longBusinessSummary"),
        )
    except Exception:
        pass
    return info
