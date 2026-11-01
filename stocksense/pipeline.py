"""End-to-end orchestration shared by the CLI and the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config
from .data import date_range_for, fetch_ohlcv, load_market_context
from .evaluation import WalkForwardResult, run_walk_forward
from .features import build_features
from .forecast import ForecastResult, direct_forecast
from .models import model_factory

DEFAULT_MODELS = ["naive", "drift", "arima", "ridge", "xgboost", "lstm"]


@dataclass
class Dataset:
    ticker: str
    prices: pd.DataFrame
    market: pd.DataFrame | None
    frame: pd.DataFrame


def load_dataset(cfg: Config, ticker: str | None = None, years: int | None = None) -> Dataset:
    d = cfg.data
    ticker = (ticker or d.ticker).upper().strip()
    start, end = date_range_for(years or d.period_years)
    prices = fetch_ohlcv(ticker, start, end, d.cache_dir, d.cache_ttl_hours)
    market = load_market_context(start, end, d.market_ticker, d.vix_ticker,
                                 d.cache_dir, d.cache_ttl_hours)
    return Dataset(ticker, prices, market, build_features(prices, market))


def evaluate(ds: Dataset, cfg: Config, models: list[str] | None = None,
             progress: Callable[[float, str], None] | None = None) -> WalkForwardResult:
    models = models or DEFAULT_MODELS
    if "naive" not in models:
        models = ["naive", *models]  # the benchmark is always required
    factories = {k: model_factory(k, cfg) for k in models}
    return run_walk_forward(ds.frame, factories, cfg.evaluation.n_folds,
                            cfg.evaluation.initial_train_frac, progress)


def backtest(result: WalkForwardResult, model: str, cfg: Config) -> BacktestResult:
    b = cfg.backtest
    p = result.predictions
    return run_backtest(p[model], p["actual"], b.threshold, b.cost_bps, b.allow_short)


def forecast(ds: Dataset, cfg: Config, horizon: int | None = None) -> ForecastResult:
    f = cfg.forecast
    return direct_forecast(ds.frame, horizon or f.horizon, f.calibration_frac,
                           f.levels, vars(cfg.models.xgboost))
