import numpy as np
import pandas as pd
import pytest

from stocksense.features import build_features


def make_prices(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic geometric-random-walk OHLCV bars on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def make_market(index: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    return pd.DataFrame({
        "mkt_close": 300 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))),
        "vix_close": 20 * np.exp(np.cumsum(rng.normal(0, 0.05, n))).clip(9, 80),
    }, index=index)


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture(scope="session")
def market(prices) -> pd.DataFrame:
    return make_market(prices.index)


@pytest.fixture(scope="session")
def frame(prices, market) -> pd.DataFrame:
    return build_features(prices, market)
