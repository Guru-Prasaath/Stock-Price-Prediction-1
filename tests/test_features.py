import numpy as np
import pandas as pd

from stocksense.features import build_features, feature_columns, labeled, rsi


def test_no_lookahead_in_features(prices, market):
    """Truncating the future must not change any past feature value."""
    full = build_features(prices, market)
    cut = prices.index[400]
    partial = build_features(prices.loc[:cut], market.loc[:cut])
    cols = feature_columns(full)
    common = partial.index
    pd.testing.assert_frame_equal(full.loc[common, cols], partial[cols], check_exact=False, rtol=1e-10)


def test_target_is_next_day_log_return(frame, prices):
    close = prices["Close"]
    expected = np.log(close.shift(-1) / close).reindex(frame.index)
    np.testing.assert_allclose(frame["target"].dropna(), expected.dropna().loc[frame["target"].dropna().index])


def test_only_last_row_is_unlabeled(frame):
    assert frame["target"].isna().sum() == 1
    assert pd.isna(frame["target"].iloc[-1])
    assert len(labeled(frame)) == len(frame) - 1


def test_features_are_complete_and_finite(frame):
    X = frame[feature_columns(frame)]
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()
    assert {"mkt_ret_0", "vix_level", "rsi_14", "volume_z"} <= set(X.columns)


def test_market_context_is_optional(prices):
    f = build_features(prices, None)
    assert not any(c.startswith(("mkt_", "vix_")) for c in f.columns)


def test_rsi_bounds(prices):
    r = rsi(prices["Close"]).dropna()
    assert r.between(0, 1).all()
