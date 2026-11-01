import numpy as np
import pandas as pd
import pytest

from stocksense.backtest import monthly_returns, run_backtest
from stocksense.config import DEFAULT_CONFIG_PATH, load_config
from stocksense.forecast import direct_forecast


@pytest.fixture
def returns():
    idx = pd.bdate_range("2023-01-02", periods=300)
    return pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 300), index=idx)


def test_always_long_without_costs_equals_buy_and_hold(returns):
    bt = run_backtest(pd.Series(1.0, index=returns.index), returns, cost_bps=0)
    np.testing.assert_allclose(bt.daily["strategy_equity"], bt.daily["benchmark_equity"])
    assert bt.strategy["exposure"] == 100


def test_costs_reduce_returns(returns):
    flip = pd.Series(np.where(np.arange(300) % 2, 1.0, -1.0), index=returns.index)
    free = run_backtest(flip, returns, cost_bps=0)
    costly = run_backtest(flip, returns, cost_bps=10)
    assert costly.strategy["total_return"] < free.strategy["total_return"]


def test_perfect_foresight_is_profitable(returns):
    bt = run_backtest(returns, returns, cost_bps=0)      # forecast == realised next-day return
    assert bt.strategy["total_return"] > bt.benchmark["total_return"]
    assert bt.strategy["max_drawdown"] == pytest.approx(0)


def test_monthly_table(returns):
    table = monthly_returns(np.expm1(returns))
    assert table.shape[1] <= 12 and table.notna().sum().sum() > 10


def test_forecast_table_is_well_formed(frame):
    fc = direct_forecast(frame, horizon=3, xgb_params={"n_estimators": 40})
    t = fc.table
    assert len(t) == 3
    assert (t["date"] > frame.index[-1]).all() and t["date"].dt.dayofweek.lt(5).all()
    assert (t["lo_95"] <= t["lo_80"]).all() and (t["lo_80"] <= t["hi_80"]).all()
    assert (t["hi_80"] <= t["hi_95"]).all()
    assert fc.base_value + fc.contributions.sum() == pytest.approx(t["pred_ret"].iloc[0], abs=1e-5)


def test_default_config_loads():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.evaluation.n_folds >= 1
    assert cfg.models.lstm.units and cfg.forecast.horizon > 0
