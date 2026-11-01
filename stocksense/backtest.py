"""Turn out-of-sample forecasts into a simple, cost-aware trading backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    daily: pd.DataFrame        # position, strategy/benchmark returns, equity, drawdown
    strategy: dict[str, float]
    benchmark: dict[str, float]


def performance(simple_returns: pd.Series, position: pd.Series | None = None) -> dict[str, float]:
    r = simple_returns.fillna(0.0)
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1) if len(r) else 0.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    dd = equity / equity.cummax() - 1
    stats = {
        "total_return": total * 100,
        "cagr": ((1 + total) ** (1 / years) - 1) * 100 if years > 0 and total > -1 else np.nan,
        "volatility": vol * 100,
        "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
        "sortino": float(r.mean() * TRADING_DAYS / downside) if downside and downside > 0 else 0.0,
        "max_drawdown": float(dd.min() * 100),
    }
    if position is not None:
        active = position != 0
        stats["exposure"] = float(active.mean() * 100)
        stats["hit_rate"] = float((r[active] > 0).mean() * 100) if active.any() else 0.0
        stats["trades"] = int((position.diff().fillna(position.iloc[0]) != 0).sum())
    return stats


def run_backtest(pred_ret: pd.Series, actual_ret: pd.Series, threshold: float = 0.0,
                 cost_bps: float = 5.0, allow_short: bool = False) -> BacktestResult:
    """Go long when the forecast exceeds ``threshold`` (short below −threshold
    if allowed), hold for one day, pay ``cost_bps`` per unit of turnover.

    The position decided at the close of day t earns the return of t+1 —
    exactly what the model's target represents, so there is no look-ahead.
    """
    ok = pred_ret.notna() & actual_ret.notna()
    p, a = pred_ret[ok], actual_ret[ok]
    position = pd.Series(np.where(p > threshold, 1.0, 0.0), index=p.index)
    if allow_short:
        position[p < -threshold] = -1.0
    turnover = position.diff().abs().fillna(position.abs())
    market = np.expm1(a)
    strat = position * market - turnover * cost_bps / 1e4

    daily = pd.DataFrame({
        "position": position,
        "strategy_ret": strat,
        "benchmark_ret": market,
    })
    # Index labels are forecast dates; returns are realised on the next session.
    daily["strategy_equity"] = (1 + daily["strategy_ret"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_ret"]).cumprod()
    daily["strategy_dd"] = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1
    daily["benchmark_dd"] = daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1
    return BacktestResult(daily, performance(strat, position), performance(market))


def monthly_returns(simple_returns: pd.Series) -> pd.DataFrame:
    """Year × month table of compounded returns (%)."""
    m = (1 + simple_returns).groupby([simple_returns.index.year, simple_returns.index.month]).prod() - 1
    table = (m * 100).unstack()
    table.index.name, table.columns.name = "Year", "Month"
    return table
