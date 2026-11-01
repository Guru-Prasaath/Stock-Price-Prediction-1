"""Forecast accuracy metrics and significance tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def forecast_metrics(actual_ret, pred_ret, close_t) -> dict[str, float]:
    """Accuracy of next-day return forecasts, in return and price space.

    ``close_t`` is the close on the forecast day, so the implied price
    forecast for t+1 is ``close_t * exp(pred_ret)``.
    """
    a, p, c = (np.asarray(x, dtype=float) for x in (actual_ret, pred_ret, close_t))
    price_true, price_pred = c * np.exp(a), c * np.exp(p)
    err = a - p
    sse_naive = np.sum(a ** 2)
    return {
        "rmse_price": float(np.sqrt(np.mean((price_true - price_pred) ** 2))),
        "mae_price": float(np.mean(np.abs(price_true - price_pred))),
        "mape": float(np.mean(np.abs((price_true - price_pred) / price_true)) * 100),
        "rmse_ret": float(np.sqrt(np.mean(err ** 2))),
        # Predicting exactly 0 counts as "up" so the random walk scores the up-day base rate.
        "dir_acc": float(np.mean((p >= 0) == (a > 0)) * 100),
        # Campbell–Thompson out-of-sample R² against a zero-return forecast.
        "r2_oos": float(1 - np.sum(err ** 2) / sse_naive) * 100 if sse_naive else np.nan,
    }


def diebold_mariano(err_model, err_bench, h: int = 1) -> tuple[float, float]:
    """Diebold–Mariano test (squared loss) with the Harvey–Leybourne–Newbold
    small-sample correction.

    Returns ``(stat, p_value)``; a negative statistic means the model has
    lower loss than the benchmark. Two-sided p-value from Student-t(n-1).
    """
    e1, e2 = np.asarray(err_model, float), np.asarray(err_bench, float)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return 0.0, 1.0
    d_mean = d.mean()
    # Newey–West long-run variance with h-1 lags.
    gamma = [np.sum((d[k:] - d_mean) * (d[:n - k] - d_mean)) / n for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        return 0.0, 1.0
    dm = d_mean / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    return stat, float(2 * stats.t.sf(abs(stat), df=n - 1))


def direction_test(actual_ret, pred_ret) -> float:
    """One-sided binomial p-value that directional accuracy beats always
    guessing the majority direction (the honest coin-flip baseline)."""
    a, p = np.asarray(actual_ret, float), np.asarray(pred_ret, float)
    hits = int(np.sum((p >= 0) == (a > 0)))
    base = max(np.mean(a > 0), 1 - np.mean(a > 0))
    return float(stats.binomtest(hits, len(a), base, alternative="greater").pvalue)
