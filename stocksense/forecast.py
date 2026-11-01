"""Multi-day price forecast with calibrated (split-conformal) intervals.

Uses the *direct* strategy: one model per horizon h predicts the cumulative
log return ``log(C[t+h] / C[t])``. Interval widths come from the empirical
quantiles of each horizon's residuals on a held-out calibration tail, so they
reflect how wrong the model actually was rather than an assumed distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import feature_columns
from .models.gbm import XGBForecaster


@dataclass
class ForecastResult:
    table: pd.DataFrame          # one row per future business day
    last_date: pd.Timestamp
    last_close: float
    contributions: pd.Series     # SHAP drivers of the 1-day forecast
    base_value: float
    levels: list[float]


def direct_forecast(frame: pd.DataFrame, horizon: int = 10, calibration_frac: float = 0.2,
                    levels=(0.8, 0.95), xgb_params: dict | None = None) -> ForecastResult:
    features = feature_columns(frame)
    close = frame["close"]
    last = frame.iloc[[-1]]
    params = dict(xgb_params or {})
    params.update(n_estimators=min(params.get("n_estimators", 300), 300), early_stopping_rounds=0)

    rows, contributions, base_value = [], None, 0.0
    for h in range(1, horizon + 1):
        target = np.log(close.shift(-h) / close)
        data = frame.assign(target=target).dropna(subset=["target"])
        cut = int(len(data) * (1 - calibration_frac))

        calib_model = XGBForecaster(**params).fit(data.iloc[:cut], features)
        resid = data["target"].iloc[cut:] - calib_model.predict(data.iloc[cut:])

        final = XGBForecaster(**params).fit(data, features)
        mean = float(final.predict(last).iloc[0])
        row = {"h": h, "pred_ret": mean}
        for lvl in levels:
            a = (1 - lvl) / 2
            row[f"lo_{int(lvl * 100)}"] = mean + float(resid.quantile(a))
            row[f"hi_{int(lvl * 100)}"] = mean + float(resid.quantile(1 - a))
        rows.append(row)
        if h == 1:
            sv = final.shap_values(last).iloc[0]
            base_value = float(sv["bias"])
            contributions = sv.drop("bias")

    # Convert cumulative log returns to price levels; keep pred_ret for reference.
    table = pd.DataFrame(rows)
    last_close = float(close.iloc[-1])
    table["pred"] = last_close * np.exp(table["pred_ret"])
    for lvl in levels:
        for side in ("lo", "hi"):
            col = f"{side}_{int(lvl * 100)}"
            table[col] = last_close * np.exp(table[col])
    table.insert(0, "date", pd.bdate_range(frame.index[-1], periods=horizon + 1)[1:])
    return ForecastResult(table, frame.index[-1], last_close, contributions, base_value, list(levels))
