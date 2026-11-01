"""Benchmarks every learned model must beat to be worth anything."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import Forecaster


class NaiveForecaster(Forecaster):
    """Random walk: tomorrow's price equals today's (zero expected return)."""

    key, name = "naive", "Random walk"
    description = "Tomorrow = today. The efficient-market benchmark."

    def fit(self, train, features):
        return self

    def predict(self, frame):
        return pd.Series(0.0, index=frame.index)


class DriftForecaster(Forecaster):
    """Expected return equals the trailing mean return (momentum)."""

    key, name = "drift", "Moving-avg drift"
    description = "Next return = mean of the last N daily returns."

    def __init__(self, window: int = 20):
        super().__init__()
        self.window = window

    def fit(self, train, features):
        return self

    def predict(self, frame):
        return frame["ret"].rolling(self.window, min_periods=self.window).mean()


class RidgeForecaster(Forecaster):
    """Regularised linear regression on the engineered features."""

    key, name = "ridge", "Ridge regression"
    description = "L2-regularised linear model on all engineered features."

    def __init__(self, alpha: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.model_ = None

    def fit(self, train, features):
        self.features_ = list(features)
        # The scaler lives inside the pipeline, so it only ever sees train rows.
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(train[self.features_], train["target"])
        return self

    def predict(self, frame):
        return pd.Series(self.model_.predict(frame[self.features_]), index=frame.index)


class ARIMAForecaster(Forecaster):
    """ARIMA on daily log returns (fit once per fold, filtered forward)."""

    key, name = "arima", "ARIMA"
    description = "Classical ARIMA(p,d,q) on log returns."

    def __init__(self, order=(1, 0, 1)):
        super().__init__()
        self.order = tuple(order)
        self.result_ = None

    _SCALE = 100.0  # fit on % returns for numerical stability

    def fit(self, train, features):
        from statsmodels.tsa.arima.model import ARIMA

        y = train["ret"].to_numpy() * self._SCALE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ARIMA(y, order=self.order, trend="c").fit()
        return self

    def predict(self, frame):
        # Apply the fitted parameters to the new history with a trailing NaN:
        # the one-step-ahead prediction at position i+1 uses data up to i only.
        y = np.append(frame["ret"].to_numpy() * self._SCALE, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            onestep = self.result_.apply(y).predict()
        return pd.Series(np.asarray(onestep)[1:] / self._SCALE, index=frame.index)
