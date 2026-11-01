"""Gradient-boosted trees (XGBoost) with built-in SHAP explanations."""

from __future__ import annotations

import pandas as pd

from .base import Forecaster


class XGBForecaster(Forecaster):
    key, name = "xgboost", "XGBoost"
    description = "Gradient-boosted trees on engineered features."

    def __init__(self, n_estimators=400, max_depth=3, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                 reg_lambda=5.0, early_stopping_rounds=30, random_state=42):
        super().__init__()
        self.params = dict(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            random_state=random_state, n_jobs=-1, objective="reg:squarederror",
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None

    def fit(self, train, features):
        from xgboost import XGBRegressor

        self.features_ = list(features)
        X, y = train[self.features_], train["target"]
        params = dict(self.params)
        if self.early_stopping_rounds and len(train) > 200:
            # Chronological hold-out (last 15% of the training window) for early stopping.
            cut = int(len(train) * 0.85)
            self.model_ = XGBRegressor(**params, early_stopping_rounds=self.early_stopping_rounds)
            self.model_.fit(X.iloc[:cut], y.iloc[:cut],
                            eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
        else:
            self.model_ = XGBRegressor(**params)
            self.model_.fit(X, y, verbose=False)
        return self

    def predict(self, frame):
        return pd.Series(self.model_.predict(frame[self.features_]), index=frame.index)

    def shap_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Exact TreeSHAP contributions (last column ``bias`` = base value)."""
        from xgboost import DMatrix

        booster = self.model_.get_booster()
        best = getattr(self.model_, "best_iteration", None)
        kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
        contrib = booster.predict(DMatrix(frame[self.features_]), pred_contribs=True, **kwargs)
        return pd.DataFrame(contrib, index=frame.index, columns=self.features_ + ["bias"])

    def feature_importance(self, frame: pd.DataFrame) -> pd.Series:
        """Mean |SHAP| per feature — global importance on ``frame``."""
        sv = self.shap_values(frame).drop(columns="bias")
        return sv.abs().mean().sort_values(ascending=False)
