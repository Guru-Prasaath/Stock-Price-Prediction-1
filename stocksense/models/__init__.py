"""Model registry: build any forecaster from its key and the config."""

from __future__ import annotations

from typing import Callable

from ..config import Config
from .base import Forecaster
from .baselines import ARIMAForecaster, DriftForecaster, NaiveForecaster, RidgeForecaster
from .gbm import XGBForecaster
from .lstm import LSTMForecaster

MODEL_CLASSES: dict[str, type[Forecaster]] = {
    cls.key: cls
    for cls in (NaiveForecaster, DriftForecaster, ARIMAForecaster,
                RidgeForecaster, XGBForecaster, LSTMForecaster)
}
MODEL_NAMES = {k: cls.name for k, cls in MODEL_CLASSES.items()}
BENCHMARK = "naive"


def model_factory(key: str, cfg: Config) -> Callable[[], Forecaster]:
    """Return a zero-arg constructor so each walk-forward fold gets a fresh model."""
    m = cfg.models
    builders: dict[str, Callable[[], Forecaster]] = {
        "naive": NaiveForecaster,
        "drift": lambda: DriftForecaster(m.drift.window),
        "arima": lambda: ARIMAForecaster(m.arima.order),
        "ridge": lambda: RidgeForecaster(m.ridge.alpha),
        "xgboost": lambda: XGBForecaster(**vars(m.xgboost)),
        "lstm": lambda: LSTMForecaster(**vars(m.lstm)),
    }
    if key not in builders:
        raise KeyError(f"Unknown model '{key}'. Choose from {list(builders)}")
    return builders[key]


__all__ = ["Forecaster", "MODEL_CLASSES", "MODEL_NAMES", "BENCHMARK", "model_factory"]
