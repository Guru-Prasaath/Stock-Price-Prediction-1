"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    period_years: int = 5
    market_ticker: str | None = "SPY"
    vix_ticker: str | None = "^VIX"
    cache_dir: str = "data/cache"
    cache_ttl_hours: int = 12


@dataclass
class EvalConfig:
    n_folds: int = 5
    initial_train_frac: float = 0.6


@dataclass
class DriftConfig:
    window: int = 20


@dataclass
class RidgeConfig:
    alpha: float = 10.0


@dataclass
class ARIMAConfig:
    order: list[int] = field(default_factory=lambda: [1, 0, 1])


@dataclass
class XGBConfig:
    n_estimators: int = 400
    max_depth: int = 3
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 10
    reg_lambda: float = 5.0
    early_stopping_rounds: int = 30


@dataclass
class LSTMConfig:
    sequence_length: int = 30
    units: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    epochs: int = 40
    batch_size: int = 32
    patience: int = 6
    learning_rate: float = 1e-3


@dataclass
class ModelsConfig:
    drift: DriftConfig = field(default_factory=DriftConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    xgboost: XGBConfig = field(default_factory=XGBConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)


@dataclass
class ForecastConfig:
    horizon: int = 10
    calibration_frac: float = 0.2
    levels: list[float] = field(default_factory=lambda: [0.8, 0.95])


@dataclass
class BacktestConfig:
    cost_bps: float = 5.0
    threshold: float = 0.0
    allow_short: bool = False


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, values: dict[str, Any] | None):
    """Recursively build a dataclass from a (possibly partial) dict."""
    values = values or {}
    kwargs = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        default = f.default_factory() if callable(f.default_factory) else None
        if is_dataclass(default):
            kwargs[f.name] = _build(type(default), values[f.name])
        else:
            kwargs[f.name] = values[f.name]
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load a config file; missing keys fall back to the dataclass defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    with open(path) as fh:
        return _build(Config, yaml.safe_load(fh))
