"""Common interface for every forecaster."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    """Predicts the next-day log return for every row of a model frame.

    Contract (enforced by tests): the prediction for row ``t`` may only use
    rows ``<= t`` of the frame passed to :meth:`predict`, plus whatever was
    learned in :meth:`fit`. Walk-forward evaluation relies on this.
    """

    key: str = ""
    name: str = ""
    description: str = ""

    def __init__(self) -> None:
        self.features_: list[str] = []

    @abstractmethod
    def fit(self, train: pd.DataFrame, features: list[str]) -> "Forecaster":
        """Learn from ``train`` (rows with a known ``target``)."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Return a Series aligned to ``frame.index`` (NaN where undefined)."""
