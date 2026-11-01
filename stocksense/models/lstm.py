"""Stacked LSTM on windows of engineered features."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import Forecaster

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def make_windows(values: np.ndarray, length: int) -> np.ndarray:
    """Window ending at row t → shape (n - length + 1, length, n_features)."""
    if len(values) < length:
        return np.empty((0, length, values.shape[1]), dtype=np.float32)
    idx = np.arange(length)[None, :] + np.arange(len(values) - length + 1)[:, None]
    return values[idx].astype(np.float32)


class LSTMForecaster(Forecaster):
    key, name = "lstm", "LSTM"
    description = "Stacked LSTM over rolling windows of engineered features."

    def __init__(self, sequence_length=30, units=(64, 32), dropout=0.2, epochs=40,
                 batch_size=32, patience=6, learning_rate=1e-3, seed=42):
        super().__init__()
        self.sequence_length = sequence_length
        self.units = tuple(units)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.seed = seed
        self.scaler_ = None
        self.y_scale_ = 1.0
        self.model_ = None
        self.history_: dict[str, list[float]] = {}

    def _build(self, n_features: int):
        import keras
        from keras import layers

        keras.utils.set_random_seed(self.seed)
        model = keras.Sequential([keras.Input(shape=(self.sequence_length, n_features))])
        for i, u in enumerate(self.units):
            model.add(layers.LSTM(u, return_sequences=i < len(self.units) - 1))
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def fit(self, train, features):
        import keras

        self.features_ = list(features)
        L = self.sequence_length
        # Scalers are fitted on the training window only — no look-ahead.
        self.scaler_ = StandardScaler().fit(train[self.features_])
        self.y_scale_ = float(train["target"].std()) or 1.0

        X = make_windows(self.scaler_.transform(train[self.features_]), L)
        y = (train["target"].to_numpy()[L - 1:] / self.y_scale_).astype(np.float32)

        self.model_ = self._build(len(self.features_))
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                             restore_best_weights=True)
        # validation_split takes the *last* 10% of windows, i.e. the most recent
        # period; shuffling the remaining windows is safe because each window is
        # self-contained.
        hist = self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                               validation_split=0.1, callbacks=[stop], shuffle=True, verbose=0)
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return self

    def predict(self, frame):
        L = self.sequence_length
        out = pd.Series(np.nan, index=frame.index)
        X = make_windows(self.scaler_.transform(frame[self.features_]), L)
        if len(X):
            # Direct call (not .predict) avoids tf.function retracing on every fold.
            pred = np.asarray(self.model_(X, training=False)).ravel() * self.y_scale_
            out.iloc[L - 1:] = pred
        return out
