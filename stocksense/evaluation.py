"""Walk-forward (expanding-window) evaluation and the model leaderboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .features import feature_columns, labeled
from .metrics import diebold_mariano, direction_test, forecast_metrics
from .models import BENCHMARK, MODEL_NAMES, Forecaster


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: int   # train on labeled rows [0, train_end)
    test_start: int  # test on labeled rows [test_start, test_end)
    test_end: int


def walk_forward_splits(n: int, n_folds: int = 5, initial_train_frac: float = 0.6) -> list[Fold]:
    """Contiguous test blocks after an expanding training window.

    Each fold trains on everything before its test block, so the model is
    always evaluated on data strictly in its future.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    initial = int(n * initial_train_frac)
    size = (n - initial) // n_folds
    if initial < 50 or size < 5:
        raise ValueError("Not enough history for walk-forward evaluation — use a longer period.")
    folds = []
    for k in range(n_folds):
        start = initial + k * size
        end = n if k == n_folds - 1 else start + size
        folds.append(Fold(k + 1, start, start, end))
    return folds


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame            # per test row: actual, close, fold, <model keys>
    leaderboard: pd.DataFrame            # one row per model
    fold_scores: pd.DataFrame            # rmse_ret per (fold, model)
    folds: list[Fold]
    histories: dict[str, list[dict]] = field(default_factory=dict)  # LSTM loss curves
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return list(self.leaderboard.index)

    @property
    def best_model(self) -> str:
        return self.leaderboard.index[0]


def run_walk_forward(
    frame: pd.DataFrame,
    factories: dict[str, Callable[[], Forecaster]],
    n_folds: int = 5,
    initial_train_frac: float = 0.6,
    progress: Callable[[float, str], None] | None = None,
) -> WalkForwardResult:
    data = labeled(frame)
    features = feature_columns(data)
    folds = walk_forward_splits(len(data), n_folds, initial_train_frac)

    test_rows = data.iloc[folds[0].test_start:]
    preds = pd.DataFrame({"actual": test_rows["target"], "close": test_rows["close"]})
    preds["fold"] = 0
    for f in folds:
        preds.iloc[f.test_start - folds[0].test_start: f.test_end - folds[0].test_start,
                   preds.columns.get_loc("fold")] = f.number

    histories: dict[str, list[dict]] = {}
    timings: dict[str, float] = {k: 0.0 for k in factories}
    total, step = len(folds) * len(factories), 0
    for key, factory in factories.items():
        col = pd.Series(np.nan, index=preds.index)
        for f in folds:
            step += 1
            if progress:
                progress(step / total, f"{MODEL_NAMES.get(key, key)} · fold {f.number}/{len(folds)}")
            t0 = time.perf_counter()
            model = factory().fit(data.iloc[:f.train_end], features)
            # Pass history up to the fold end so sequence models have context;
            # predictions for each test row only use rows at or before it.
            p = model.predict(data.iloc[:f.test_end]).iloc[f.test_start:f.test_end]
            col.loc[p.index] = p.to_numpy()
            timings[key] += time.perf_counter() - t0
            if getattr(model, "history_", None):
                histories.setdefault(key, []).append(model.history_)
        preds[key] = col

    leaderboard, fold_scores = score(preds, list(factories))
    return WalkForwardResult(preds, leaderboard, fold_scores, folds, histories, timings)


def score(preds: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leaderboard (sorted by return RMSE) with significance vs the random walk."""
    rows, fold_rows = {}, []
    bench_err = preds["actual"] - 0.0
    for key in models:
        ok = preds[key].notna()
        p = preds.loc[ok]
        m = forecast_metrics(p["actual"], p[key], p["close"])
        if key == BENCHMARK:
            m["dm_stat"], m["dm_p"], m["dir_p"] = np.nan, np.nan, np.nan
        else:
            m["dm_stat"], m["dm_p"] = diebold_mariano(p["actual"] - p[key], bench_err[ok])
            m["dir_p"] = direction_test(p["actual"], p[key])
        m["coverage"] = float(ok.mean() * 100)
        rows[key] = m
        for fold, g in p.groupby("fold"):
            fold_rows.append({"fold": int(fold), "model": key,
                              "rmse_ret": float(np.sqrt(np.mean((g["actual"] - g[key]) ** 2)))})
    board = pd.DataFrame.from_dict(rows, orient="index").sort_values("rmse_ret")
    board.insert(0, "name", [MODEL_NAMES.get(k, k) for k in board.index])
    board["beats_benchmark"] = (board["dm_stat"] < 0) & (board["dm_p"] < 0.05)
    return board, pd.DataFrame(fold_rows)


def verdict(board: pd.DataFrame) -> tuple[str, str]:
    """Plain-English conclusion; returns (tone, message) with tone in good|neutral."""
    winners = board[board["beats_benchmark"]]
    if not winners.empty:
        best = winners.iloc[0]
        return "good", (
            f"**{best['name']}** beats the random walk with statistical significance "
            f"(Diebold–Mariano p = {best['dm_p']:.3f}) — a rare and noteworthy result. "
            "Confirm it holds on other tickers and periods before trusting it."
        )
    best = board.drop(index=BENCHMARK, errors="ignore").iloc[0]
    return "neutral", (
        f"No model significantly beats the random walk. The closest is **{best['name']}** "
        f"(Diebold–Mariano p = {best['dm_p']:.2f}). That is the expected outcome for liquid "
        "stocks and is consistent with the weak-form efficient-market hypothesis — "
        "low price-RMSE alone mostly reflects that tomorrow's price is close to today's."
    )
