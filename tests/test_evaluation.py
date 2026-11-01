import numpy as np
import pytest

from stocksense.config import Config
from stocksense.evaluation import run_walk_forward, verdict, walk_forward_splits
from stocksense.metrics import diebold_mariano, direction_test, forecast_metrics
from stocksense.models import model_factory


def test_splits_are_contiguous_and_strictly_in_the_future():
    folds = walk_forward_splits(1000, n_folds=4, initial_train_frac=0.6)
    assert folds[0].test_start == 600
    assert folds[-1].test_end == 1000
    for f in folds:
        assert f.train_end == f.test_start          # train strictly before test
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start           # no gaps, no overlap


def test_splits_reject_tiny_datasets():
    with pytest.raises(ValueError):
        walk_forward_splits(60, n_folds=5)


def test_perfect_forecast_metrics():
    a = np.array([0.01, -0.02, 0.005])
    m = forecast_metrics(a, a, np.array([100, 101, 99]))
    assert m["rmse_price"] == pytest.approx(0)
    assert m["dir_acc"] == 100
    assert m["r2_oos"] == pytest.approx(100)


def test_random_walk_has_zero_r2():
    a = np.random.default_rng(0).normal(0, 0.01, 200)
    assert forecast_metrics(a, np.zeros_like(a), np.full_like(a, 100))["r2_oos"] == pytest.approx(0)


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(0)
    bench = rng.normal(0, 1.0, 500)
    better = bench * 0.5
    stat, p = diebold_mariano(better, bench)
    assert stat < 0 and p < 0.01
    assert diebold_mariano(bench, bench) == (0.0, 1.0)


def test_direction_test_is_not_fooled_by_base_rate():
    actual = np.r_[np.full(80, 0.01), np.full(20, -0.01)]   # 80% up days
    always_up = np.full(100, 0.001)
    assert direction_test(actual, always_up) > 0.4          # 80% accuracy is just the base rate


def test_walk_forward_end_to_end(frame):
    cfg = Config()
    cfg.models.xgboost.n_estimators = 50
    factories = {k: model_factory(k, cfg) for k in ("naive", "drift", "ridge", "xgboost")}
    res = run_walk_forward(frame, factories, n_folds=3, initial_train_frac=0.6)
    assert set(res.leaderboard.index) == set(factories)
    assert res.predictions[["ridge", "xgboost"]].notna().all().all()
    assert res.predictions["fold"].between(1, 3).all()
    assert len(res.fold_scores) == 3 * len(factories)
    tone, msg = verdict(res.leaderboard)
    assert tone in {"good", "neutral"} and msg
