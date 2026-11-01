import numpy as np
import pandas as pd
import pytest

from stocksense.config import Config
from stocksense.features import feature_columns, labeled
from stocksense.models import MODEL_CLASSES, model_factory
from stocksense.models.lstm import make_windows

FAST = Config()
FAST.models.xgboost.n_estimators = 50
FAST.models.lstm.epochs = 2
FAST.models.lstm.units = [8]
FAST.models.lstm.sequence_length = 10


@pytest.mark.parametrize("key", list(MODEL_CLASSES))
def test_predictions_never_use_the_future(key, frame):
    """Perturbing rows after t must leave predictions at rows <= t unchanged."""
    data = labeled(frame)
    feats = feature_columns(data)
    model = model_factory(key, FAST)().fit(data.iloc[:300], feats)

    window = data.iloc[:420]
    tampered = window.copy()
    tampered.iloc[380:, :] = tampered.iloc[380:, :] * 3.0 + 1.0

    a = model.predict(window).iloc[:380]
    b = model.predict(tampered).iloc[:380]
    pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-6)


@pytest.mark.parametrize("key", list(MODEL_CLASSES))
def test_predict_returns_aligned_series(key, frame):
    data = labeled(frame)
    model = model_factory(key, FAST)().fit(data.iloc[:300], feature_columns(data))
    out = model.predict(data.iloc[:350])
    assert isinstance(out, pd.Series)
    assert out.index.equals(data.index[:350])
    assert out.iloc[-50:].notna().all()


def test_ridge_scaler_is_fit_on_train_only(frame):
    data = labeled(frame)
    feats = feature_columns(data)
    model = model_factory("ridge", FAST)().fit(data.iloc[:300], feats)
    scaler = model.model_.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, data[feats].iloc[:300].mean().to_numpy())


def test_xgboost_shap_values_sum_to_prediction(frame):
    data = labeled(frame)
    model = model_factory("xgboost", FAST)().fit(data.iloc[:300], feature_columns(data))
    sv = model.shap_values(data.iloc[300:310])
    np.testing.assert_allclose(sv.sum(axis=1), model.predict(data.iloc[300:310]), atol=1e-5)


def test_make_windows_shape_and_alignment():
    values = np.arange(20, dtype=float).reshape(10, 2)
    w = make_windows(values, 4)
    assert w.shape == (7, 4, 2)
    np.testing.assert_array_equal(w[-1], values[-4:])   # last window ends at the last row
    assert make_windows(values[:3], 4).shape == (0, 4, 2)
