import json
import os

import pytest
from sklearn.metrics import median_absolute_error
from darts.models.forecasting.croston import Croston
from darts.models.forecasting.kalman_forecaster import KalmanForecaster

from model.training.training_utils import fit_model
from synthetic_data import make_pivot_series_weekly

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "model", "parameters.json")
with open(PARAMS_FILE) as f:
    PARS = json.load(f)
TRAIN_PARAMS = PARS["training"]["parameters"]
VALIDATION = PARS["preprocess"]["parameters"]["benchmark"]
PCT = TRAIN_PARAMS["sales_increase_importance"]
MODELS = lambda: [KalmanForecaster(dim_x=1), Croston(version="sba")]


@pytest.mark.filterwarnings("ignore")
class TestFitModel:
    def test_constant_series_is_predicted_as_constant(self):
        data = make_pivot_series_weekly(n_series=1).iloc[[0]].apply(lambda col: 1.0)
        future, validation, _, _, suggested, _ = fit_model(data, MODELS(), TRAIN_PARAMS)
        assert [round(v) for v in validation[0][:VALIDATION]] == [1] * VALIDATION
        assert round(future[0][0]) == 1
        assert round(suggested[0][0]) == 1

    def test_alternating_series_is_learned(self):
        data = make_pivot_series_weekly(n_series=1).iloc[[0]].copy()
        data.iloc[0, :] = [0.0 if i % 2 == 0 else 1.0 for i in range(data.shape[1])]
        future, validation, _, _, suggested, _ = fit_model(data, MODELS(), TRAIN_PARAMS)
        actual = data.iloc[0, data.shape[1] - VALIDATION:]
        assert median_absolute_error(validation[0], actual) < 1
        assert round(future[0][0]) == 1
        assert suggested[0][0] == pytest.approx(future[0][0] * (1 + PCT), abs=0.1)
