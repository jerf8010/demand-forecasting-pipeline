"""End-to-end demo on synthetic data: forecast -> suggested order -> offline scoring.

    python examples/run_demo.py
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "model"), os.path.join(ROOT, "tests")]

from darts.models.forecasting.croston import Croston
from darts.models.forecasting.kalman_forecaster import KalmanForecaster

from model.scoring.scoring_utils import scoring
from model.training.training_utils import fit_model
from synthetic_data import make_pivot_series_weekly

with open(os.path.join(ROOT, "model", "parameters.json")) as f:
    pars = json.load(f)
train_params = pars["training"]["parameters"]
window = train_params["validation_window"]

data = make_pivot_series_weekly(n_series=8, n_weeks=104)
train, test = data.iloc[:, :-window], data.iloc[:, -window:]

future, validation, weights, trend, suggested, pct = fit_model(
    data_for_train=data,
    darts_models=[KalmanForecaster(dim_x=1), Croston(version="sba")],
    params=train_params,
)
print("\nSuggested order for next week (first 5 series):")
for uid, value in list(zip(data.index, suggested))[:5]:
    print(f"  {uid}: {value[0]:.2f}")

_, mae, _ = scoring(train, test, validation)
print("\nBenchmark vs model MAE by customer-volume segment:")
print(mae)
