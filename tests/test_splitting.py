import json
import os

import pandas as pd
import pytest

from model.splitting.splitting_utils import split
from synthetic_data import BRANCH, ROUTE, make_calendar, make_itinerary, make_sales

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "model", "parameters.json")
with open(PARAMS_FILE) as f:
    PARAMS = json.load(f)["preprocess"]["parameters"]

CAL = make_calendar()
ITINERARY = make_itinerary(["c1"], CAL, visit_days=(0, 2, 4))
SALES = make_sales(["c1"], ["p1", "p2"])


def predictions(values):
    return pd.DataFrame({
        "unique_id": [f"uid_{BRANCH}_{ROUTE}_c1_{p}" for p in values],
        "predictions": list(values.values()),
    })


class TestSplit:
    def test_total_per_product_is_preserved(self):
        preds = {"p1": 7.0, "p2": 3.0}
        out = split(predictions(preds), SALES.copy(), ITINERARY.copy(), CAL, PARAMS)
        per_product = out.groupby("Product")["SuggestedRecal"].sum()
        assert per_product["p1"] == pytest.approx(7.0)
        assert per_product["p2"] == pytest.approx(3.0)

    def test_one_row_per_visit_day(self):
        out = split(predictions({"p1": 5.0}), SALES.copy(), ITINERARY.copy(), CAL, PARAMS)
        assert len(out) == 3                                      # 3 visits in the target week
        assert out["Date"].nunique() == 3

    def test_suggestions_follow_historical_day_shares(self):
        sales = SALES[SALES["Product"] == "p1"].copy()
        sales["Units_Sold"] = 0
        weekday_of_visit = pd.to_datetime(ITINERARY["dateVisitPlan"]).min().dayofweek
        sales.loc[sales["Date"].dt.dayofweek == weekday_of_visit, "Units_Sold"] = 10
        out = split(predictions({"p1": 6.0}), sales, ITINERARY.copy(), CAL, PARAMS)
        first_visit = out.sort_values("Date").iloc[0]
        assert first_visit["SuggestedRecal"] == pytest.approx(6.0)
