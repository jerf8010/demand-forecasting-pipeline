import json
import os

import pandas as pd
import pytest

from model.preprocess.preprocess_utils import columns2unique_id, data_process, get_fiscal_calendar
from synthetic_data import BRANCH, ROUTE, WEEK, make_calendar, make_itinerary, make_sales

PARAMS_FILE = os.path.join(os.path.dirname(__file__), "..", "model", "parameters.json")
with open(PARAMS_FILE) as f:
    PARAMS = json.load(f)["preprocess"]["parameters"]
PARAMS = {**PARAMS, "branch_name": ["branch_a"], "route": [ROUTE]}
CAL = make_calendar()


def run(sales, itinerary):
    return data_process(sales.copy(), CAL, itinerary.copy(), PARAMS)


class TestFiscalCalendar:
    def test_has_52_weeks_of_7_days(self):
        cal = get_fiscal_calendar(2022)
        assert len(cal) == 364
        assert cal["week_num"].min() == 1 and cal["week_num"].max() == 52
        assert (cal.groupby("week_num").size() == 7).all()

    def test_weeks_start_on_thursday(self):
        cal = get_fiscal_calendar(2022)
        first_days = cal[cal["day"] == 1]["Date"]
        assert (first_days.dt.dayofweek == 3).all()      # Thursday


class TestColumns2UniqueId:
    def test_builds_unique_id_and_drops_id_columns(self):
        df = pd.DataFrame({"Branch": [1], "Customer": [2], "Product": [3], "x": [9]})
        out = columns2unique_id(df, ["Branch", "Customer", "Product"])
        assert out["unique_id"].tolist() == ["uid_1_2_3"]
        assert list(out.columns) == ["x", "unique_id"]


class TestDataProcess:
    def test_returns_none_when_no_customer_is_in_the_itinerary(self):
        sales = make_sales(["c1"], ["p1"])
        assert run(sales, make_itinerary(["other"])) is None

    def test_valid_series_are_pivoted_and_split(self):
        sales = make_sales(["c1", "c2"], ["p1"])
        pivot, returns, train, test = run(sales, make_itinerary(["c1", "c2"]))
        assert sorted(pivot.index) == [f"uid_{BRANCH}_{ROUTE}_c1_p1", f"uid_{BRANCH}_{ROUTE}_c2_p1"]
        assert test.shape[1] == PARAMS["benchmark"]
        assert train.shape[1] + test.shape[1] == pivot.shape[1]
        assert (pivot.values >= 0).all()

    def test_customers_outside_the_itinerary_are_dropped(self):
        sales = make_sales(["c1", "c2"], ["p1"])
        pivot, *_ = run(sales, make_itinerary(["c1"]))
        assert all("_c2_" not in uid for uid in pivot.index)

    def test_recently_added_series_are_removed(self):
        old = make_sales(["c1"], ["p1"])
        new = make_sales(["c2"], ["p1"], first_sale_week=31)      # first sale 2 weeks before the target
        pivot, *_ = run(pd.concat([old, new]), make_itinerary(["c1", "c2"]))
        assert all("_c2_" not in uid for uid in pivot.index)
        assert any("_c1_" in uid for uid in pivot.index)

    def test_series_without_sales_in_last_eight_weeks_are_removed(self):
        active = make_sales(["c1"], ["p1"])
        stopped = make_sales(["c2"], ["p1"], last_sale_week=WEEK - 10)
        pivot, *_ = run(pd.concat([active, stopped]), make_itinerary(["c1", "c2"]))
        assert all("_c2_" not in uid for uid in pivot.index)
