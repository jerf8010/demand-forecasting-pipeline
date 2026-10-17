"""Synthetic data generators used by the test-suite and the demo script.

Everything here is randomly generated: there is no real sales, customer or
route information in this repository.
"""
import numpy as np
import pandas as pd

from model.preprocess.preprocess_utils import get_fiscal_calendar

YEAR, WEEK = 2022, 34          # target week used across the tests
BRANCH, ROUTE = "10001", "0500"


def make_calendar() -> pd.DataFrame:
    """Fiscal calendar (52 weeks of 7 days, week 1 starts on a Thursday) for two years."""
    return pd.concat([get_fiscal_calendar(YEAR - 1), get_fiscal_calendar(YEAR)], ignore_index=True)


def _week_bounds(calendar: pd.DataFrame, week: int, year: int):
    days = calendar[(calendar["week_num"] == week) & (calendar["year"] == year)]["Date"]
    return days.min(), days.max()


def make_itinerary(customers, calendar=None, year=YEAR, week=WEEK, visit_days=(0, 3)) -> pd.DataFrame:
    """Visit plan for the target week: each customer is visited on `visit_days` (0 = first day of the week)."""
    calendar = make_calendar() if calendar is None else calendar
    first, _ = _week_bounds(calendar, week, year)
    rows = [
        {"branch_code": BRANCH, "route_code": ROUTE, "customer_code": c,
         "dateVisitPlan": first + pd.Timedelta(days=d)}
        for c in customers for d in visit_days
    ]
    return pd.DataFrame(rows)


def make_sales(customers, products, first_sale_week=1, last_sale_week=WEEK - 1, seed=0,
               calendar=None, year=YEAR) -> pd.DataFrame:
    """Daily sales (`Units_Sold`) for every (customer, product) between two fiscal weeks (inclusive)."""
    rng = np.random.default_rng(seed)
    calendar = make_calendar() if calendar is None else calendar
    start, _ = _week_bounds(calendar, first_sale_week, year)
    _, end = _week_bounds(calendar, last_sale_week, year)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for c in customers:
        for p in products:
            units = rng.poisson(lam=3, size=len(dates)) + 1          # strictly positive daily sales
            rows.append(pd.DataFrame({
                "Date": dates, "Branch": BRANCH, "Branch_Name": "Branch_A", "Channel_Name": "Channel_A",
                "Route": ROUTE, "Customer": c, "Product": p, "Product_Name": f"product_{p}",
                "Units_Sold": units,
            }))
    return pd.concat(rows, ignore_index=True)


def make_pivot_series_weekly(n_series=6, n_weeks=104, seed=0) -> pd.DataFrame:
    """Weekly series pivot (rows = unique_id, columns = week-ending Thursdays) with intermittent demand."""
    rng = np.random.default_rng(seed)
    columns = pd.date_range("2020-01-02", periods=n_weeks, freq="W-THU")
    data = rng.poisson(lam=rng.uniform(1, 6, size=(n_series, 1)), size=(n_series, n_weeks))
    data = data * (rng.random((n_series, n_weeks)) > 0.25)            # ~25 % zero weeks
    index = [f"uid_{BRANCH}_{ROUTE}_{1000 + i}_{i}" for i in range(n_series)]
    return pd.DataFrame(data.astype(float), index=index, columns=columns)
